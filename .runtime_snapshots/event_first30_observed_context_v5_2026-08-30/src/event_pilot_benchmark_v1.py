"""出来事原本v1の共通前半と盤面保存方式を実測する。"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.event_pilot_analysis_v1 import load_completed_run_events


IDENTITY_PLACEHOLDER = "<generated-content>"
ATTEMPT_PLACEHOLDER = "<attempt>"
DEFAULT_REFERENCE_MULTIPLIERS = (1, 2, 5, 10)
PARQUET_COMPRESSION = "zstd"


def compare_common_prefix(partial_run: Path, full_run: Path) -> dict[str, Any]:
    """途中切り処理と通し処理の共通前半を生成ID非依存で比較する。"""

    partial_manifest, partial_events = load_completed_run_events(partial_run)
    full_manifest, full_events = load_completed_run_events(full_run)
    _validate_common_source(partial_manifest["source"], full_manifest["source"])
    cutoff = int(partial_manifest["source"]["processing_end_frame_exclusive"])
    partial_prefix = tuple(_events_before(partial_events, cutoff))
    full_prefix = tuple(_events_before(full_events, cutoff))
    partial_normalized = _normalized_events(
        partial_prefix, partial_manifest["build_id"], partial_manifest["attempt_id"]
    )
    full_normalized = _normalized_events(
        full_prefix, full_manifest["build_id"], full_manifest["attempt_id"]
    )
    mismatch_index = _first_mismatch(partial_normalized, full_normalized)
    return {
        "prefix_match": mismatch_index is None,
        "partial_event_count": len(partial_events),
        "partial_prefix_event_count": len(partial_prefix),
        "partial_events_at_processing_end": len(partial_events) - len(partial_prefix),
        "full_prefix_event_count": len(full_prefix),
        "cutoff_frame_exclusive": cutoff,
        "first_mismatch_index": mismatch_index,
        "partial_prefix_sha256": _rows_sha256(partial_normalized),
        "full_prefix_sha256": _rows_sha256(full_normalized),
    }


def benchmark_board_storage(
    run_dirs: Sequence[Path],
    *,
    repeats: int = 20,
    reference_multipliers: Sequence[int] = DEFAULT_REFERENCE_MULTIPLIERS,
) -> dict[str, Any]:
    """盤面複製と盤面ID参照の容量・JSON読込時間を比較する。"""

    if not run_dirs:
        raise ValueError("保存方式の比較には完了試行が1件以上必要です")
    if repeats < 1:
        raise ValueError("読込反復回数は1以上が必要です")
    events = _load_distinct_events(run_dirs)
    inline_rows, reference_rows, board_rows = _build_storage_rows(events)
    inline_payload = _json_lines(inline_rows)
    reference_payload = _json_lines(reference_rows)
    board_payload = _json_lines(board_rows)
    projections = _storage_projections(
        len(inline_payload), len(reference_payload), len(board_payload), reference_multipliers
    )
    return {
        "event_count": len(events),
        "unique_board_count": len(board_rows),
        "inline_state_bytes": len(inline_payload),
        "reference_state_bytes": len(reference_payload),
        "board_table_bytes": len(board_payload),
        "standalone_reference_bundle_bytes": len(reference_payload) + len(board_payload),
        "read_repeats": repeats,
        "inline_read_median_ms": _median_ms(
            repeats, lambda: _read_inline(inline_payload)
        ),
        "reference_cold_read_median_ms": _median_ms(
            repeats, lambda: _read_reference(reference_payload, board_payload)
        ),
        "reference_cached_read_median_ms": _cached_reference_ms(
            reference_payload, board_payload, repeats
        ),
        "projected_bytes_by_references_per_board": projections,
        "first_reference_count_where_id_bundle_is_smaller": _first_smaller(projections),
    }


def benchmark_parquet_board_storage(
    run_dirs: Sequence[Path],
    *,
    repeats: int = 10,
    reference_multipliers: Sequence[int] = DEFAULT_REFERENCE_MULTIPLIERS,
) -> dict[str, Any]:
    """Parquetで盤面複製と盤面ID参照の容量・読込時間を比較する。"""

    if repeats < 1:
        raise ValueError("読込反復回数は1以上が必要です")
    events = _load_distinct_events(run_dirs)
    pa, parquet = _load_arrow()
    board_table = _arrow_board_table(events, pa)
    board_payload = _parquet_bytes(board_table, parquet, pa)
    projections: dict[str, dict[str, Any]] = {}
    for count in reference_multipliers:
        if count < 1:
            raise ValueError("盤面あたり参照数は1以上が必要です")
        inline_table = _arrow_state_table(events, count, True, pa)
        reference_table = _arrow_state_table(events, count, False, pa)
        inline_payload = _parquet_bytes(inline_table, parquet, pa)
        reference_payload = _parquet_bytes(reference_table, parquet, pa)
        projections[str(count)] = _parquet_projection(
            inline_payload, reference_payload, board_payload, repeats, parquet, pa
        )
    return {
        "event_count": len(events),
        "unique_board_count": board_table.num_rows,
        "compression": PARQUET_COMPRESSION,
        "pyarrow_version": pa.__version__,
        "read_repeats": repeats,
        "board_table_bytes": len(board_payload),
        "by_references_per_board": projections,
        "first_reference_count_where_id_bundle_is_smaller": _first_arrow_smaller(
            projections
        ),
    }


def _validate_common_source(partial: dict[str, Any], full: dict[str, Any]) -> None:
    keys = (
        "source_video_sha256",
        "processing_start_frame",
        "time_base_numerator",
        "time_base_denominator",
    )
    if any(partial[key] != full[key] for key in keys):
        raise ValueError("途中切り処理と通し処理の元映像または時間基準が一致しません")
    if int(partial["processing_end_frame_exclusive"]) >= int(
        full["processing_end_frame_exclusive"]
    ):
        raise ValueError("途中切り処理の終了位置は通し処理より前である必要があります")


def _events_before(
    events: Iterable[dict[str, Any]], cutoff: int
) -> Iterable[dict[str, Any]]:
    for event in events:
        if int(event["timing"]["available_frame"]) < cutoff:
            yield event


def _normalized_events(
    events: Iterable[dict[str, Any]], build_id: str, attempt_id: str
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _replace_identity(event, str(build_id), str(attempt_id)) for event in events
    )


def _replace_identity(value: Any, build_id: str, attempt_id: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_identity(item, build_id, attempt_id) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_identity(item, build_id, attempt_id) for item in value]
    if isinstance(value, str):
        return value.replace(build_id, IDENTITY_PLACEHOLDER).replace(
            attempt_id, ATTEMPT_PLACEHOLDER
        )
    return value


def _first_mismatch(
    left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]
) -> int | None:
    for index, (left_row, right_row) in enumerate(zip(left, right, strict=False)):
        if left_row != right_row:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _rows_sha256(rows: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256(_json_lines(rows)).hexdigest()


def _load_distinct_events(run_dirs: Sequence[Path]) -> tuple[dict[str, Any], ...]:
    events: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    for run_dir in run_dirs:
        resolved = str(run_dir.resolve())
        if resolved in seen_runs:
            raise ValueError("同じ完了試行を容量比較へ重複指定できません")
        seen_runs.add(resolved)
        _, loaded = load_completed_run_events(run_dir)
        events.extend(loaded)
    return tuple(events)


def _build_storage_rows(
    events: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inline_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    board_rows: list[dict[str, Any]] = []
    seen_boards: set[str] = set()
    for event in events:
        payload = event["payload"]
        board_id = str(payload["board_observation_id"])
        if board_id in seen_boards:
            raise ValueError(f"盤面IDが重複しています: {board_id}")
        seen_boards.add(board_id)
        base = {"event_id": event["event_id"], "board_observation_id": board_id}
        board = {"grid": payload["grid"], "unknown_mask": payload["unknown_mask"]}
        inline_rows.append({**base, **board})
        reference_rows.append(base)
        board_rows.append({"board_observation_id": board_id, **board})
    return inline_rows, reference_rows, board_rows


def _json_lines(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _storage_projections(
    inline_bytes: int, reference_bytes: int, board_bytes: int, multipliers: Sequence[int]
) -> dict[str, dict[str, int]]:
    projections: dict[str, dict[str, int]] = {}
    for count in multipliers:
        if count < 1:
            raise ValueError("盤面あたり参照数は1以上が必要です")
        duplicated = inline_bytes * count
        referenced = board_bytes + reference_bytes * count
        projections[str(count)] = {
            "duplicated_grid_bytes": duplicated,
            "id_reference_bundle_bytes": referenced,
            "id_reference_savings_bytes": duplicated - referenced,
        }
    return projections


def _first_smaller(projections: dict[str, dict[str, int]]) -> int | None:
    for count in sorted(int(key) for key in projections):
        if projections[str(count)]["id_reference_savings_bytes"] > 0:
            return count
    return None


def _load_arrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("Parquet実測にはpyarrowが必要です") from error
    return pa, parquet


def _arrow_board_table(events: Sequence[dict[str, Any]], pa: Any) -> Any:
    board_ids: list[str] = []
    grids: list[list[int]] = []
    masks: list[list[int]] = []
    for event in events:
        payload = event["payload"]
        board_ids.append(str(payload["board_observation_id"]))
        grids.append(_flatten(payload["grid"]))
        masks.append(_flatten(payload["unknown_mask"]))
    return pa.table(
        {
            "board_observation_id": pa.array(board_ids),
            "grid": pa.array(grids, type=pa.list_(pa.int8(), 78)),
            "unknown_mask": pa.array(masks, type=pa.list_(pa.int8(), 78)),
        }
    )


def _arrow_state_table(
    events: Sequence[dict[str, Any]], multiplier: int, inline: bool, pa: Any
) -> Any:
    event_ids: list[str] = []
    board_ids: list[str] = []
    grids: list[list[int]] = []
    masks: list[list[int]] = []
    for use_index in range(multiplier):
        for event in events:
            payload = event["payload"]
            event_ids.append(f"{event['event_id']}:use-{use_index}")
            board_ids.append(str(payload["board_observation_id"]))
            if inline:
                grids.append(_flatten(payload["grid"]))
                masks.append(_flatten(payload["unknown_mask"]))
    values: dict[str, Any] = {"event_id": event_ids, "board_observation_id": board_ids}
    if inline:
        values["grid"] = pa.array(grids, type=pa.list_(pa.int8(), 78))
        values["unknown_mask"] = pa.array(masks, type=pa.list_(pa.int8(), 78))
    return pa.table(values)


def _flatten(rows: Sequence[Sequence[int]]) -> list[int]:
    return [int(value) for row in rows for value in row]


def _parquet_bytes(table: Any, parquet: Any, pa: Any) -> bytes:
    sink = pa.BufferOutputStream()
    parquet.write_table(table, sink, compression=PARQUET_COMPRESSION)
    return sink.getvalue().to_pybytes()


def _parquet_projection(
    inline_payload: bytes,
    reference_payload: bytes,
    board_payload: bytes,
    repeats: int,
    parquet: Any,
    pa: Any,
) -> dict[str, Any]:
    referenced = len(reference_payload) + len(board_payload)
    board_lookup = _read_arrow_board_map(board_payload, parquet, pa)
    return {
        "duplicated_grid_bytes": len(inline_payload),
        "id_reference_state_bytes": len(reference_payload),
        "id_reference_bundle_bytes": referenced,
        "id_reference_savings_bytes": len(inline_payload) - referenced,
        "inline_read_median_ms": _median_ms(
            repeats, lambda: _read_arrow_inline(inline_payload, parquet, pa)
        ),
        "reference_cold_read_median_ms": _median_ms(
            repeats,
            lambda: _read_arrow_reference(reference_payload, board_payload, parquet, pa),
        ),
        "reference_cached_read_median_ms": _median_ms(
            repeats,
            lambda: _read_arrow_reference_cached(reference_payload, board_lookup, parquet, pa),
        ),
    }


def _read_arrow_table(payload: bytes, parquet: Any, pa: Any) -> Any:
    return parquet.read_table(pa.BufferReader(payload))


def _read_arrow_inline(payload: bytes, parquet: Any, pa: Any) -> int:
    table = _read_arrow_table(payload, parquet, pa)
    return int(table.column("grid").chunk(0)[0].values[0].as_py()) if table.num_rows else 0


def _read_arrow_board_map(payload: bytes, parquet: Any, pa: Any) -> dict[str, Any]:
    table = _read_arrow_table(payload, parquet, pa)
    ids = table.column("board_observation_id").to_pylist()
    grids = table.column("grid").to_pylist()
    return dict(zip(ids, grids, strict=True))


def _read_arrow_reference(
    reference_payload: bytes, board_payload: bytes, parquet: Any, pa: Any
) -> int:
    boards = _read_arrow_board_map(board_payload, parquet, pa)
    return _read_arrow_reference_cached(reference_payload, boards, parquet, pa)


def _read_arrow_reference_cached(
    reference_payload: bytes, boards: dict[str, Any], parquet: Any, pa: Any
) -> int:
    table = _read_arrow_table(reference_payload, parquet, pa)
    board_ids = table.column("board_observation_id").to_pylist()
    return sum(int(boards[str(board_id)][0]) for board_id in board_ids)


def _first_arrow_smaller(projections: dict[str, dict[str, Any]]) -> int | None:
    for count in sorted(int(key) for key in projections):
        if int(projections[str(count)]["id_reference_savings_bytes"]) > 0:
            return count
    return None


def _median_ms(repeats: int, operation: Any) -> float:
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        operation()
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
    return round(statistics.median(durations), 6)


def _read_inline(payload: bytes) -> int:
    rows = [json.loads(line) for line in payload.splitlines()]
    return sum(int(row["grid"][0][0]) for row in rows)


def _board_map(payload: bytes) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in payload.splitlines()]
    return {str(row["board_observation_id"]): row for row in rows}


def _read_reference(reference_payload: bytes, board_payload: bytes) -> int:
    boards = _board_map(board_payload)
    rows = [json.loads(line) for line in reference_payload.splitlines()]
    return sum(int(boards[str(row["board_observation_id"])]["grid"][0][0]) for row in rows)


def _cached_reference_ms(reference_payload: bytes, board_payload: bytes, repeats: int) -> float:
    boards = _board_map(board_payload)

    def operation() -> int:
        rows = [json.loads(line) for line in reference_payload.splitlines()]
        return sum(
            int(boards[str(row["board_observation_id"])]["grid"][0][0]) for row in rows
        )

    return _median_ms(repeats, operation)


__all__ = [
    "benchmark_board_storage",
    "benchmark_parquet_board_storage",
    "compare_common_prefix",
]
