"""完成済み出来事原本から動画単位の学習用Parquet三表を生成する。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.event_exchange_cross_validation_v1 import (
    ExchangeCrossValidationReportV1,
    audit_exchange_cross_quarantine_v1,
    cross_validate_exchange_events_v1,
    online_uncertain_quarantine_starts_v1,
    unsupported_high_confidence_game_indices_v1,
)
from src.event_learning_tables_v1 import (
    LEARNING_TABLE_VERSION,
    LearningTables,
    build_learning_tables,
    load_completed_run_batches,
)


PARQUET_COMPRESSION = "zstd"
HASH_CHUNK_BYTES = 1024 * 1024
ROOT_COMPLETE_VERSION = "event-learning-root-complete/v1"
VIDEO_COMPLETE_VERSION = "event-learning-video-complete/v1"


class LearningTableBuildError(ValueError):
    """学習表の一括生成条件を満たさない場合の例外。"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target", action="append", default=[])
    return parser.parse_args()


def _load_pilot_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        target = row.get("target_id", "")
        if not _is_development_row(row) or not target:
            continue
        if target in selected:
            raise LearningTableBuildError(f"pilot対象が重複しています: {target}")
        selected[target] = _normalized_development_row(row)
    if not selected:
        raise LearningTableBuildError("pilot対象がありません")
    return selected


def _is_development_row(row: Mapping[str, str]) -> bool:
    return (
        row.get("subset") == "pilot"
        or row.get("role") == "development_increment_only"
    )


def _normalized_development_row(row: Mapping[str, str]) -> dict[str, str]:
    value = dict(row)
    value["source_group_id"] = str(
        row.get("source_group_id") or row.get("youtube_id") or ""
    )
    target = str(row.get("target_id") or "")
    value["source_video_id"] = str(
        row.get("canonical_video_alias") or f"video_{target}"
    )
    return value


def _select_targets(
    pilot: Mapping[str, dict[str, str]], requested: Sequence[str],
) -> list[str]:
    targets = sorted(set(requested)) if requested else sorted(pilot)
    missing = set(targets) - set(pilot)
    if missing:
        raise LearningTableBuildError(f"pilot外の対象です: {sorted(missing)}")
    return targets


def _find_run(
    run_root: Path, target: str, source: Mapping[str, str] | None = None,
) -> Path:
    video_id = str((source or {}).get("source_video_id") or f"video_{target}")
    candidates = sorted(
        path.parent for path in run_root.glob(
            f"schema=v1/video={video_id}/build=*/attempt=*/COMPLETE"
        )
    )
    if len(candidates) != 1:
        raise LearningTableBuildError(
            f"{target}の完成原本は1件必要です: {len(candidates)}件"
        )
    return candidates[0]


def _prepare_output_root(path: Path) -> None:
    if path.exists():
        raise LearningTableBuildError(f"出力先は新規でなければなりません: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _write_video(
    output_root: Path, target: str, pilot_row: Mapping[str, str], run_dir: Path,
) -> dict[str, Any]:
    fold = _positive_int(pilot_row.get("fold"), "fold")
    source_video_id = str(pilot_row.get("source_video_id") or f"video_{target}")
    batches = load_completed_run_batches(run_dir)
    events = tuple(event for batch in batches for event in batch.events)
    _require_expected_source_video(events, source_video_id, run_dir)
    cross = cross_validate_exchange_events_v1(events)
    cross_audit = audit_exchange_cross_quarantine_v1(cross)
    if cross_audit["contract_pass"] is not True:
        raise LearningTableBuildError("物理着地の行単位隔離契約が不合格です")
    tables = build_learning_tables(
        batches, fold=fold, tier=str(pilot_row.get("tier", "")),
        source_group_id=str(pilot_row.get("source_group_id", "")),
        context_quarantined_segments=unsupported_high_confidence_game_indices_v1(cross),
        online_context_quarantine_starts=(
            online_uncertain_quarantine_starts_v1(cross)
        ),
    )
    video_dir = output_root / "schema=v1" / f"video={source_video_id}"
    video_dir.mkdir(parents=True, exist_ok=False)
    paths = _write_parquet_tables(video_dir, tables)
    summary = _video_summary(
        target, fold, pilot_row, run_dir, tables, paths, cross, cross_audit,
    )
    manifest = video_dir / "manifest.json"
    _write_json_exclusive(manifest, summary)
    _write_json_exclusive(
        video_dir / "COMPLETE",
        {"format_version": VIDEO_COMPLETE_VERSION, "manifest_sha256": _sha256_file(manifest)},
    )
    return summary


def _require_expected_source_video(
    events: Sequence[Mapping[str, Any]], source_video_id: str, run_dir: Path,
) -> None:
    """読込済み全出来事の来歴IDが期待するID一個だけかを表生成前に検査する。"""

    if not events:
        raise LearningTableBuildError(f"完成原本に出来事がありません: {run_dir}")
    loaded = {str(event.get("source_video_id") or "") for event in events}
    if loaded != {source_video_id}:
        raise LearningTableBuildError(
            f"原本の来歴IDが期待と一致しません: 期待={source_video_id} "
            f"実際={sorted(loaded)} 原本={run_dir}"
        )


def _write_parquet_tables(
    video_dir: Path, tables: LearningTables,
) -> dict[str, Path]:
    pa, parquet = _load_arrow()
    rows_by_name = {
        "states": tables.states, "labels": tables.labels,
        "event_links": tables.event_links,
    }
    schemas = {
        "states": _state_schema(pa), "labels": _label_schema(pa),
        "event_links": _event_link_schema(pa),
    }
    paths: dict[str, Path] = {}
    for name, rows in rows_by_name.items():
        path = video_dir / f"{name}.parquet"
        table = pa.Table.from_pylist(list(rows), schema=schemas[name])
        parquet.write_table(table, path, compression=PARQUET_COMPRESSION)
        paths[name] = path
    return paths


def _state_schema(pa: Any) -> Any:
    fields = [
        pa.field("schema_version", pa.string()), pa.field("state_id", pa.string()),
        pa.field("source_video_id", pa.string()), pa.field("source_group_id", pa.string()),
        pa.field("build_id", pa.string()),
        pa.field("partition_fold", pa.int8()), pa.field("tier", pa.string()),
        pa.field("availability_batch_id", pa.string()),
        pa.field("through_sequence", pa.int64()), pa.field("batch_first_sequence", pa.int64()),
        pa.field("online_segment_index", pa.int32()),
        pa.field("available_frame", pa.int64()),
        pa.field("available_ms", pa.int64()), pa.field("trigger_event_count", pa.int32()),
    ]
    for side in ("p1", "p2"):
        fields.extend(_state_side_schema(pa, side))
    fields.extend(_state_quality_schema(pa))
    return pa.schema(fields)


def _state_side_schema(pa: Any, side: str) -> list[Any]:
    return [
        pa.field(f"a_{side}_grid", pa.list_(pa.int8(), 78)),
        pa.field(f"a_{side}_unknown_mask", pa.list_(pa.int8(), 78)),
        *[pa.field(f"a_{side}_{name}", pa.int16()) for name in (
            "color_count", "garbage_count", "occupied_count", "unknown_count",
        )],
        pa.field(f"a_{side}_score", pa.int64()),
        *[pa.field(f"a_{side}_{name}", pa.int8()) for name in (
            "next_first", "next_second", "double_next_first", "double_next_second",
        )],
        pa.field(f"a_{side}_tsumo_count", pa.int64()),
        pa.field(f"a_{side}_all_clear_pending", pa.bool_()),
        pa.field(f"a_{side}_stable_confidence", pa.int64()),
        pa.field(f"a_{side}_chain_mechanism", pa.string()),
        pa.field(f"quality_{side}_match_end_locked", pa.bool_()),
        pa.field(f"quality_{side}_post_match_lockdown", pa.bool_()),
        pa.field(f"quality_{side}_missing_fields", pa.list_(pa.string())),
        pa.field(f"b_{side}_pending_garbage", pa.int64()),
        pa.field(f"b_{side}_chain_active", pa.bool_()),
        pa.field(f"b_{side}_all_clear_event_state", pa.bool_()),
        pa.field(f"b_{side}_provisional_generated", pa.int64()),
        pa.field(f"b_{side}_provisional_score", pa.int64()),
        pa.field(f"b_{side}_provisional_chain_count", pa.int16()),
        pa.field(f"b_{side}_causal_pending_garbage", pa.int64()),
        pa.field(f"b_{side}_causal_effective_rate", pa.int64()),
    ]


def _state_quality_schema(pa: Any) -> list[Any]:
    return [
        pa.field("a_input_usable", pa.bool_()),
        pa.field("a_reason_codes", pa.list_(pa.string())),
        pa.field("b_input_usable", pa.bool_()),
        pa.field("b_reason_codes", pa.list_(pa.string())),
        pa.field("c_input_usable", pa.bool_()),
        pa.field("c_reason_codes", pa.list_(pa.string())),
        pa.field("quality_same_frame_ambiguous_amount", pa.int64()),
        pa.field("quality_boundary_expiry_gap", pa.int64()),
        pa.field("quality_disallowed_event_count", pa.int32()),
        pa.field("quality_physical_accounting_unsupported_segment", pa.bool_()),
        pa.field("b_causal_exchange_usable", pa.bool_()),
        pa.field("b_causal_observed_attack_balance", pa.int64()),
        pa.field("quality_causal_quarantined_finalized_count", pa.int32()),
        pa.field("quality_causal_quarantined_finalized_amount", pa.int64()),
        pa.field("quality_causal_quarantined_landing_count", pa.int32()),
        pa.field("quality_causal_quarantined_landing_amount", pa.int64()),
        pa.field("quality_causal_ledger_pending_disagreement", pa.int64()),
    ]


def _label_schema(pa: Any) -> Any:
    return pa.schema([
        pa.field("schema_version", pa.string()), pa.field("state_id", pa.string()),
        pa.field("source_video_id", pa.string()), pa.field("source_group_id", pa.string()),
        pa.field("partition_fold", pa.int8()),
        pa.field("a_input_usable", pa.bool_()), pa.field("b_input_usable", pa.bool_()),
        pa.field("c_input_usable", pa.bool_()), pa.field("game_key", pa.string()),
        pa.field("official_game_number", pa.int32()), pa.field("winner_side", pa.string()),
        pa.field("p1_won", pa.bool_()), pa.field("official_assignment_event_id", pa.string()),
        pa.field("official_winner_event_id", pa.string()),
        pa.field("win_label_available", pa.bool_()),
        pa.field("label_reason_codes", pa.list_(pa.string())),
        pa.field("evaluation_scope_usable", pa.bool_()),
        pa.field("evaluation_scope_reason_codes", pa.list_(pa.string())),
        pa.field("sample_weight", pa.float64()), pa.field("training_usable", pa.bool_()),
    ])


def _event_link_schema(pa: Any) -> Any:
    return pa.schema([
        pa.field("schema_version", pa.string()), pa.field("state_id", pa.string()),
        pa.field("event_id", pa.string()), pa.field("event_sequence", pa.int64()),
        pa.field("event_type", pa.string()), pa.field("side", pa.string()),
        pa.field("availability_batch_id", pa.string()),
        pa.field("model_input_allowed", pa.bool_()),
    ])


def _video_summary(
    target: str, fold: int, pilot_row: Mapping[str, str], run_dir: Path,
    tables: LearningTables, paths: Mapping[str, Path],
    cross: ExchangeCrossValidationReportV1, cross_audit: Mapping[str, Any],
) -> dict[str, Any]:
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    game_keys = {row["game_key"] for row in tables.labels if row["game_key"] is not None}
    usable = [row for row in tables.labels if row["training_usable"]]
    return {
        "schema_version": LEARNING_TABLE_VERSION, "target_id": target,
        "source_video_id": pilot_row.get("source_video_id", f"video_{target}"),
        "partition_fold": fold,
        "source_group_id": pilot_row.get("source_group_id", ""),
        "tier": pilot_row.get("tier", ""), "source_build_id": run_manifest["build_id"],
        "source_semantic_sha256": run_manifest["semantic_sha256"],
        "state_count": len(tables.states), "label_count": len(tables.labels),
        "event_link_count": len(tables.event_links), "game_count": len(game_keys),
        "training_usable_count": len(usable),
        "usable_weight_sum": sum(float(row["sample_weight"]) for row in usable),
        "physical_cross": _physical_cross_summary(cross, cross_audit, tables),
        "state_id_sha256": _logical_sha256([row["state_id"] for row in tables.states]),
        "pyarrow_version": _load_arrow()[0].__version__,
        "tables": {
            name: _file_entry(path, getattr(tables, name))
            for name, path in sorted(paths.items())
        },
    }


def _physical_cross_summary(
    cross: ExchangeCrossValidationReportV1, audit: Mapping[str, Any],
    tables: LearningTables,
) -> dict[str, Any]:
    marked = {
        int(row["online_segment_index"]) for row in tables.states
        if row["quality_physical_accounting_unsupported_segment"] is True
    }
    expected = set(audit["unsupported_game_indices"])
    state_games = {int(row["online_segment_index"]) for row in tables.states}
    expected_with_states = expected & state_games
    return {
        "quarantine_contract_pass": audit["contract_pass"] is True,
        "all_physical_candidates_explained": cross.physical_gate_pass,
        "high_confidence_candidate_count": cross.high_confidence_candidate_count,
        "high_confidence_unsupported_count": cross.high_confidence_unsupported_count,
        "unsupported_game_count": len(expected),
        "unsupported_game_indices": sorted(expected),
        "marked_state_game_count": len(marked),
        "unsupported_game_without_state_count": len(expected - state_games),
        "all_unsupported_games_marked": marked == expected_with_states,
        "allocated_supply_overuse_count": cross.allocated_supply_overuse_count,
        "provisional_used_in_confirmed_balance_count": (
            cross.provisional_used_in_confirmed_balance_count
        ),
    }


def _file_entry(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "name": path.name, "byte_count": path.stat().st_size,
        "sha256": _sha256_file(path), "logical_sha256": _logical_sha256(rows),
    }


def _logical_sha256(rows: Sequence[Any]) -> str:
    hasher = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            row, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)
    return hasher.hexdigest()


def _write_root_complete(output_root: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    summary_path = output_root / "SUMMARY.json"
    _write_json_exclusive(summary_path, {
        "schema_version": LEARNING_TABLE_VERSION,
        "video_count": len(summaries),
        "physical_cross": _root_physical_cross_summary(summaries),
        "videos": list(summaries),
    })
    _write_json_exclusive(output_root / "COMPLETE", {
        "format_version": ROOT_COMPLETE_VERSION,
        "summary_sha256": _sha256_file(summary_path),
    })


def _root_physical_cross_summary(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [summary["physical_cross"] for summary in summaries]
    return {
        "quarantine_contract_pass_count": sum(
            row["quarantine_contract_pass"] is True for row in rows
        ),
        "all_physical_candidates_explained_count": sum(
            row["all_physical_candidates_explained"] is True for row in rows
        ),
        "high_confidence_candidate_count": sum(
            int(row["high_confidence_candidate_count"]) for row in rows
        ),
        "high_confidence_unsupported_count": sum(
            int(row["high_confidence_unsupported_count"]) for row in rows
        ),
        "unsupported_game_count": sum(int(row["unsupported_game_count"]) for row in rows),
        "all_unsupported_games_marked": all(
            row["all_unsupported_games_marked"] is True for row in rows
        ),
        "allocated_supply_overuse_count": sum(
            int(row["allocated_supply_overuse_count"]) for row in rows
        ),
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
    ) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def _positive_int(value: str | None, name: str) -> int:
    try:
        number = int(value or "")
    except ValueError as error:
        raise LearningTableBuildError(f"{name}が整数ではありません") from error
    if number <= 0:
        raise LearningTableBuildError(f"{name}は正でなければなりません")
    return number


def _load_arrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("学習表生成にはpyarrowが必要です") from error
    return pa, parquet


def main() -> None:
    args = _parse_args()
    pilot = _load_pilot_rows(args.pilot_manifest)
    targets = _select_targets(pilot, args.target)
    _prepare_output_root(args.output_root)
    summaries = [
        _write_video(
            args.output_root, target, pilot[target],
            _find_run(args.run_root, target, pilot[target]),
        )
        for target in targets
    ]
    _write_root_complete(args.output_root, summaries)
    print(json.dumps({
        "output_root": str(args.output_root), "video_count": len(summaries),
        "state_count": sum(int(item["state_count"]) for item in summaries),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
