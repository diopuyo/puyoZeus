"""確定盤面スナップショットを出来事原本v1へ変換する既定OFFアダプター。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.event_source_v1 import SCHEMA_VERSION


ADAPTER_VERSION = "stable-board-npz-adapter/v1.2"
BOARD_ROWS = 13
BOARD_COLUMNS = 6
COLOR_UNKNOWN = 10
GARBAGE_VALUE = 9
ALLOWED_CELL_VALUES = frozenset({0, 1, 2, 3, 4, 5, GARBAGE_VALUE, COLOR_UNKNOWN})
SIDE_ORDER = {"1P": 0, "2P": 1}
SIDE_VALUES = {"1P": "p1", "2P": "p2"}
REQUIRED_NPZ_KEYS = frozenset({"grids", "video_id", "side", "frame_idx", "board_provenance"})
OPTIONAL_OBSERVATION_NPZ_KEYS = frozenset(
    {
        "score", "next1_a", "next1_b", "dnext_a", "dnext_b", "tsumo_count",
        "all_clear_pending", "stable_persistence_confidence", "chain_mechanism",
        "match_end_locked", "post_match_lockdown_active",
    }
)


class SnapshotAdapterError(ValueError):
    """盤面スナップショットを正確に変換できない場合の例外。"""


@dataclass(frozen=True, slots=True)
class VideoTimeBase:
    """1フレームの秒数を分数で表す。"""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        _require_integer(self.numerator, "時刻基準の分子")
        _require_integer(self.denominator, "時刻基準の分母")
        if self.numerator <= 0 or self.denominator <= 0:
            raise SnapshotAdapterError("時刻基準の分子と分母は正でなければなりません")

    def frame_to_ms(self, frame_idx: int) -> int:
        _require_integer(frame_idx, "フレーム番号")
        if frame_idx < 0:
            raise SnapshotAdapterError("フレーム番号は0以上でなければなりません")
        return frame_idx * self.numerator * 1000 // self.denominator


@dataclass(frozen=True, slots=True)
class StableBoardObservation:
    """確定盤面と同じ行で当時利用できた盤面外観測。"""

    score: int | None
    next_pair: tuple[int | None, int | None] | None
    double_next_pair: tuple[int | None, int | None] | None
    tsumo_count: int | None
    all_clear_pending: bool | None
    stable_persistence_confidence: int | None
    chain_mechanism: str | None
    match_end_locked: bool | None
    post_match_lockdown_active: bool | None


@dataclass(frozen=True, slots=True)
class StableBoardSnapshot:
    """収集中に利用可能になった一側の確定盤面。"""

    source_video_id: str
    frame_idx: int
    side: str
    grid: tuple[tuple[int, ...], ...]
    board_provenance: str
    observation: StableBoardObservation | None = None


@dataclass(frozen=True, slots=True)
class SnapshotSelection:
    """NPZ全行から実観測盤面を選別した結果と母数。"""

    snapshots: tuple[StableBoardSnapshot, ...]
    input_row_count: int
    observed_row_count: int
    excluded_provenance_counts: tuple[tuple[str, int], ...]


def load_observed_stable_snapshots(npz_path: Path) -> tuple[StableBoardSnapshot, ...]:
    """新しい収集形式から実観測の確定盤面だけを読み出す。"""

    return load_stable_snapshot_selection(npz_path).snapshots


def load_stable_snapshot_selection(npz_path: Path) -> SnapshotSelection:
    """実観測盤面と、除外した物理推定盤面の母数を読み出す。"""

    with np.load(npz_path, allow_pickle=True) as data:
        missing = REQUIRED_NPZ_KEYS - set(data.files)
        if missing:
            raise SnapshotAdapterError(f"必須NPZ列がありません: {sorted(missing)}")
        selected_keys = REQUIRED_NPZ_KEYS | (OPTIONAL_OBSERVATION_NPZ_KEYS & set(data.files))
        columns = {key: np.asarray(data[key]) for key in selected_keys}
        row_count = _validate_npz_lengths(columns)
        video_ids = _video_id_values(columns["video_id"], row_count)
        provenances = _provenance_values(columns["board_provenance"], row_count)
        observed_indices = tuple(
            index for index, value in enumerate(provenances) if value == "observed"
        )
        snapshots = tuple(
            _snapshot_from_row(columns, video_ids, provenances, index)
            for index in observed_indices
        )
    _validate_snapshot_sequence(snapshots)
    excluded = Counter(value for value in provenances if value != "observed")
    return SnapshotSelection(
        snapshots,
        row_count,
        len(snapshots),
        tuple(sorted(excluded.items())),
    )


def build_stable_board_batches(
    snapshots: Sequence[StableBoardSnapshot],
    *,
    build_id: str,
    attempt_id: str,
    time_base: VideoTimeBase,
    occurrence_start_frame: int = 0,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """同じ公開フレームの両側盤面を一つの原子的な一括更新へまとめる。"""

    if not snapshots:
        raise SnapshotAdapterError("確定盤面スナップショットがありません")
    _require_integer(occurrence_start_frame, "物理発生範囲の開始フレーム")
    if occurrence_start_frame < 0:
        raise SnapshotAdapterError("物理発生範囲の開始フレームは0以上でなければなりません")
    _validate_snapshot_sequence(snapshots)
    ordered = sorted(snapshots, key=lambda item: (item.frame_idx, SIDE_ORDER[item.side]))
    if ordered[0].frame_idx < occurrence_start_frame:
        raise SnapshotAdapterError("盤面フレームが物理発生範囲の開始より前です")
    groups = _group_by_frame(ordered)
    batches: list[tuple[dict[str, Any], ...]] = []
    previous_by_side: dict[str, int] = {}
    seq = 0
    for frame_idx, group in groups:
        batch_id = f"{build_id}:batch-frame-{frame_idx:012d}"
        batch = tuple(
            _snapshot_event(
                item, seq + index, batch_id, len(group), index, build_id, attempt_id,
                time_base, _occurrence_earliest(item, previous_by_side, occurrence_start_frame),
            )
            for index, item in enumerate(group)
        )
        batches.append(batch)
        for item in group:
            previous_by_side[item.side] = item.frame_idx
        seq += len(batch)
    return tuple(batches)


def _validate_npz_lengths(data: Mapping[str, np.ndarray]) -> int:
    grids = np.asarray(data["grids"])
    if grids.ndim != 3 or grids.shape[1:] != (BOARD_ROWS, BOARD_COLUMNS):
        raise SnapshotAdapterError("grids は (N,13,6) でなければなりません")
    row_count = int(grids.shape[0])
    for key in set(data) - {"grids", "video_id"}:
        values = np.asarray(data[key])
        if values.ndim != 1 or len(values) != row_count:
            raise SnapshotAdapterError(f"{key} の行数がgridsと一致しません")
    return row_count


def _video_id_values(values: np.ndarray, row_count: int) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim == 0:
        video_id = _text_value(array.item(), "video_id")
        return (video_id,) * row_count
    if array.ndim != 1 or len(array) != row_count:
        raise SnapshotAdapterError("video_id の行数がgridsと一致しません")
    return tuple(_text_value(value, "video_id") for value in array)


def _provenance_values(values: np.ndarray, row_count: int) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != row_count:
        raise SnapshotAdapterError("board_provenance の行数がgridsと一致しません")
    return tuple(_text_value(value, "board_provenance") for value in array)


def _snapshot_from_row(
    data: Mapping[str, np.ndarray],
    video_ids: Sequence[str],
    provenances: Sequence[str],
    index: int,
) -> StableBoardSnapshot:
    grid_array = np.asarray(data["grids"][index])
    grid = tuple(
        tuple(_require_integer(value, "盤面セル") for value in row) for row in grid_array
    )
    snapshot = StableBoardSnapshot(
        source_video_id=video_ids[index],
        frame_idx=_require_integer(data["frame_idx"][index], "frame_idx"),
        side=_text_value(data["side"][index], "side"),
        grid=grid,
        board_provenance=provenances[index],
        observation=_observation_from_row(data, index),
    )
    _validate_snapshot(snapshot)
    return snapshot


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise SnapshotAdapterError(f"{name} は整数でなければなりません")
    return int(value)


def _text_value(value: Any, name: str) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        try:
            text = bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SnapshotAdapterError(f"{name} はUTF-8文字列でなければなりません") from exc
    elif isinstance(value, (str, np.str_)):
        text = str(value)
    else:
        raise SnapshotAdapterError(f"{name} は文字列でなければなりません")
    if not text:
        raise SnapshotAdapterError(f"{name} は空にできません")
    return text


def _observation_from_row(
    data: Mapping[str, np.ndarray], index: int,
) -> StableBoardObservation | None:
    if not (OPTIONAL_OBSERVATION_NPZ_KEYS & set(data)):
        return None
    return StableBoardObservation(
        score=_optional_nonnegative_int(data, "score", index),
        next_pair=_optional_color_pair(data, "next1_a", "next1_b", index),
        double_next_pair=_optional_color_pair(data, "dnext_a", "dnext_b", index),
        tsumo_count=_optional_nonnegative_int(data, "tsumo_count", index),
        all_clear_pending=_optional_bool(data, "all_clear_pending", index),
        stable_persistence_confidence=_optional_nonnegative_int(
            data, "stable_persistence_confidence", index,
        ),
        chain_mechanism=_optional_text(data, "chain_mechanism", index),
        match_end_locked=_optional_bool(data, "match_end_locked", index),
        post_match_lockdown_active=_optional_bool(
            data, "post_match_lockdown_active", index,
        ),
    )


def _optional_nonnegative_int(
    data: Mapping[str, np.ndarray], key: str, index: int,
) -> int | None:
    if key not in data:
        return None
    value = _require_integer(data[key][index], key)
    return value if value >= 0 else None


def _optional_color_pair(
    data: Mapping[str, np.ndarray], first: str, second: str, index: int,
) -> tuple[int | None, int | None] | None:
    if first not in data and second not in data:
        return None
    return (
        _optional_color(data, first, index),
        _optional_color(data, second, index),
    )


def _optional_color(
    data: Mapping[str, np.ndarray], key: str, index: int,
) -> int | None:
    if key not in data:
        return None
    value = _require_integer(data[key][index], key)
    if value in {1, 2, 3, 4, 5}:
        return value
    if value in {-1, 0, GARBAGE_VALUE, COLOR_UNKNOWN}:
        return None
    raise SnapshotAdapterError(f"{key} に未知の色値があります: {value}")


def _optional_bool(
    data: Mapping[str, np.ndarray], key: str, index: int,
) -> bool | None:
    if key not in data:
        return None
    value = _require_integer(data[key][index], key)
    if value in {0, 1}:
        return bool(value)
    if value == -1:
        return None
    raise SnapshotAdapterError(f"{key} は0、1、-1のいずれかでなければなりません")


def _optional_text(
    data: Mapping[str, np.ndarray], key: str, index: int,
) -> str | None:
    if key not in data:
        return None
    raw = data[key][index]
    if isinstance(raw, (bytes, np.bytes_)):
        try:
            value = bytes(raw).decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise SnapshotAdapterError(f"{key} はUTF-8文字列でなければなりません") from exc
    elif isinstance(raw, (str, np.str_)):
        value = str(raw).strip()
    else:
        raise SnapshotAdapterError(f"{key} は文字列でなければなりません")
    return None if value.lower() in {"", "nan", "none"} else value


def _validate_snapshot(snapshot: StableBoardSnapshot) -> None:
    if not snapshot.source_video_id:
        raise SnapshotAdapterError("元映像IDは空にできません")
    if snapshot.side not in SIDE_VALUES:
        raise SnapshotAdapterError("side は1Pまたは2Pでなければなりません")
    if snapshot.frame_idx < 0:
        raise SnapshotAdapterError("フレーム番号は0以上でなければなりません")
    if snapshot.board_provenance != "observed":
        raise SnapshotAdapterError("実観測でない盤面はstable_board_observedへ変換できません")
    if len(snapshot.grid) != BOARD_ROWS or any(len(row) != BOARD_COLUMNS for row in snapshot.grid):
        raise SnapshotAdapterError("盤面は13行6列でなければなりません")
    if any(value not in ALLOWED_CELL_VALUES for row in snapshot.grid for value in row):
        raise SnapshotAdapterError("盤面に許可されていないセル値があります")


def _validate_snapshot_sequence(snapshots: Sequence[StableBoardSnapshot]) -> None:
    identities = {snapshot.source_video_id for snapshot in snapshots}
    if len(identities) > 1:
        raise SnapshotAdapterError("異なる元映像の盤面を同じ実行へ混在できません")
    seen: set[tuple[int, str]] = set()
    for snapshot in snapshots:
        _validate_snapshot(snapshot)
        key = (snapshot.frame_idx, snapshot.side)
        if key in seen:
            raise SnapshotAdapterError("同じフレーム・同じ側の盤面が重複しています")
        seen.add(key)


def _group_by_frame(
    snapshots: Sequence[StableBoardSnapshot],
) -> tuple[tuple[int, tuple[StableBoardSnapshot, ...]], ...]:
    groups: list[tuple[int, tuple[StableBoardSnapshot, ...]]] = []
    current_frame = -1
    current: list[StableBoardSnapshot] = []
    for snapshot in snapshots:
        if current and snapshot.frame_idx != current_frame:
            groups.append((current_frame, tuple(current)))
            current = []
        current_frame = snapshot.frame_idx
        current.append(snapshot)
    if current:
        groups.append((current_frame, tuple(current)))
    return tuple(groups)


def _snapshot_event(
    snapshot: StableBoardSnapshot,
    seq: int,
    batch_id: str,
    batch_size: int,
    batch_index: int,
    build_id: str,
    attempt_id: str,
    time_base: VideoTimeBase,
    occurred_earliest_frame: int,
) -> dict[str, Any]:
    available_ms = time_base.frame_to_ms(snapshot.frame_idx)
    board_id = f"{build_id}:board-{SIDE_VALUES[snapshot.side]}-{snapshot.frame_idx:012d}"
    return {
        "record_kind": "event",
        "schema_version": SCHEMA_VERSION,
        "source_video_id": snapshot.source_video_id,
        "build_id": build_id,
        "attempt_id": attempt_id,
        "seq": seq,
        "event_id": f"{build_id}:{seq}",
        "availability_batch_id": batch_id,
        "batch_index": batch_index,
        "batch_size": batch_size,
        "event_type": "stable_board_observed",
        "side": SIDE_VALUES[snapshot.side],
        "timing": _timing(
            occurred_earliest_frame,
            snapshot.frame_idx,
            time_base.frame_to_ms(occurred_earliest_frame),
            available_ms,
        ),
        "assertion": {"state": "confirmed", "value_form": "exact"},
        "evidence": [_board_evidence(snapshot.frame_idx)],
        "checks": _board_checks(snapshot.grid),
        "missing_information": ["occurrence_frame_not_observed"],
        "relations": {
            "board_observation_id": board_id,
            "revision": {"action": "none", "target_event_ids": []},
        },
        "heavy_evidence_refs": [],
        "payload": _board_payload(snapshot, board_id),
    }


def _occurrence_earliest(
    snapshot: StableBoardSnapshot,
    previous_by_side: Mapping[str, int],
    occurrence_start_frame: int,
) -> int:
    previous = previous_by_side.get(snapshot.side)
    return occurrence_start_frame if previous is None else min(snapshot.frame_idx, previous + 1)


def _timing(
    occurred_earliest_frame: int,
    available_frame: int,
    occurred_earliest_ms: int,
    available_ms: int,
) -> dict[str, int]:
    return {
        "occurred_earliest_frame": occurred_earliest_frame,
        "occurred_earliest_ms": occurred_earliest_ms,
        "occurred_latest_frame": available_frame,
        "occurred_latest_ms": available_ms,
        "available_frame": available_frame,
        "available_ms": available_ms,
    }


def _board_evidence(frame_idx: int) -> dict[str, Any]:
    return {
        "evidence_type": "confirmed_board_snapshot",
        "method_id": "collect_boards_lean",
        "method_version": ADAPTER_VERSION,
        "from_frame": frame_idx,
        "to_frame": frame_idx,
    }


def _board_checks(grid: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
    return [
        {
            "check_id": "board_shape",
            "check_version": ADAPTER_VERSION,
            "result": "pass",
            "observed_rows": len(grid),
            "observed_columns": len(grid[0]),
            "reason_codes": [],
        },
        {
            "check_id": "board_cell_domain",
            "check_version": ADAPTER_VERSION,
            "result": "pass",
            "observed_cells": sum(len(row) for row in grid),
            "reason_codes": [],
        },
    ]


def _board_payload(
    snapshot: StableBoardSnapshot, board_id: str
) -> dict[str, Any]:
    flat = [value for row in snapshot.grid for value in row]
    payload = {
        "board_observation_id": board_id,
        "board_provenance": snapshot.board_provenance,
        "grid": [list(row) for row in snapshot.grid],
        "unknown_mask": [
            [1 if value == COLOR_UNKNOWN else 0 for value in row] for row in snapshot.grid
        ],
        "occupied_cell_count": sum(value != 0 for value in flat),
        "color_cell_count": sum(value in {1, 2, 3, 4, 5} for value in flat),
        "garbage_cell_count": sum(value == GARBAGE_VALUE for value in flat),
        "unknown_cell_count": sum(value == COLOR_UNKNOWN for value in flat),
    }
    if snapshot.observation is not None:
        payload["observed_context"] = _observation_payload(snapshot.observation)
    return payload


def _observation_payload(observation: StableBoardObservation) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    missing: list[str] = []
    _store_optional(payload, missing, "score", observation.score)
    _store_pair(payload, missing, "next_pair", observation.next_pair)
    _store_pair(payload, missing, "double_next_pair", observation.double_next_pair)
    _store_optional(payload, missing, "tsumo_count", observation.tsumo_count)
    _store_optional(payload, missing, "all_clear_pending", observation.all_clear_pending)
    _store_optional(
        payload, missing, "stable_persistence_confidence",
        observation.stable_persistence_confidence,
    )
    _store_optional(payload, missing, "chain_mechanism", observation.chain_mechanism)
    _store_optional(payload, missing, "match_end_locked", observation.match_end_locked)
    _store_optional(
        payload, missing, "post_match_lockdown_active",
        observation.post_match_lockdown_active,
    )
    if missing:
        payload["missing_fields"] = missing
    return payload


def _store_optional(
    payload: dict[str, Any], missing: list[str], key: str, value: Any,
) -> None:
    if value is None:
        missing.append(key)
    else:
        payload[key] = value


def _store_pair(
    payload: dict[str, Any], missing: list[str], key: str,
    pair: tuple[int | None, int | None] | None,
) -> None:
    if pair is None:
        missing.append(key)
        return
    known: dict[str, int] = {}
    for component, value in zip(("first", "second"), pair):
        if value is None:
            missing.append(f"{key}.{component}")
        else:
            known[component] = value
    if known:
        payload[key] = known


__all__ = [
    "SnapshotSelection",
    "SnapshotAdapterError",
    "StableBoardObservation",
    "StableBoardSnapshot",
    "VideoTimeBase",
    "build_stable_board_batches",
    "load_observed_stable_snapshots",
    "load_stable_snapshot_selection",
]
