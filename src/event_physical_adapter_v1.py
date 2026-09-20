"""物理観測サイドカーを出来事原本v1へ変換する。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.event_physical_observer_v1 import (
    PHYSICAL_OBSERVER_VERSION,
    PHYSICAL_SIDECAR_VERSION,
    PhysicalObservationRow,
)
from src.event_snapshot_adapter_v1 import VideoTimeBase
from src.event_source_v1 import SCHEMA_VERSION


PHYSICAL_ADAPTER_VERSION = "event-physical-adapter/v1"
ALLOWED_OBSERVATION_TYPES = frozenset({
    "chain_detected", "all_clear_gained_candidate",
    "all_clear_consumed_candidate", "landing_board_compared", "landing_incomplete",
})
SIDE_VALUES = frozenset({"p1", "p2"})
COUNTER_NAMES = frozenset({
    "preplacement_side_frame_count", "eligible_side_frame_count",
    "preplacement_chain_event_present_frame_count",
    "preplacement_ojama_fall_state_frame_count",
    "chain_event_present_frame_count", "chain_repeated_frame_count",
    "chain_candidate_update_count", "chain_started_count",
    "chain_update_suppressed_count", "chain_ignored_mechanism_update_count",
    "chain_reidentified_update_count", "boundary_chain_residual_suppressed_count",
    "match_evidence_rejected_side_frame_count",
    "all_clear_gained_count", "all_clear_consumed_count",
    "ojama_fall_entry_count", "landing_completed_count", "landing_incomplete_count",
    "landing_incomplete_new_fall_before_stable_count",
    "landing_incomplete_game_boundary_before_stable_count",
    "landing_incomplete_processing_end_before_stable_count",
    "landing_consistent_garbage_addition_count", "landing_zero_garbage_change_count",
    "landing_unexpected_board_change_count", "landing_stale_before_board_count",
    "landing_missing_before_board_count",
})
ELIGIBILITY_COUNTER_NAMES = frozenset({
    "preplacement_side_frame_count", "eligible_side_frame_count",
    "preplacement_chain_event_present_frame_count",
    "preplacement_ojama_fall_state_frame_count",
})
DIAGNOSTIC_COUNTER_NAMES = frozenset({
    "chain_reidentified_update_count", "boundary_chain_residual_suppressed_count",
    "match_evidence_rejected_side_frame_count",
})
PRE_DIAGNOSTIC_COUNTER_NAMES = COUNTER_NAMES - DIAGNOSTIC_COUNTER_NAMES
LEGACY_COUNTER_NAMES = PRE_DIAGNOSTIC_COUNTER_NAMES - ELIGIBILITY_COUNTER_NAMES
ALLOWED_COUNTER_NAME_SETS = {
    COUNTER_NAMES, PRE_DIAGNOSTIC_COUNTER_NAMES, LEGACY_COUNTER_NAMES,
}


class EventPhysicalAdapterError(ValueError):
    """物理観測を安全に出来事化できない場合の例外。"""


@dataclass(frozen=True, slots=True)
class EventPhysicalSidecar:
    """検査済み物理観測サイドカー。"""

    processing_start_frame: int
    processing_end_frame_exclusive: int
    observed_frame_count: int
    inspected_side_count: int
    open_landing_count: int
    counters: dict[str, int]
    rows: tuple[PhysicalObservationRow, ...]


def load_event_physical_sidecar(path: Path) -> EventPhysicalSidecar:
    """UTF-8 JSONサイドカーを検査して読み込む。"""
    return parse_event_physical_sidecar_bytes(path.read_bytes())


def parse_event_physical_sidecar_bytes(payload: bytes) -> EventPhysicalSidecar:
    """同じバイト列を内容要約と変換の両方へ使える形で解析する。"""
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EventPhysicalAdapterError("物理観測サイドカーが正しいUTF-8 JSONではありません") from exc
    if not isinstance(value, Mapping):
        raise EventPhysicalAdapterError("物理観測サイドカーはJSON objectでなければなりません")
    return _parse_sidecar(value)


def build_event_physical_events(
    sidecar: EventPhysicalSidecar, *, source_video_id: str,
    build_id: str, attempt_id: str, time_base: VideoTimeBase,
) -> tuple[dict[str, Any], ...]:
    """観測行を連鎖・全消し・着地比較の出来事へ展開する。"""
    events: list[dict[str, Any]] = []
    for row in sidecar.rows:
        if row.observation_type == "chain_detected":
            events.append(_chain_event(
                row, source_video_id, build_id, attempt_id, time_base,
                sidecar.processing_start_frame,
            ))
        elif row.observation_type.startswith("all_clear_"):
            events.append(_all_clear_event(
                row, source_video_id, build_id, attempt_id, time_base,
            ))
        elif row.observation_type == "landing_board_compared":
            events.append(_landing_event(row, source_video_id, build_id, attempt_id, time_base))
        else:
            events.append(_incomplete_event(row, source_video_id, build_id, attempt_id, time_base))
    _validate_event_positions(events, sidecar)
    return tuple(events)


def _parse_sidecar(value: Mapping[str, Any]) -> EventPhysicalSidecar:
    if value.get("schema_version") != PHYSICAL_SIDECAR_VERSION:
        raise EventPhysicalAdapterError("物理観測サイドカーの版が不正です")
    if value.get("observer_version") != PHYSICAL_OBSERVER_VERSION:
        raise EventPhysicalAdapterError("物理観測器の版が不正です")
    start = _integer(value.get("processing_start_frame"), "処理開始")
    end = _integer(value.get("processing_end_frame_exclusive"), "処理終了")
    observed = _integer(value.get("observed_frame_count"), "観測フレーム数")
    inspected = _integer(value.get("inspected_side_count"), "検査side数")
    open_count = _integer(value.get("open_landing_count"), "未完了着地数")
    counters, eligibility_contract = _parse_counters(value.get("counters"))
    rows = _parse_rows(value.get("rows"), start, end)
    if start < 0 or end <= start or observed < 0 or inspected < 0:
        raise EventPhysicalAdapterError("処理範囲または観測母数が不正です")
    if open_count not in {0, 1, 2} or inspected > observed * 2:
        raise EventPhysicalAdapterError("未完了着地数または検査side数が不正です")
    if _integer(value.get("row_count"), "観測行数") != len(rows):
        raise EventPhysicalAdapterError("観測行数が実データと一致しません")
    _validate_counters(counters, rows, inspected, eligibility_contract)
    return EventPhysicalSidecar(
        start, end, observed, inspected, open_count, counters, rows,
    )


def _parse_rows(value: Any, start: int, end: int) -> tuple[PhysicalObservationRow, ...]:
    if not isinstance(value, list):
        raise EventPhysicalAdapterError("物理観測行は配列でなければなりません")
    rows = tuple(_parse_row(item) for item in value)
    frames = [row.frame_idx for row in rows]
    if frames != sorted(frames):
        raise EventPhysicalAdapterError("物理観測行のフレーム順が増加していません")
    if any(row.frame_idx < start or row.frame_idx >= end for row in rows):
        raise EventPhysicalAdapterError("物理観測行が処理範囲外です")
    return rows


def _parse_row(value: Any) -> PhysicalObservationRow:
    if not isinstance(value, Mapping):
        raise EventPhysicalAdapterError("物理観測行はobjectでなければなりません")
    side = str(value.get("side", ""))
    kind = str(value.get("observation_type", ""))
    payload = value.get("payload")
    if side not in SIDE_VALUES or kind not in ALLOWED_OBSERVATION_TYPES:
        raise EventPhysicalAdapterError("物理観測行のsideまたは種類が不正です")
    if not isinstance(payload, Mapping):
        raise EventPhysicalAdapterError("物理観測payloadはobjectでなければなりません")
    row = PhysicalObservationRow(
        _integer(value.get("frame_idx"), "観測フレーム"),
        _number(value.get("t_sec"), "観測時刻"),
        _integer(value.get("game_idx"), "局所試合番号"),
        side, kind, dict(payload),
    )
    _validate_row_payload(row)
    return row


def _validate_row_payload(row: PhysicalObservationRow) -> None:
    if row.frame_idx < 0 or row.t_sec < 0.0 or row.game_idx < 0:
        raise EventPhysicalAdapterError("物理観測位置は0以上でなければなりません")
    if row.observation_type == "chain_detected":
        required = {
            "trigger_sec", "projected_end_sec", "chain_count", "total_erased", "total_score",
            "base_score", "all_clear_bonus_applied", "is_all_clear", "mechanism",
            "score_estimated", "observation_state", "resolver_chain_ordinal",
        }
        if required - row.payload.keys():
            raise EventPhysicalAdapterError("連鎖観測payloadに必須値がありません")
        _validate_chain_payload(row.payload, row.t_sec)
    elif row.observation_type.startswith("all_clear_"):
        _validate_all_clear_payload(row)
    elif row.observation_type == "landing_board_compared":
        _validate_landing_payload(row.payload)
    elif not isinstance(row.payload.get("reason"), str):
        raise EventPhysicalAdapterError("未完了着地に理由がありません")


def _validate_chain_payload(payload: Mapping[str, Any], observed_sec: float) -> None:
    trigger = _number(payload["trigger_sec"], "連鎖起点時刻")
    projected_end = _number(payload["projected_end_sec"], "連鎖終了予測時刻")
    if not 0.0 <= trigger <= observed_sec or projected_end < trigger:
        raise EventPhysicalAdapterError("連鎖観測時刻の順序が不正です")
    for key in ("chain_count", "total_erased", "total_score", "base_score", "all_clear_bonus_applied"):
        if _integer(payload[key], key) < 0:
            raise EventPhysicalAdapterError("連鎖観測の数値が負です")
    if _integer(payload["chain_count"], "連鎖数") < 1:
        raise EventPhysicalAdapterError("連鎖数は1以上でなければなりません")
    if not isinstance(payload["is_all_clear"], bool) or not isinstance(payload["score_estimated"], bool):
        raise EventPhysicalAdapterError("連鎖観測の真偽値が不正です")
    if not _nonempty_string(payload.get("mechanism")):
        raise EventPhysicalAdapterError("連鎖観測の方式が不正です")
    if not _nonempty_string(payload.get("observation_state")):
        raise EventPhysicalAdapterError("連鎖観測の導出状態が不正です")
    if _integer(payload["resolver_chain_ordinal"], "連鎖識別番号") < 1:
        raise EventPhysicalAdapterError("連鎖識別番号は1以上でなければなりません")


def _validate_all_clear_payload(row: PhysicalObservationRow) -> None:
    required = {
        "resolver_chain_ordinal", "chain_count", "mechanism", "observation_state",
    }
    if required - row.payload.keys():
        raise EventPhysicalAdapterError("全消し候補payloadに必須値がありません")
    if _integer(row.payload["resolver_chain_ordinal"], "連鎖識別番号") < 1:
        raise EventPhysicalAdapterError("全消し候補の連鎖識別番号が不正です")
    if _integer(row.payload["chain_count"], "連鎖数") < 1:
        raise EventPhysicalAdapterError("全消し候補の連鎖数が不正です")
    if not _nonempty_string(row.payload.get("mechanism")):
        raise EventPhysicalAdapterError("全消し候補の方式が不正です")
    if not _nonempty_string(row.payload.get("observation_state")):
        raise EventPhysicalAdapterError("全消し候補の導出状態が不正です")
    if row.observation_type == "all_clear_consumed_candidate":
        if _integer(row.payload.get("all_clear_bonus_amount"), "全消し加算") <= 0:
            raise EventPhysicalAdapterError("全消し消費量は正でなければなりません")


def _validate_landing_payload(payload: Mapping[str, Any]) -> None:
    if _integer(payload.get("fall_start_frame"), "落下開始フレーム") < 0:
        raise EventPhysicalAdapterError("落下開始フレームが負です")
    if not isinstance(payload.get("before_is_immediate_previous_observation"), bool):
        raise EventPhysicalAdapterError("着地前盤面の鮮度値が不正です")
    opponent_active = payload.get("opponent_chain_active_at_fall_start")
    if opponent_active is not None and not isinstance(opponent_active, bool):
        raise EventPhysicalAdapterError("落下開始時の相手連鎖状態が不正です")
    state = payload.get("comparison_state")
    if state not in {
        "consistent_garbage_addition", "zero_garbage_change",
        "unexpected_board_change", "stale_before_board", "missing_before_board",
    }:
        raise EventPhysicalAdapterError("着地盤面比較状態が不正です")
    _validate_grid(payload.get("after_grid"), "着地後盤面")
    if state == "missing_before_board":
        if payload.get("before_grid") is not None or "differences" in payload:
            raise EventPhysicalAdapterError("欠損した着地前盤面から差分を作れません")
        return
    _validate_grid(payload.get("before_grid"), "着地前盤面")
    if not isinstance(payload.get("differences"), Mapping):
        raise EventPhysicalAdapterError("着地盤面差がありません")
    _validate_optional_raw_landing(payload)


def _validate_optional_raw_landing(payload: Mapping[str, Any]) -> None:
    state = payload.get("raw_observation_state")
    if state is None:
        return
    if state == "not_available":
        if any(key in payload for key in (
            "raw_before_grid", "raw_after_grid", "raw_differences",
            "raw_comparison_state", "confirmed_raw_agreement",
        )):
            raise EventPhysicalAdapterError("未観測の生盤面に比較値があります")
        return
    if state != "direct_cnn_stable_frame":
        raise EventPhysicalAdapterError("生盤面の観測状態が不正です")
    _validate_grid(payload.get("raw_before_grid"), "着地前生盤面")
    _validate_grid(payload.get("raw_after_grid"), "着地後生盤面")
    if not isinstance(payload.get("raw_differences"), Mapping):
        raise EventPhysicalAdapterError("生盤面差がありません")
    if payload.get("raw_comparison_state") not in {
        "consistent_garbage_addition", "zero_garbage_change",
        "unexpected_board_change",
    }:
        raise EventPhysicalAdapterError("生盤面の比較状態が不正です")
    if not isinstance(payload.get("confirmed_raw_agreement"), bool):
        raise EventPhysicalAdapterError("確定盤面と生盤面の一致値が不正です")


def _chain_event(
    row: PhysicalObservationRow, source: str, build: str, attempt: str,
    time_base: VideoTimeBase, processing_start_frame: int,
) -> dict[str, Any]:
    payload = row.payload
    earliest = _seconds_to_frame(float(payload["trigger_sec"]), time_base)
    earliest = max(processing_start_frame, min(earliest, row.frame_idx))
    timing = _range_timing(earliest, row.frame_idx, time_base)
    payload = _chain_semantic_payload(row, timing)
    return _event_base(
        source, build, attempt, "chain_started", row.side, timing,
        {"state": "provisional", "value_form": "range"},
        "chain_event_detector", ["physical_start_frame_not_directly_observed"], payload,
    )


def _all_clear_event(
    row: PhysicalObservationRow, source: str, build: str, attempt: str,
    time_base: VideoTimeBase,
) -> dict[str, Any]:
    event_type = row.observation_type.replace("_candidate", "")
    timing = _range_timing(row.frame_idx, row.frame_idx, time_base)
    payload = {
        "local_game_index_unverified": row.game_idx,
        "derivation_state": row.payload["observation_state"],
        "chain_count": row.payload["chain_count"],
        "mechanism": row.payload["mechanism"],
        "resolver_chain_ordinal": row.payload["resolver_chain_ordinal"],
    }
    if event_type.endswith("consumed"):
        payload["all_clear_bonus_amount"] = int(row.payload["all_clear_bonus_amount"])
    return _event_base(
        source, build, attempt, event_type, row.side, timing,
        {"state": "provisional", "value_form": "exact"},
        "chain_physics_reconstruction", ["all_clear_not_independently_observed"], payload,
    )


def _landing_event(
    row: PhysicalObservationRow, source: str, build: str, attempt: str,
    time_base: VideoTimeBase,
) -> dict[str, Any]:
    start = _integer(row.payload["fall_start_frame"], "落下開始フレーム")
    state = str(row.payload["comparison_state"])
    assertion = "confirmed" if state == "consistent_garbage_addition" else "unknown"
    missing = [] if assertion == "confirmed" else [state]
    value_form = "exact" if assertion == "confirmed" else "unknown"
    before_frame = row.payload.get("before_frame")
    evidence_start = start if before_frame is None else before_frame
    return _event_base(
        source, build, attempt, "garbage_landing_board_compared", row.side,
        _range_timing(start, row.frame_idx, time_base),
        {"state": assertion, "value_form": value_form},
        "stable_board_difference", missing,
        _landing_semantic_payload(row),
        evidence_from_frame=_integer(evidence_start, "盤面差証拠開始フレーム"),
    )


def _incomplete_event(
    row: PhysicalObservationRow, source: str, build: str, attempt: str,
    time_base: VideoTimeBase,
) -> dict[str, Any]:
    start = _integer(row.payload["fall_start_frame"], "落下開始フレーム")
    reason = str(row.payload["reason"])
    return _event_base(
        source, build, attempt, "garbage_landing_observation_incomplete", row.side,
        _range_timing(start, row.frame_idx, time_base),
        {"state": "unknown", "value_form": "unknown"},
        "board_state_transition", [reason],
        _incomplete_semantic_payload(row),
    )


def _chain_semantic_payload(
    row: PhysicalObservationRow, timing: Mapping[str, int],
) -> dict[str, Any]:
    keys = (
        "chain_count", "total_erased", "total_score", "base_score",
        "all_clear_bonus_applied", "is_all_clear", "mechanism",
        "score_estimated", "observation_state",
        "resolver_chain_ordinal",
    )
    payload = {key: row.payload[key] for key in keys}
    payload.update({
        "local_game_index_unverified": row.game_idx,
        "trigger_frame_candidate": timing["occurred_earliest_frame"],
        "detected_frame": timing["available_frame"],
    })
    return payload


def _landing_semantic_payload(row: PhysicalObservationRow) -> dict[str, Any]:
    keys = (
        "fall_start_frame", "before_grid", "after_grid", "before_counts",
        "after_counts", "differences", "comparison_state",
        "before_is_immediate_previous_observation",
        "raw_before_grid", "raw_after_grid", "raw_differences",
        "raw_comparison_state", "confirmed_raw_agreement", "raw_observation_state",
        "opponent_chain_active_at_fall_start",
    )
    payload = {key: row.payload[key] for key in keys if row.payload.get(key) is not None}
    before_frame = row.payload.get("before_frame")
    if before_frame is not None:
        payload["before_frame"] = before_frame
    payload["local_game_index_unverified"] = row.game_idx
    return payload


def _incomplete_semantic_payload(row: PhysicalObservationRow) -> dict[str, Any]:
    payload = {
        "fall_start_frame": row.payload["fall_start_frame"],
        "reason": row.payload["reason"],
        "local_game_index_unverified": row.game_idx,
    }
    if row.payload.get("before_frame") is not None:
        payload["before_frame"] = row.payload["before_frame"]
    return payload


def _event_base(
    source: str, build: str, attempt: str, event_type: str, side: str,
    timing: dict[str, int], assertion: dict[str, str], evidence_type: str,
    missing: list[str], payload: dict[str, Any],
    evidence_from_frame: int | None = None,
) -> dict[str, Any]:
    evidence_start = (
        timing["occurred_earliest_frame"]
        if evidence_from_frame is None else evidence_from_frame
    )
    return {
        "record_kind": "event", "schema_version": SCHEMA_VERSION,
        "source_video_id": source, "build_id": build, "attempt_id": attempt,
        "seq": 0, "event_id": f"{build}:0",
        "availability_batch_id": f"{build}:unsequenced", "batch_index": 0, "batch_size": 1,
        "event_type": event_type, "side": side, "timing": timing, "assertion": assertion,
        "evidence": [{"evidence_type": evidence_type, "method_id": PHYSICAL_OBSERVER_VERSION,
                      "method_version": PHYSICAL_ADAPTER_VERSION,
                      "from_frame": evidence_start,
                      "to_frame": timing["occurred_latest_frame"]}],
        "checks": [_adapter_check()], "missing_information": missing,
        "relations": {"revision": {"action": "none", "target_event_ids": []}},
        "heavy_evidence_refs": [], "payload": payload,
    }


def _range_timing(earliest: int, available: int, time_base: VideoTimeBase) -> dict[str, int]:
    return {
        "occurred_earliest_frame": earliest,
        "occurred_earliest_ms": time_base.frame_to_ms(earliest),
        "occurred_latest_frame": available,
        "occurred_latest_ms": time_base.frame_to_ms(available),
        "available_frame": available,
        "available_ms": time_base.frame_to_ms(available),
    }


def _seconds_to_frame(seconds: float, time_base: VideoTimeBase) -> int:
    value = Decimal(str(seconds)) * time_base.denominator / time_base.numerator
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _validate_event_positions(
    events: Sequence[dict[str, Any]], sidecar: EventPhysicalSidecar,
) -> None:
    for event in events:
        timing = event["timing"]
        if timing["occurred_earliest_frame"] < sidecar.processing_start_frame:
            raise EventPhysicalAdapterError("物理出来事が処理開始より前です")
        if timing["available_frame"] >= sidecar.processing_end_frame_exclusive:
            raise EventPhysicalAdapterError("物理出来事が処理終了以後にあります")


def _validate_grid(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) != 13:
        raise EventPhysicalAdapterError(f"{name}は13行でなければなりません")
    if any(not isinstance(row, list) or len(row) != 6 for row in value):
        raise EventPhysicalAdapterError(f"{name}は6列でなければなりません")


def _parse_counters(value: Any) -> tuple[dict[str, int], bool]:
    keys = frozenset(value) if isinstance(value, Mapping) else frozenset()
    if not isinstance(value, Mapping) or keys not in ALLOWED_COUNTER_NAME_SETS:
        raise EventPhysicalAdapterError("物理観測カウンタのキー集合が不正です")
    counters = {str(key): _integer(item, str(key)) for key, item in value.items()}
    eligibility_contract = ELIGIBILITY_COUNTER_NAMES.issubset(keys)
    for key in ELIGIBILITY_COUNTER_NAMES | DIAGNOSTIC_COUNTER_NAMES:
        counters.setdefault(key, 0)
    if any(item < 0 for item in counters.values()):
        raise EventPhysicalAdapterError("物理観測カウンタが負です")
    return counters, eligibility_contract


def _validate_counters(
    counters: Mapping[str, int], rows: Sequence[PhysicalObservationRow], inspected: int,
    eligibility_contract: bool,
) -> None:
    present_parts = (
        counters["chain_repeated_frame_count"]
        + counters["chain_candidate_update_count"]
    )
    update_parts = (
        counters["chain_started_count"] + counters["chain_update_suppressed_count"]
        + counters["chain_ignored_mechanism_update_count"]
    )
    incomplete = sum(
        counters[f"landing_incomplete_{reason}_count"]
        for reason in (
            "new_fall_before_stable", "game_boundary_before_stable",
            "processing_end_before_stable",
        )
    )
    if counters["chain_event_present_frame_count"] != present_parts:
        raise EventPhysicalAdapterError("連鎖観測母数の内訳が一致しません")
    if counters["chain_candidate_update_count"] != update_parts:
        raise EventPhysicalAdapterError("連鎖候補更新の内訳が一致しません")
    diagnosed = (
        counters["chain_reidentified_update_count"]
        + counters["boundary_chain_residual_suppressed_count"]
    )
    if diagnosed > counters["chain_update_suppressed_count"]:
        raise EventPhysicalAdapterError("連鎖抑制の診断内訳が抑制総数を超えています")
    if counters["chain_event_present_frame_count"] > inspected:
        raise EventPhysicalAdapterError("連鎖観測母数が検査side数を超えています")
    if eligibility_contract:
        eligibility = (
            counters["preplacement_side_frame_count"]
            + counters["eligible_side_frame_count"]
        )
        if eligibility != inspected:
            raise EventPhysicalAdapterError("物理観測の利用可否母数が一致しません")
    _validate_landing_counters(counters, rows, incomplete)


def _validate_landing_counters(
    counters: Mapping[str, int], rows: Sequence[PhysicalObservationRow], incomplete: int,
) -> None:
    if counters["landing_incomplete_count"] != incomplete:
        raise EventPhysicalAdapterError("未完了着地の理由別内訳が一致しません")
    if counters["ojama_fall_entry_count"] != counters["landing_completed_count"] + incomplete:
        raise EventPhysicalAdapterError("おじゃま落下入口の内訳が一致しません")
    expected = {
        "chain_started_count": _row_type_count(rows, "chain_detected"),
        "all_clear_gained_count": _row_type_count(rows, "all_clear_gained_candidate"),
        "all_clear_consumed_count": _row_type_count(rows, "all_clear_consumed_candidate"),
        "landing_completed_count": _row_type_count(rows, "landing_board_compared"),
        "landing_incomplete_count": _row_type_count(rows, "landing_incomplete"),
    }
    for state in (
        "consistent_garbage_addition", "zero_garbage_change",
        "unexpected_board_change", "stale_before_board", "missing_before_board",
    ):
        expected[f"landing_{state}_count"] = _row_payload_count(
            rows, "landing_board_compared", "comparison_state", state,
        )
    if any(counters[key] != value for key, value in expected.items()):
        raise EventPhysicalAdapterError("物理観測カウンタと観測行数が一致しません")


def _row_type_count(rows: Sequence[PhysicalObservationRow], kind: str) -> int:
    return sum(row.observation_type == kind for row in rows)


def _row_payload_count(
    rows: Sequence[PhysicalObservationRow], kind: str, key: str, value: str,
) -> int:
    return sum(
        row.observation_type == kind and row.payload.get(key) == value for row in rows
    )


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EventPhysicalAdapterError(f"{name}は整数でなければなりません")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EventPhysicalAdapterError(f"{name}は数値でなければなりません")
    return float(value)


def _adapter_check() -> dict[str, Any]:
    return {
        "check_id": "event_physical_sidecar_valid",
        "check_version": PHYSICAL_ADAPTER_VERSION,
        "result": "pass", "reason_codes": [],
    }


__all__ = [
    "EventPhysicalAdapterError",
    "EventPhysicalSidecar",
    "PHYSICAL_ADAPTER_VERSION",
    "build_event_physical_events",
    "load_event_physical_sidecar",
    "parse_event_physical_sidecar_bytes",
]
