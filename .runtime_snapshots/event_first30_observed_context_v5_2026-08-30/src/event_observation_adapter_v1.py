"""境界証拠と事後WIN★観測を出来事原本v1へ変換する。"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.event_snapshot_adapter_v1 import VideoTimeBase
from src.event_source_v1 import SCHEMA_VERSION, validate_event_batch


OBSERVATION_SCHEMA_VERSION = "event-observation-sidecar/v1"
OBSERVATION_ADAPTER_VERSION = "event-observation-adapter/v1"
OFFICIAL_GAME_ASSIGNMENT_VERSION = "official-game-assignment/v1"
ALLOWED_WINNER_RESULTS = frozenset(
    {"winner_observed", "ambiguous", "panel_unavailable"}
)
EVENT_PRIORITY = {
    "match_boundary_evidence": 0,
    "chain_started": 4,
    "attack_provisional_updated": 5,
    "garbage_expired_at_boundary": 2,
    "attack_provisional_unfinalized": 6,
    "attack_finalized": 10,
    "garbage_cancelled": 11,
    "garbage_sent": 12,
    "garbage_fall_reserved": 13,
    "garbage_fall_started": 14,
    "garbage_fall_completed": 15,
    "all_clear_consumed": 16,
    "all_clear_gained": 17,
    "garbage_landing_board_compared": 18,
    "garbage_landing_observation_incomplete": 19,
    "stable_board_observed": 20,
    "winner_observed": 90,
    "official_game_assignment": 100,
}
SIDE_PRIORITY = {"p1": 0, "p2": 1, "both": 2, "system": 3, "unknown": 4}


class EventObservationAdapterError(ValueError):
    """境界・勝者観測を安全に出来事化できない場合の例外。"""


@dataclass(frozen=True, slots=True)
class BoundaryObservation:
    closing_game_index: int
    opening_game_index: int
    observed_sec: Decimal
    evidence_type: str


@dataclass(frozen=True, slots=True)
class WinnerPanelObservation:
    game_index: int
    result: str
    winner_side: str
    evidence_start_sec: Decimal
    evidence_end_sec: Decimal
    available_sec: Decimal
    availability_reason: str


@dataclass(frozen=True, slots=True)
class EventObservationSidecar:
    processing_start_sec: Decimal
    processing_end_sec: Decimal
    winner_detector_status: str
    boundaries: tuple[BoundaryObservation, ...]
    winner_results: tuple[WinnerPanelObservation, ...]


def load_event_observation_sidecar(path: Path) -> EventObservationSidecar:
    """収集器の事後観測サイドカーを厳格に読み出す。"""

    return parse_event_observation_sidecar_bytes(path.read_bytes())


def parse_event_observation_sidecar_bytes(payload: bytes) -> EventObservationSidecar:
    """内容要約値と同じバイト列から観測サイドカーを厳格に解釈する。"""

    try:
        value = json.loads(payload.decode("utf-8"), parse_float=Decimal)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EventObservationAdapterError("観測サイドカーがUTF-8 JSONではありません") from error
    if not isinstance(value, dict):
        raise EventObservationAdapterError("観測サイドカーはJSONオブジェクトが必要です")
    if value.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise EventObservationAdapterError("観測サイドカーの仕様版が不正です")
    if value.get("generated_posthoc") is not True:
        raise EventObservationAdapterError("事後生成の明示がない観測サイドカーです")
    start = _decimal(value.get("processing_start_sec"), "処理開始秒")
    end = _decimal(value.get("processing_end_sec"), "処理終了秒")
    status = str(value.get("winner_detector_status", ""))
    if end < start or status not in {"completed", "failed"}:
        raise EventObservationAdapterError("処理区間または勝者検出状態が不正です")
    boundaries = _load_boundaries(value.get("accepted_boundary_evidence"), start, end)
    winners = _load_winners(value.get("winner_panel_results"), start, end, status)
    if status == "completed" and len(winners) != len(boundaries) + 1:
        raise EventObservationAdapterError("勝者パネル結果数が処理区間の試合数と一致しません")
    return EventObservationSidecar(start, end, status, boundaries, winners)


def build_event_observation_events(
    observation: EventObservationSidecar,
    *,
    source_video_id: str,
    build_id: str,
    attempt_id: str,
    time_base: VideoTimeBase,
    processing_start_frame: int,
    processing_end_frame_exclusive: int,
) -> tuple[dict[str, Any], ...]:
    """境界は観測時、勝者は事後処理末尾を利用可能位置として出来事化する。"""

    events = [
        _boundary_event(item, source_video_id, build_id, attempt_id, time_base)
        for item in observation.boundaries
    ]
    events.extend(
        _winner_event(
            item,
            source_video_id,
            build_id,
            attempt_id,
            time_base,
            processing_end_frame_exclusive,
        )
        for item in observation.winner_results
        if item.result == "winner_observed"
    )
    _validate_event_range(events, processing_start_frame, processing_end_frame_exclusive)
    return tuple(events)


def merge_and_resequence_batches(
    base_batches: Sequence[Sequence[dict[str, Any]]],
    extra_events: Sequence[dict[str, Any]],
    *,
    build_id: str,
    attempt_id: str,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """全出来事を利用可能位置で統合し、通番と一括更新IDを再計算する。"""

    events = [copy.deepcopy(event) for batch in base_batches for event in batch]
    prior_ids = _remember_prior_event_ids(events)
    events.extend(copy.deepcopy(event) for event in extra_events)
    ordered = sorted(events, key=_event_order_key)
    batches: list[tuple[dict[str, Any], ...]] = []
    seq = 0
    for available_frame, group in _group_by_available_frame(ordered):
        batch_id = f"{build_id}:batch-frame-{available_frame:012d}"
        batch = tuple(
            _reidentify(event, seq + index, index, len(group), batch_id, attempt_id)
            for index, event in enumerate(group)
        )
        batches.append(batch)
        seq += len(batch)
    _remap_prior_event_relations(batches, prior_ids)
    _resolve_revision_targets(batches)
    _refresh_assignment_sequence_ranges(batches)
    expected = 0
    for batch in batches:
        validate_event_batch(batch, expected_first_seq=expected)
        expected += len(batch)
    return tuple(batches)


def _remember_prior_event_ids(
    events: Sequence[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], str], ...]:
    """再採番前の出来事IDを、同じ辞書への参照と組にして保持する。"""

    pairs = tuple((event, str(event["event_id"])) for event in events)
    ids = [old_id for _, old_id in pairs]
    if len(ids) != len(set(ids)):
        raise EventObservationAdapterError("統合前の出来事IDが重複しています")
    return pairs


def _remap_prior_event_relations(
    batches: Sequence[Sequence[dict[str, Any]]],
    prior_ids: Sequence[tuple[dict[str, Any], str]],
) -> None:
    """再採番済みの既存出来事を指す関係IDを新しいIDへ追従させる。"""

    remap = {old_id: str(event["event_id"]) for event, old_id in prior_ids}
    for event in (item for batch in batches for item in batch):
        relations = event.get("relations", {})
        revision = relations.get("revision", {})
        _remap_id_list(revision, "target_event_ids", remap)
        _remap_id_list(relations, "verified_cause_event_ids", remap)
        _remap_id_list(relations, "unfinalized_event_ids", remap)


def _remap_id_list(
    container: dict[str, Any], key: str, remap: dict[str, str],
) -> None:
    values = container.get(key)
    if not isinstance(values, list):
        return
    container[key] = [remap.get(str(value), str(value)) for value in values]


def _resolve_revision_targets(
    batches: Sequence[Sequence[dict[str, Any]]],
) -> None:
    events = [event for batch in batches for event in batch]
    by_token: dict[str, str] = {}
    for event in events:
        token = event.get("payload", {}).get("provisional_observation_id")
        if token is None:
            continue
        if not isinstance(token, str) or token in by_token:
            raise EventObservationAdapterError("暫定観測IDが空または重複しています")
        by_token[token] = str(event["event_id"])
    for event in events:
        relations = event.get("relations", {})
        revision = relations.get("revision", {})
        tokens = revision.pop("target_observation_ids", None)
        if tokens is not None:
            revision["target_event_ids"] = _resolve_observation_ids(
                tokens, by_token, event,
            )
        unfinalized = relations.pop("unfinalized_observation_ids", None)
        if unfinalized is not None:
            relations["unfinalized_event_ids"] = _resolve_observation_ids(
                unfinalized, by_token, event,
            )


def _resolve_observation_ids(
    tokens: Any, by_token: dict[str, str], event: dict[str, Any],
) -> list[str]:
    if not isinstance(tokens, list) or any(token not in by_token for token in tokens):
        raise EventObservationAdapterError("暫定観測IDを解決できません")
    targets = [by_token[str(token)] for token in tokens]
    if any(int(target.rsplit(":", 1)[1]) >= int(event["seq"]) for target in targets):
        raise EventObservationAdapterError("暫定観測は閉鎖行より前でなければなりません")
    return targets


def _refresh_assignment_sequence_ranges(
    batches: Sequence[Sequence[dict[str, Any]]],
) -> None:
    """再採番後の公式試合範囲を現在の通番から必ず再計算する。"""
    events = [event for batch in batches for event in batch]
    boundaries = _indexed_events(
        events, "match_boundary_evidence", "closing_game_index_unverified",
    )
    for assignment in (
        event for event in events
        if event["event_type"] == "official_game_assignment"
    ):
        index = int(assignment["payload"]["local_game_index"])
        previous, closing = boundaries.get(index - 1), boundaries.get(index)
        assignment["payload"]["from_sequence"] = (
            0 if previous is None else int(previous["seq"]) + 1
        )
        assignment["payload"]["to_sequence"] = _assignment_end_sequence(
            assignment, closing, events,
        )


def _assignment_end_sequence(
    assignment: dict[str, Any], closing: dict[str, Any] | None,
    events: Sequence[dict[str, Any]],
) -> int:
    if closing is not None:
        return int(closing["seq"])
    available = int(assignment["timing"]["available_frame"])
    online = [
        int(event["seq"]) for event in events
        if int(event["timing"]["available_frame"]) < available
        and event["event_type"] not in {"winner_observed", "official_game_assignment"}
    ]
    if not online:
        raise EventObservationAdapterError("公式試合範囲にオンライン出来事がありません")
    return max(online)


def build_official_game_assignment_events(
    batches: Sequence[Sequence[dict[str, Any]]],
    *,
    build_id: str,
    attempt_id: str,
    time_base: VideoTimeBase,
    processing_start_frame: int,
    processing_end_frame_exclusive: int,
) -> tuple[dict[str, Any], ...]:
    """公式勝者と境界が結ぶ通番範囲を、事後の追記関係として作る。"""

    if processing_start_frame != 0:
        return ()
    events = [event for batch in batches for event in batch]
    boundaries = _indexed_events(
        events, "match_boundary_evidence", "closing_game_index_unverified"
    )
    winners = _indexed_events(events, "winner_observed", "game_index_unverified")
    online = [
        event for event in events
        if int(event["timing"]["available_frame"]) < processing_end_frame_exclusive
    ]
    if not online:
        return ()
    candidates = (
        _official_assignment_event(
            index, winner, boundaries, online, build_id, attempt_id, time_base,
            processing_start_frame, processing_end_frame_exclusive,
        )
        for index, winner in sorted(winners.items())
    )
    return tuple(event for event in candidates if event is not None)


def _load_boundaries(
    value: Any, start: Decimal, end: Decimal
) -> tuple[BoundaryObservation, ...]:
    rows = _mapping_rows(value, "境界証拠")
    result: list[BoundaryObservation] = []
    for index, row in enumerate(rows):
        observed = _decimal(row.get("observed_sec"), "境界観測秒")
        item = BoundaryObservation(
            _integer(row.get("closing_game_index"), "終了試合index"),
            _integer(row.get("opening_game_index"), "開始試合index"),
            observed,
            str(row.get("evidence_type", "")),
        )
        if item.closing_game_index != index or item.opening_game_index != index + 1:
            raise EventObservationAdapterError("境界証拠の試合indexが連続していません")
        if not start <= observed <= end or not item.evidence_type:
            raise EventObservationAdapterError("境界証拠の時刻または種類が不正です")
        result.append(item)
    if any(a.observed_sec >= b.observed_sec for a, b in zip(result, result[1:])):
        raise EventObservationAdapterError("境界証拠の時刻が増加していません")
    return tuple(result)


def _load_winners(
    value: Any, start: Decimal, end: Decimal, detector_status: str
) -> tuple[WinnerPanelObservation, ...]:
    rows = _mapping_rows(value, "勝者パネル結果")
    if detector_status == "failed" and rows:
        raise EventObservationAdapterError("勝者検出失敗時に結果を持てません")
    result = tuple(_winner_from_row(row, start, end) for row in rows)
    indices = [item.game_index for item in result]
    if indices != list(range(len(indices))):
        raise EventObservationAdapterError("勝者パネル結果の試合indexが連続していません")
    return result


def _winner_from_row(
    row: dict[str, Any], start: Decimal, end: Decimal
) -> WinnerPanelObservation:
    item = WinnerPanelObservation(
        _integer(row.get("game_index"), "勝者試合index"),
        str(row.get("result", "")),
        str(row.get("winner_side", "")),
        _decimal(row.get("evidence_start_sec"), "勝者証拠開始秒"),
        _decimal(row.get("evidence_end_sec"), "勝者証拠終了秒"),
        _decimal(row.get("available_sec"), "勝者利用可能秒"),
        str(row.get("availability_reason", "")),
    )
    if item.result not in ALLOWED_WINNER_RESULTS:
        raise EventObservationAdapterError("勝者パネル結果の種類が不正です")
    expected_side = {"1P", "2P"} if item.result == "winner_observed" else {"unknown"}
    if item.winner_side not in expected_side:
        raise EventObservationAdapterError("勝者パネル結果と対象側が矛盾しています")
    if not start <= item.evidence_start_sec <= item.evidence_end_sec <= item.available_sec:
        raise EventObservationAdapterError("勝者パネル結果の時刻順が不正です")
    if item.available_sec != end or item.availability_reason != "posthoc_full_clip_detector":
        raise EventObservationAdapterError("事後勝者の利用可能位置が処理末尾ではありません")
    return item


def _boundary_event(
    item: BoundaryObservation,
    source_video_id: str,
    build_id: str,
    attempt_id: str,
    time_base: VideoTimeBase,
) -> dict[str, Any]:
    frame = _seconds_to_frame(item.observed_sec, time_base)
    return _event_base(
        source_video_id,
        build_id,
        attempt_id,
        "match_boundary_evidence",
        "system",
        _point_timing(frame, time_base),
        {"state": "confirmed", "value_form": "exact"},
        [{"evidence_type": item.evidence_type, "method_id": "collect_boards_lean",
          "method_version": OBSERVATION_ADAPTER_VERSION, "from_frame": frame, "to_frame": frame}],
        [],
        {
            "closing_game_index_unverified": item.closing_game_index,
            "opening_game_index_unverified": item.opening_game_index,
            "formal_assignment": False,
        },
    )


def _winner_event(
    item: WinnerPanelObservation,
    source_video_id: str,
    build_id: str,
    attempt_id: str,
    time_base: VideoTimeBase,
    processing_end_frame_exclusive: int,
) -> dict[str, Any]:
    earliest = _seconds_to_frame(item.evidence_start_sec, time_base)
    latest = _seconds_to_frame(item.evidence_end_sec, time_base)
    timing = _range_timing(earliest, latest, processing_end_frame_exclusive, time_base)
    return _event_base(
        source_video_id,
        build_id,
        attempt_id,
        "winner_observed",
        "p1" if item.winner_side == "1P" else "p2",
        timing,
        {"state": "confirmed", "value_form": "exact"},
        [{"evidence_type": "win_panel_number_change", "method_id": "MatchWinnerDetector",
          "method_version": OBSERVATION_ADAPTER_VERSION, "from_frame": earliest, "to_frame": latest}],
        ["available_only_posthoc"],
        {
            "game_index_unverified": item.game_index,
            "winner_side": item.winner_side,
            "availability_reason": item.availability_reason,
        },
    )


def _event_base(
    source_video_id: str,
    build_id: str,
    attempt_id: str,
    event_type: str,
    side: str,
    timing: dict[str, int],
    assertion: dict[str, str],
    evidence: list[dict[str, Any]],
    missing: list[str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "record_kind": "event",
        "schema_version": SCHEMA_VERSION,
        "source_video_id": source_video_id,
        "build_id": build_id,
        "attempt_id": attempt_id,
        "seq": 0,
        "event_id": f"{build_id}:0",
        "availability_batch_id": f"{build_id}:unsequenced",
        "batch_index": 0,
        "batch_size": 1,
        "event_type": event_type,
        "side": side,
        "timing": timing,
        "assertion": assertion,
        "evidence": evidence,
        "checks": [_adapter_check()],
        "missing_information": missing,
        "relations": {"revision": {"action": "none", "target_event_ids": []}},
        "heavy_evidence_refs": [],
        "payload": payload,
    }


def _indexed_events(
    events: Sequence[dict[str, Any]], event_type: str, payload_key: str
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for event in events:
        if event["event_type"] != event_type:
            continue
        index = int(event["payload"][payload_key])
        if index in result:
            raise EventObservationAdapterError("同じ局所試合indexの出来事が重複しています")
        result[index] = event
    return result


def _official_assignment_event(
    index: int,
    winner: dict[str, Any],
    boundaries: dict[int, dict[str, Any]],
    online: Sequence[dict[str, Any]],
    build_id: str,
    attempt_id: str,
    time_base: VideoTimeBase,
    processing_start_frame: int,
    processing_end_frame_exclusive: int,
) -> dict[str, Any] | None:
    range_values = _official_assignment_range(
        index, boundaries, online, processing_start_frame,
        processing_end_frame_exclusive,
    )
    if range_values is None:
        return None
    from_seq, to_seq, start_frame, end_frame, boundary_causes = range_values
    event = _event_base(
        str(winner["source_video_id"]), build_id, attempt_id,
        "official_game_assignment", "system",
        _range_timing(start_frame, end_frame, processing_end_frame_exclusive, time_base),
        {"state": "confirmed", "value_form": "exact"},
        [{"evidence_type": "winner_and_boundary_relation", "method_id": "official_game_assignment",
          "method_version": OFFICIAL_GAME_ASSIGNMENT_VERSION,
          "from_frame": start_frame, "to_frame": end_frame}],
        ["available_only_posthoc"],
        {"official_game_number": index + 1, "local_game_index": index,
         "from_sequence": from_seq, "to_sequence": to_seq,
         "winner_side": winner["payload"]["winner_side"],
         "range_excludes_posthoc_labels": True},
    )
    event["relations"]["verified_cause_event_ids"] = [
        winner["event_id"], *boundary_causes
    ]
    return event


def _official_assignment_range(
    index: int,
    boundaries: dict[int, dict[str, Any]],
    online: Sequence[dict[str, Any]],
    processing_start_frame: int,
    processing_end_frame_exclusive: int,
) -> tuple[int, int, int, int, list[str]] | None:
    previous = boundaries.get(index - 1)
    if index > 0 and previous is None:
        raise EventObservationAdapterError("公式試合範囲の開始境界がありません")
    closing = boundaries.get(index)
    from_seq = 0 if previous is None else int(previous["seq"]) + 1
    to_seq = int(closing["seq"]) if closing is not None else max(
        int(event["seq"]) for event in online
    )
    if from_seq > to_seq:
        if closing is None:
            return None
        raise EventObservationAdapterError("公式試合の通番範囲が逆転しています")
    start_frame = processing_start_frame if previous is None else int(
        previous["timing"]["available_frame"]
    )
    end_frame = processing_end_frame_exclusive if closing is None else int(
        closing["timing"]["available_frame"]
    )
    causes = [
        event["event_id"] for event in (previous, closing) if event is not None
    ]
    return from_seq, to_seq, start_frame, end_frame, causes


def _adapter_check() -> dict[str, Any]:
    return {
        "check_id": "event_observation_sidecar_valid",
        "check_version": OBSERVATION_ADAPTER_VERSION,
        "result": "pass",
        "reason_codes": [],
    }


def _point_timing(frame: int, time_base: VideoTimeBase) -> dict[str, int]:
    milliseconds = time_base.frame_to_ms(frame)
    return _range_timing(frame, frame, frame, time_base, milliseconds)


def _range_timing(
    earliest: int,
    latest: int,
    available: int,
    time_base: VideoTimeBase,
    available_ms: int | None = None,
) -> dict[str, int]:
    return {
        "occurred_earliest_frame": earliest,
        "occurred_earliest_ms": time_base.frame_to_ms(earliest),
        "occurred_latest_frame": latest,
        "occurred_latest_ms": time_base.frame_to_ms(latest),
        "available_frame": available,
        "available_ms": time_base.frame_to_ms(available) if available_ms is None else available_ms,
    }


def _seconds_to_frame(seconds: Decimal, time_base: VideoTimeBase) -> int:
    value = seconds * time_base.denominator / time_base.numerator
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _validate_event_range(events: Sequence[dict[str, Any]], start: int, end: int) -> None:
    for event in events:
        timing = event["timing"]
        if timing["occurred_earliest_frame"] < start:
            raise EventObservationAdapterError("観測出来事が処理開始より前です")
        if timing["occurred_latest_frame"] > end or timing["available_frame"] > end:
            raise EventObservationAdapterError("観測出来事が処理終了より後です")


def _event_order_key(event: dict[str, Any]) -> tuple[int, int, int, int, str]:
    source_order = event.get("relations", {}).get("accounting_source_order")
    priority = EVENT_PRIORITY.get(str(event["event_type"]), 99)
    if isinstance(source_order, int) and source_order > 0:
        stream_priority, within_stream = -1, source_order
    else:
        stream_priority, within_stream = priority, 0
    return (
        int(event["timing"]["available_frame"]),
        stream_priority,
        within_stream,
        SIDE_PRIORITY[str(event["side"])],
        str(event["event_type"]),
    )


def _group_by_available_frame(
    events: Sequence[dict[str, Any]],
) -> Iterable[tuple[int, tuple[dict[str, Any], ...]]]:
    current_frame: int | None = None
    current: list[dict[str, Any]] = []
    for event in events:
        frame = int(event["timing"]["available_frame"])
        if current and frame != current_frame:
            yield int(current_frame), tuple(current)
            current = []
        current_frame = frame
        current.append(event)
    if current:
        yield int(current_frame), tuple(current)


def _reidentify(
    event: dict[str, Any],
    seq: int,
    batch_index: int,
    batch_size: int,
    batch_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    event["seq"] = seq
    event["event_id"] = f"{event['build_id']}:{seq}"
    event["attempt_id"] = attempt_id
    event["availability_batch_id"] = batch_id
    event["batch_index"] = batch_index
    event["batch_size"] = batch_size
    return event


def _mapping_rows(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise EventObservationAdapterError(f"{name}はJSONオブジェクト配列が必要です")
    return value


def _decimal(value: Any, name: str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as error:
        raise EventObservationAdapterError(f"{name}が数値ではありません") from error
    if not result.is_finite() or result < 0:
        raise EventObservationAdapterError(f"{name}は0以上の有限値が必要です")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EventObservationAdapterError(f"{name}は0以上の整数が必要です")
    return value


__all__ = [
    "EventObservationAdapterError",
    "EventObservationSidecar",
    "OBSERVATION_ADAPTER_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "OFFICIAL_GAME_ASSIGNMENT_VERSION",
    "build_official_game_assignment_events",
    "build_event_observation_events",
    "load_event_observation_sidecar",
    "merge_and_resequence_batches",
    "parse_event_observation_sidecar_bytes",
]
