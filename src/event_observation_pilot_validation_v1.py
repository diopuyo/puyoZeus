"""盤面外観測を含む完了試行の保存則と未来情報分離を再検査する。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Sequence

from src.event_observation_adapter_v1 import (
    EventObservationSidecar,
    parse_event_observation_sidecar_bytes,
)
from src.event_pilot_analysis_v1 import load_completed_run_events


ALLOWED_EVENT_TYPES = frozenset(
    {
        "stable_board_observed",
        "match_boundary_evidence",
        "winner_observed",
        "official_game_assignment",
        "chain_started",
        "attack_provisional_updated",
        "attack_provisional_unfinalized",
        "attack_finalized",
        "garbage_cancelled",
        "garbage_sent",
        "garbage_fall_reserved",
        "garbage_fall_started",
        "garbage_fall_completed",
        "garbage_same_frame_order_ambiguous",
        "garbage_expired_at_boundary",
        "all_clear_consumed",
        "all_clear_gained",
        "garbage_landing_board_compared",
        "garbage_landing_observation_incomplete",
    }
)
EXPECTED_CONFIG_VERSION = "event-snapshot-recognition-config/4"


class ObservationPilotValidationError(ValueError):
    """盤面外観測の実行成果物が検収条件を満たさない場合の例外。"""


def validate_observation_pilot_run(run_dir: Path) -> dict[str, Any]:
    """一試行を独立に読み直し、件数・時刻・隔離条件を検査する。"""

    manifest, events = load_completed_run_events(run_dir)
    return validate_observation_pilot_loaded_run(run_dir, manifest, events)


def validate_observation_pilot_loaded_run(
    run_dir: Path, manifest: dict[str, Any], events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """直前に完成検査した共有出来事列から盤面外観測を再計算する。"""

    config = _load_object(run_dir / "recognition-config.json")
    observation, sidecar_sha256 = _embedded_observation(config)
    source = manifest["source"]
    grouped = _group_events(events)
    _validate_boundaries(grouped["match_boundary_evidence"], observation, source)
    _validate_winners(grouped["winner_observed"], observation, source)
    _validate_assignments(grouped, source)
    expected_total = sum(len(rows) for rows in grouped.values())
    if expected_total != len(events):
        raise ObservationPilotValidationError("許可外の出来事種類が混在しています")
    excluded = Counter(
        item.result for item in observation.winner_results
        if item.result != "winner_observed"
    )
    return {
        "run_dir": str(run_dir.resolve()),
        "build_id": manifest["build_id"],
        "attempt_id": manifest["attempt_id"],
        "event_count": len(events),
        "stable_board_event_count": len(grouped["stable_board_observed"]),
        "boundary_event_count": len(grouped["match_boundary_evidence"]),
        "winner_observed_event_count": len(grouped["winner_observed"]),
        "excluded_winner_result_counts": dict(sorted(excluded.items())),
        "winner_events_available_before_processing_end": 0,
        "official_game_assignment_event_count": len(
            grouped["official_game_assignment"]
        ),
        "observation_sidecar_sha256": sidecar_sha256,
        "validation_pass": True,
    }


def _embedded_observation(
    config: dict[str, Any],
) -> tuple[EventObservationSidecar, str]:
    if config.get("format_version") != EXPECTED_CONFIG_VERSION:
        raise ObservationPilotValidationError("認識設定の仕様版が盤面外観測対応ではありません")
    embedded = config.get("event_observation_input")
    if not isinstance(embedded, dict) or embedded.get("format") != "embedded-utf8-json":
        raise ObservationPilotValidationError("観測サイドカーが自己完結していません")
    content = embedded.get("content_utf8")
    expected_sha = embedded.get("sha256")
    if not isinstance(content, str) or not isinstance(expected_sha, str):
        raise ObservationPilotValidationError("埋込み観測サイドカーが不正です")
    payload = content.encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ObservationPilotValidationError("埋込み観測サイドカーの内容要約が一致しません")
    return parse_event_observation_sidecar_bytes(payload), expected_sha


def _group_events(events: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {event_type: [] for event_type in ALLOWED_EVENT_TYPES}
    for event in events:
        event_type = str(event["event_type"])
        if event_type not in ALLOWED_EVENT_TYPES:
            continue
        grouped[event_type].append(event)
    return grouped


def _validate_boundaries(
    events: Sequence[dict[str, Any]],
    observation: EventObservationSidecar,
    source: dict[str, Any],
) -> None:
    by_index = _unique_events(events, "closing_game_index_unverified")
    if len(by_index) != len(observation.boundaries):
        raise ObservationPilotValidationError("境界出来事件数がサイドカーと一致しません")
    for item in observation.boundaries:
        event = by_index.get(item.closing_game_index)
        expected_frame = _frame(item.observed_sec, source)
        if event is None or event["side"] != "system":
            raise ObservationPilotValidationError("境界出来事の対象または試合indexが不正です")
        timing = event["timing"]
        positions = (
            timing["occurred_earliest_frame"], timing["occurred_latest_frame"],
            timing["available_frame"],
        )
        if positions != (expected_frame, expected_frame, expected_frame):
            raise ObservationPilotValidationError("境界出来事の利用可能位置が不正です")


def _validate_winners(
    events: Sequence[dict[str, Any]],
    observation: EventObservationSidecar,
    source: dict[str, Any],
) -> None:
    by_index = _unique_events(events, "game_index_unverified")
    expected = [item for item in observation.winner_results if item.result == "winner_observed"]
    if len(by_index) != len(expected):
        raise ObservationPilotValidationError("公式勝者出来事件数がサイドカーと一致しません")
    processing_end = int(source["processing_end_frame_exclusive"])
    for item in expected:
        event = by_index.get(item.game_index)
        if event is None or event["payload"]["winner_side"] != item.winner_side:
            raise ObservationPilotValidationError("公式勝者側または試合indexが不正です")
        timing = event["timing"]
        occurrence = (_frame(item.evidence_start_sec, source), _frame(item.evidence_end_sec, source))
        actual = (timing["occurred_earliest_frame"], timing["occurred_latest_frame"])
        if actual != occurrence or timing["available_frame"] != processing_end:
            raise ObservationPilotValidationError("公式勝者の物理区間または利用可能位置が不正です")
        if event["missing_information"] != ["available_only_posthoc"]:
            raise ObservationPilotValidationError("公式勝者に事後限定の明示がありません")


def _validate_assignments(
    grouped: dict[str, list[dict[str, Any]]],
    source: dict[str, Any],
) -> None:
    assignments = _unique_events(
        grouped["official_game_assignment"], "local_game_index"
    )
    winners = _unique_events(grouped["winner_observed"], "game_index_unverified")
    if int(source["processing_start_frame"]) != 0:
        if assignments:
            raise ObservationPilotValidationError("途中開始処理に公式試合番号があります")
        return
    if set(assignments) != set(winners):
        raise ObservationPilotValidationError("公式試合割当が確認済み勝者と一致しません")
    boundaries = _unique_events(
        grouped["match_boundary_evidence"], "closing_game_index_unverified"
    )
    online = _online_events(grouped, int(source["processing_end_frame_exclusive"]))
    for index, assignment in assignments.items():
        _validate_assignment(index, assignment, winners[index], boundaries, online, source)


def _validate_assignment(
    index: int,
    assignment: dict[str, Any],
    winner: dict[str, Any],
    boundaries: dict[int, dict[str, Any]],
    online: Sequence[dict[str, Any]],
    source: dict[str, Any],
) -> None:
    previous, closing = boundaries.get(index - 1), boundaries.get(index)
    from_seq = 0 if previous is None else int(previous["seq"]) + 1
    to_seq = int(closing["seq"]) if closing is not None else max(
        int(event["seq"]) for event in online
    )
    payload = assignment["payload"]
    if (payload["official_game_number"], payload["from_sequence"], payload["to_sequence"]) != (
        index + 1, from_seq, to_seq
    ):
        raise ObservationPilotValidationError("公式試合番号または通番範囲が不正です")
    expected_causes = {winner["event_id"]}
    expected_causes.update(
        event["event_id"] for event in (previous, closing) if event is not None
    )
    if set(assignment["relations"].get("verified_cause_event_ids", [])) != expected_causes:
        raise ObservationPilotValidationError("公式試合割当の確認済み根拠が不正です")
    if assignment["timing"]["available_frame"] != int(source["processing_end_frame_exclusive"]):
        raise ObservationPilotValidationError("公式試合割当が処理末尾より前に利用可能です")


def _online_events(
    grouped: dict[str, list[dict[str, Any]]], processing_end: int
) -> list[dict[str, Any]]:
    return [
        event for event_type, events in grouped.items()
        for event in events
        if event_type not in {"winner_observed", "official_game_assignment"}
        and int(event["timing"]["available_frame"]) < processing_end
    ]


def _unique_events(
    events: Sequence[dict[str, Any]], payload_key: str
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for event in events:
        index = int(event["payload"][payload_key])
        if index in result:
            raise ObservationPilotValidationError("同じ局所試合indexの出来事が重複しています")
        result[index] = event
    return result


def _frame(seconds: Decimal, source: dict[str, Any]) -> int:
    numerator = seconds * int(source["time_base_denominator"])
    denominator = int(source["time_base_numerator"])
    return int((numerator / denominator).to_integral_value(rounding=ROUND_HALF_UP))


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ObservationPilotValidationError("認識設定がJSONオブジェクトではありません")
    return value


__all__ = [
    "ObservationPilotValidationError",
    "validate_observation_pilot_run",
]
