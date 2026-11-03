"""ReducerStateを公開OverlaySnapshotへ射影する純粋関数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import (
    EvaluationAvailability,
    EvaluationMode,
    FaultCode,
    OverlaySnapshot,
    SNAPSHOT_SCHEMA_VERSION,
)
from .reducer import DisplayState, JobState, LaneState, MatchPhase, ReducerState


@dataclass(frozen=True, slots=True)
class PublicationContext:
    """配信層が注入するpublish時点の値。"""

    stream_seq: int
    published_at_utc: str
    publish_monotonic_ms: int


def _age_ms(anchor_ms: int | None, now_ms: int) -> int | None:
    if anchor_ms is None:
        return None
    return max(0, now_ms - anchor_ms)


def _unavailable_practical() -> dict[str, Any]:
    return {
        "availability": "unavailable",
        "request_id": None,
        "input_generation": None,
        "input_digest": None,
        "calculation_latency_ms": None,
        "calculation_age_ms": None,
        "p1_win_probability": None,
        "p2_win_probability": None,
        "advantage_score": None,
        "is_even": None,
        "origin": None,
        "calibration_id": None,
        "evaluated_positions": None,
    }


def _unavailable_best_action() -> dict[str, Any]:
    return {
        "availability": "unavailable",
        "request_id": None,
        "input_generation": None,
        "input_digest": None,
        "calculation_latency_ms": None,
        "calculation_age_ms": None,
        "p1_position_value": None,
        "p1_position_value_low": None,
        "p1_position_value_high": None,
        "scale": "advantage_score",
        "aggregation_profile_id": None,
        "search_profile_id": None,
        "searched_depth": None,
        "searched_nodes": None,
    }


def _pending_practical(lane: LaneState) -> dict[str, Any]:
    payload = _unavailable_practical()
    payload.update(
        availability="pending",
        request_id=lane.request_id,
        input_generation=lane.input_generation,
        input_digest=lane.input_digest,
    )
    return payload


def _pending_best_action(lane: LaneState) -> dict[str, Any]:
    payload = _unavailable_best_action()
    payload.update(
        availability="pending",
        request_id=lane.request_id,
        input_generation=lane.input_generation,
        input_digest=lane.input_digest,
    )
    return payload


def _available_practical(lane: LaneState, context: PublicationContext) -> dict[str, Any]:
    values = lane.values
    return {
        "availability": "available",
        "request_id": lane.request_id,
        "input_generation": lane.input_generation,
        "input_digest": lane.input_digest,
        "calculation_latency_ms": values.get("calculation_latency_ms"),
        "calculation_age_ms": _age_ms(lane.completed_monotonic_ms, context.publish_monotonic_ms),
        "p1_win_probability": values.get("p1_win_probability"),
        "p2_win_probability": values.get("p2_win_probability"),
        "advantage_score": values.get("advantage_score"),
        "is_even": values.get("is_even"),
        "origin": values.get("origin"),
        "calibration_id": values.get("calibration_id"),
        "evaluated_positions": values.get("evaluated_positions"),
    }


def _available_best_action(lane: LaneState, context: PublicationContext) -> dict[str, Any]:
    values = lane.values
    return {
        "availability": "available",
        "request_id": lane.request_id,
        "input_generation": lane.input_generation,
        "input_digest": lane.input_digest,
        "calculation_latency_ms": values.get("calculation_latency_ms"),
        "calculation_age_ms": _age_ms(lane.completed_monotonic_ms, context.publish_monotonic_ms),
        "p1_position_value": values.get("p1_position_value"),
        "p1_position_value_low": values.get("p1_position_value_low"),
        "p1_position_value_high": values.get("p1_position_value_high"),
        "scale": "advantage_score",
        "aggregation_profile_id": values.get("aggregation_profile_id"),
        "search_profile_id": values.get("search_profile_id"),
        "searched_depth": values.get("searched_depth"),
        "searched_nodes": values.get("searched_nodes"),
    }


def _project_lane(lane: LaneState, context: PublicationContext, *, practical: bool) -> dict[str, Any]:
    if lane.availability is EvaluationAvailability.UNAVAILABLE:
        return _unavailable_practical() if practical else _unavailable_best_action()
    if lane.availability is EvaluationAvailability.PENDING:
        return _pending_practical(lane) if practical else _pending_best_action(lane)
    return _available_practical(lane, context) if practical else _available_best_action(lane, context)


def _has_integrity_fault(state: ReducerState) -> bool:
    return state.match_phase is MatchPhase.INTEGRITY_FAULT or bool(state.fault_codes)


def _must_hide(state: ReducerState) -> bool:
    return (
        state.match_id is None
        or state.match_phase in {MatchPhase.PRE_MATCH, MatchPhase.INTERMISSION, MatchPhase.INTEGRITY_FAULT}
        or state.display_state in {DisplayState.HIDDEN, DisplayState.AWAITING}
        or bool(state.fault_codes)
    )


def _display_payload(state: ReducerState, context: PublicationContext) -> dict[str, Any]:
    if _has_integrity_fault(state):
        visibility, status = "hidden", "integrity_fault"
    elif _must_hide(state):
        visibility, status = "hidden", "waiting"
    elif state.display_state is DisplayState.HOLD:
        visibility, status = "visible", "hold"
    else:
        visibility, status = "visible", state.display_state.value
    in_hold = status == "hold"
    return {
        "visibility": visibility,
        "status": status,
        "update_reason": state.update_reason.value,
        "primary_hold_reason": state.hold_reasons[0].value if in_hold and state.hold_reasons else None,
        "all_hold_reasons": [reason.value for reason in state.hold_reasons] if in_hold else [],
        "hold_started_ms": state.hold_started_source_ms if in_hold else None,
        "hold_elapsed_ms": _age_ms(state.hold_started_monotonic_ms, context.publish_monotonic_ms) if in_hold else None,
    }


def _evaluation_payload(state: ReducerState, context: PublicationContext) -> dict[str, Any]:
    if state.match_phase is MatchPhase.TERMINAL:
        practical = _project_lane(state.practical_lane, context, practical=True)
        best_action = _unavailable_best_action()
    elif state.match_phase is MatchPhase.RESULT or _must_hide(state):
        practical = _unavailable_practical()
        best_action = _unavailable_best_action()
    else:
        practical = _project_lane(state.practical_lane, context, practical=True)
        best_action = _project_lane(state.best_action_lane, context, practical=False)
    if state.match_phase in {MatchPhase.TERMINAL, MatchPhase.RESULT}:
        return {"practical": practical, "best_action": best_action, "player_adjusted": None}
    if state.mode is EvaluationMode.PRACTICAL:
        best_action = _unavailable_best_action()
    elif state.mode is EvaluationMode.BEST_ACTION:
        practical = _unavailable_practical()
    return {"practical": practical, "best_action": best_action, "player_adjusted": None}


def _root_calculation_age(state: ReducerState, context: PublicationContext) -> int | None:
    if _must_hide(state):
        return None
    terminal = state.match_phase is MatchPhase.TERMINAL
    lane = (
        state.best_action_lane
        if state.mode is EvaluationMode.BEST_ACTION and not terminal
        else state.practical_lane
    )
    if lane.availability is not EvaluationAvailability.AVAILABLE:
        return None
    return _age_ms(lane.completed_monotonic_ms, context.publish_monotonic_ms)


def _runtime_payload(state: ReducerState) -> dict[str, Any]:
    practical_depth = int(state.practical_lane.job_state is JobState.QUEUED)
    best_action_depth = int(state.best_action_lane.job_state is JobState.QUEUED)
    worker_status = "starting" if state.runtime_status.value == "starting" else "healthy"
    return {
        "status": state.runtime_status.value,
        "tier": state.tier.value,
        "tier_profile_id": state.tier_profile_id,
        "practical_queue_depth": practical_depth,
        "best_action_queue_depth": best_action_depth,
        "practical_worker_health": worker_status,
        "best_action_worker_health": "disabled" if state.mode is EvaluationMode.PRACTICAL else worker_status,
        "discarded_jobs_since_match_start": state.discarded_jobs_since_match_start,
        "telemetry_health": "healthy",
    }


def _integrity_payload(state: ReducerState) -> dict[str, Any]:
    if not _has_integrity_fault(state):
        return {"status": "ok", "fault_codes": []}
    codes = state.fault_codes or (FaultCode.INTERNAL_INVARIANT_FAILED,)
    return {"status": "fault", "fault_codes": [code.value for code in codes]}


def _terminal_payload(state: ReducerState) -> dict[str, str | None]:
    return {
        "state": state.terminal_state.value,
        "winner": None if state.terminal_winner is None else state.terminal_winner.value,
        "evidence_kind": (
            None if state.terminal_evidence_kind is None else state.terminal_evidence_kind.value
        ),
        "result_code": (
            None if state.terminal_result_code is None else state.terminal_result_code.value
        ),
    }


def project_snapshot(state: ReducerState, context: PublicationContext) -> OverlaySnapshot:
    """状態を進めずに、公開snapshotを一件生成する。"""
    timing = {
        "source_available_frame": state.source_available_frame,
        "source_available_ms": state.source_available_ms,
        "published_at_utc": context.published_at_utc,
        "capture_to_publish_latency_ms": _age_ms(state.source_captured_monotonic_ms, context.publish_monotonic_ms),
        "calculation_age_ms": _root_calculation_age(state, context),
        "last_confirmed_age_ms": _age_ms(state.last_confirmed_monotonic_ms, context.publish_monotonic_ms),
    }
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "identity": {
            "session_id": state.session_id,
            "stream_seq": context.stream_seq,
            "reducer_revision": state.reducer_revision,
            "match_id": state.match_id,
            "match_state_seq": state.match_state_seq,
            "input_generation": state.input_generation,
        },
        "timing": timing,
        "display": _display_payload(state, context),
        "integrity": _integrity_payload(state),
        "mode": {"evaluation_mode": state.mode.value, "practical_basis": "board_baseline"},
        "evaluations": _evaluation_payload(state, context),
        "input": {
            "event_seq": state.last_event_seq,
            "p1_board_provenance": state.p1_board_provenance.value,
            "p2_board_provenance": state.p2_board_provenance.value,
            "recognition_quality": {
                "status": state.recognition_status.value,
                "reason_codes": [reason.value for reason in state.recognition_reasons],
            },
            "unresolved_physics": list(state.unresolved_physics),
            "physical_prediction_used": state.physical_prediction_used,
        },
        "terminal": _terminal_payload(state),
        "runtime": _runtime_payload(state),
        "assets": dict(state.assets),
    }
    return OverlaySnapshot.from_mapping(payload)
