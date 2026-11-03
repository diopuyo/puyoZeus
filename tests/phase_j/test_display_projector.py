"""Phase J display projectorの純粋射影試験。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from src.phase_j.contracts import EvaluationAvailability, FaultCode, HoldReason, UpdateReason
from src.phase_j.display_projector import PublicationContext, project_snapshot
from src.phase_j.reducer import (
    Bootstrap,
    DisplayState,
    MatchPhase,
    ReducerEvent,
    ReducerEventKind,
    ReducerState,
    initial_state,
    reduce_event,
)
from src.phase_j.validator import validate_snapshot


def _event(kind: ReducerEventKind, seq: int, **payload: Any) -> ReducerEvent:
    return ReducerEvent(kind, seq, f"event:{seq}:{kind.value}", payload)


def _context(*, stream_seq: int = 0, monotonic_ms: int = 0) -> PublicationContext:
    return PublicationContext(stream_seq, "2026-09-04T00:00:00Z", monotonic_ms)


def _state() -> ReducerState:
    return initial_state(Bootstrap(session_id="session-1", capture_session_id="capture-1"))


def _active_state() -> ReducerState:
    state = reduce_event(
        _state(),
        _event(ReducerEventKind.FORMAL_BOUNDARY, 1, match_id="match-1"),
    ).state
    observation = _event(
        ReducerEventKind.OBSERVATION_AVAILABLE,
        2,
        capture_session_id="capture-1",
        capture_seq=1,
        capture_digest="capture-1:1",
        source_available_frame=900,
        source_available_ms=30_000,
        captured_monotonic_ms=4_990,
        observed_monotonic_ms=5_000,
        recognition_status="trusted",
        recognition_reasons=[],
        p1_stable=True,
        p2_stable=True,
        p1_board_provenance="confirmed",
        p2_board_provenance="confirmed",
        unresolved_physics=[],
        physical_prediction_used=False,
        input_digest="input-1",
    )
    return reduce_event(state, observation).state


def _live_state() -> ReducerState:
    state = _active_state()
    lane = state.practical_lane
    prediction = _event(
        ReducerEventKind.PREDICTION_COMPLETED,
        3,
        lane="practical",
        request_id=lane.request_id,
        input_generation=lane.input_generation,
        input_digest=lane.input_digest,
        completed_monotonic_ms=5_100,
        evaluation={
            "calculation_latency_ms": 80,
            "p1_win_probability": 0.6,
            "p2_win_probability": 0.4,
            "advantage_score": 20,
            "is_even": False,
            "origin": "model",
            "calibration_id": "calibration-v1",
            "evaluated_positions": 1,
        },
    )
    return reduce_event(state, prediction).state


def _terminal_state(winner: str = "1P") -> ReducerState:
    state = _live_state()
    result_code = "p1_win" if winner == "1P" else "p2_win"
    event = _event(
        ReducerEventKind.TERMINAL_EVIDENCE, 4,
        match_id=state.match_id, capture_session_id=state.capture_session_id,
        asset_bundle_id=state.asset_bundle_id, source_available_frame=901,
        source_available_ms=30_033, captured_monotonic_ms=5_190,
        observed_monotonic_ms=5_200,
        evidence_kind="visual_result_logo_bilateral_2x2",
        winner=winner, result_code=result_code,
    )
    return reduce_event(state, event).state


def test_initial_projection_is_legal_hidden_waiting_snapshot() -> None:
    snapshot = project_snapshot(_state(), _context()).to_mapping()
    assert snapshot["display"]["visibility"] == "hidden"
    assert snapshot["display"]["status"] == "waiting"
    assert snapshot["identity"]["stream_seq"] == 0
    for lane_name in ("practical", "best_action"):
        lane = snapshot["evaluations"][lane_name]
        assert lane["availability"] == "unavailable"
        assert lane["request_id"] is lane["input_generation"] is lane["input_digest"] is None


def test_publication_context_controls_stream_sequence_without_mutating_state() -> None:
    state = _state()
    first = project_snapshot(state, _context(stream_seq=3)).to_mapping()
    second = project_snapshot(state, _context(stream_seq=4)).to_mapping()
    assert first["identity"]["stream_seq"] == 3
    assert second["identity"]["stream_seq"] == 4
    assert state.reducer_revision == 0
    assert "stream_seq" not in ReducerState.__dataclass_fields__


def test_live_projection_exposes_committed_value_and_monotonic_ages() -> None:
    snapshot = project_snapshot(_live_state(), _context(stream_seq=7, monotonic_ms=5_250)).to_mapping()
    assert snapshot["display"]["status"] == "live"
    assert snapshot["evaluations"]["practical"]["availability"] == "available"
    assert snapshot["evaluations"]["practical"]["p1_win_probability"] == 0.6
    assert snapshot["evaluations"]["practical"]["calculation_age_ms"] == 150
    assert snapshot["timing"]["capture_to_publish_latency_ms"] == 260
    assert snapshot["timing"]["last_confirmed_age_ms"] == 250


def test_hold_uses_source_time_for_start_and_monotonic_time_for_elapsed() -> None:
    state = _live_state()
    held = reduce_event(
        state,
        _event(
            ReducerEventKind.HOLD_REASON_ADDED,
            4,
            reason="calculation_pending",
            source_available_ms=30_500,
            monotonic_ms=5_200,
        ),
    ).state
    snapshot = project_snapshot(held, _context(monotonic_ms=5_900)).to_mapping()
    assert snapshot["display"]["visibility"] == "visible"
    assert snapshot["display"]["status"] == "hold"
    assert snapshot["display"]["hold_started_ms"] == 30_500
    assert snapshot["display"]["hold_elapsed_ms"] == 700
    assert snapshot["evaluations"]["practical"]["availability"] == "available"


def test_expired_hold_remains_visible_and_nulls_target_lane_values() -> None:
    state = _live_state()
    held = reduce_event(
        state,
        _event(
            ReducerEventKind.HOLD_REASON_ADDED,
            4,
            reason="calculation_pending",
            source_available_ms=30_500,
            monotonic_ms=5_000,
        ),
    ).state
    first = reduce_event(
        held,
        _event(ReducerEventKind.TIMER_FIRED, 5, timer_id=held.hold_timer_id, fired_monotonic_ms=6_000),
    ).state
    expired = reduce_event(
        first,
        _event(ReducerEventKind.TIMER_FIRED, 6, timer_id=first.hold_timer_id, fired_monotonic_ms=7_000),
    ).state
    snapshot = project_snapshot(expired, _context(monotonic_ms=7_000)).to_mapping()
    assert snapshot["display"]["status"] == "hold"
    assert snapshot["display"]["update_reason"] == "hold_expired"
    assert snapshot["evaluations"]["practical"]["availability"] == "pending"
    assert snapshot["evaluations"]["practical"]["p1_win_probability"] is None
    assert snapshot["display"]["primary_hold_reason"] == "calculation_pending"


def test_integrity_fault_projection_is_hidden_and_unavailable() -> None:
    state = reduce_event(_state(), _event(ReducerEventKind.CONFIG_CHANGE_REQUESTED, 1)).state
    failed = reduce_event(state, _event(ReducerEventKind.CONFIG_CHANGE_REQUESTED, 3)).state
    assert failed.fault_codes == (FaultCode.SEQUENCE_GAP,)
    snapshot = project_snapshot(failed, _context()).to_mapping()
    assert snapshot["display"] == {
        "visibility": "hidden",
        "status": "integrity_fault",
        "update_reason": "integrity_fault",
        "primary_hold_reason": None,
        "all_hold_reasons": [],
        "hold_started_ms": None,
        "hold_elapsed_ms": None,
    }
    assert snapshot["integrity"] == {"status": "fault", "fault_codes": ["sequence_gap"]}
    assert snapshot["evaluations"]["practical"]["availability"] == "unavailable"


def test_intermission_projection_never_leaks_previous_match_value() -> None:
    state = _live_state()
    intermission = reduce_event(state, _event(ReducerEventKind.INTERMISSION, 4)).state
    snapshot = project_snapshot(intermission, _context(monotonic_ms=6_000)).to_mapping()
    assert snapshot["display"]["visibility"] == "hidden"
    assert snapshot["display"]["status"] == "waiting"
    assert snapshot["evaluations"]["practical"]["availability"] == "unavailable"
    assert intermission.practical_lane.availability is EvaluationAvailability.UNAVAILABLE


def test_terminal_projection_replaces_model_with_bilateral_exact_fact() -> None:
    snapshot = project_snapshot(
        _terminal_state("2P"), _context(stream_seq=8, monotonic_ms=5_300),
    ).to_mapping()

    assert snapshot["display"]["status"] == "terminal_fact"
    assert snapshot["terminal"] == {
        "state": "confirmed", "winner": "2P",
        "evidence_kind": "visual_result_logo_bilateral_2x2",
        "result_code": "p2_win",
    }
    practical = snapshot["evaluations"]["practical"]
    assert practical["p1_win_probability"] == 0.0
    assert practical["p2_win_probability"] == 1.0
    assert practical["advantage_score"] == -100.0
    assert snapshot["evaluations"]["best_action"]["availability"] == "unavailable"
    assert validate_snapshot(snapshot).is_valid


def test_result_projection_removes_probability_but_keeps_result_code() -> None:
    terminal = _terminal_state("1P")
    result = reduce_event(
        terminal,
        _event(
            ReducerEventKind.TIMER_FIRED, 5,
            timer_id=terminal.terminal_timer_id, fired_monotonic_ms=6_200,
        ),
    ).state
    snapshot = project_snapshot(
        result, _context(stream_seq=9, monotonic_ms=6_200),
    ).to_mapping()

    assert snapshot["display"]["status"] == "result"
    assert snapshot["terminal"]["result_code"] == "p1_win"
    assert snapshot["evaluations"]["practical"]["availability"] == "unavailable"
    assert snapshot["evaluations"]["practical"]["p1_win_probability"] is None
    assert validate_snapshot(snapshot).is_valid


@pytest.mark.parametrize(
    "state",
    [
        replace(
            _live_state(),
            match_id=None,
            display_state=DisplayState.HOLD,
            hold_reasons=(HoldReason.CALCULATION_PENDING,),
            hold_started_source_ms=30_500,
            hold_started_monotonic_ms=5_200,
            update_reason=UpdateReason.HOLD_STARTED,
        ),
        replace(
            _live_state(),
            match_phase=MatchPhase.PRE_MATCH,
            display_state=DisplayState.HOLD,
            hold_reasons=(HoldReason.CALCULATION_PENDING,),
            hold_started_source_ms=30_500,
            hold_started_monotonic_ms=5_200,
            update_reason=UpdateReason.HOLD_STARTED,
        ),
    ],
    ids=("match_idなし", "pre_match"),
)
def test_p1_4_pre_match_state_never_publishes_visible_hold(state: ReducerState) -> None:
    snapshot = project_snapshot(state, _context(monotonic_ms=5_900)).to_mapping()
    assert snapshot["display"]["visibility"] == "hidden"
    assert snapshot["display"]["status"] == "waiting"
    assert snapshot["display"]["primary_hold_reason"] is None
    assert snapshot["display"]["all_hold_reasons"] == []
    assert snapshot["display"]["hold_started_ms"] is None
    assert snapshot["display"]["hold_elapsed_ms"] is None
    assert snapshot["evaluations"]["practical"]["availability"] == "unavailable"
    assert validate_snapshot(snapshot).is_valid


@pytest.mark.parametrize(
    ("phase", "fault_codes", "expected_status"),
    [
        (MatchPhase.INTERMISSION, (), "waiting"),
        (MatchPhase.INTEGRITY_FAULT, (FaultCode.SEQUENCE_GAP,), "integrity_fault"),
        (MatchPhase.INTEGRITY_FAULT, (), "integrity_fault"),
    ],
    ids=("intermission", "fault", "fault_code欠落"),
)
def test_p1_4_nonactive_phase_strips_stale_hold_metadata(
    phase: MatchPhase,
    fault_codes: tuple[FaultCode, ...],
    expected_status: str,
) -> None:
    inconsistent = replace(
        _live_state(),
        match_phase=phase,
        display_state=DisplayState.HOLD,
        fault_codes=fault_codes,
        hold_reasons=(HoldReason.CALCULATION_PENDING,),
        hold_started_source_ms=30_500,
        hold_started_monotonic_ms=5_200,
        update_reason=UpdateReason.HOLD_STARTED,
    )
    snapshot = project_snapshot(inconsistent, _context(monotonic_ms=5_900)).to_mapping()
    assert snapshot["display"]["visibility"] == "hidden"
    assert snapshot["display"]["status"] == expected_status
    assert snapshot["display"]["primary_hold_reason"] is None
    assert snapshot["display"]["all_hold_reasons"] == []
    assert snapshot["display"]["hold_started_ms"] is None
    assert snapshot["display"]["hold_elapsed_ms"] is None
    assert snapshot["evaluations"]["practical"]["availability"] == "unavailable"
    assert validate_snapshot(snapshot).is_valid


def test_representative_projected_states_all_pass_public_validator() -> None:
    """reducerの主要状態が公開契約と実際に接続できる。"""
    initial = _state()
    awaiting = _active_state()
    live = _live_state()
    held = reduce_event(
        live,
        _event(
            ReducerEventKind.HOLD_REASON_ADDED,
            4,
            reason="calculation_pending",
            source_available_ms=30_500,
            monotonic_ms=5_200,
        ),
    ).state
    first_tick = reduce_event(
        held,
        _event(ReducerEventKind.TIMER_FIRED, 5, timer_id=held.hold_timer_id, fired_monotonic_ms=6_200),
    ).state
    expired = reduce_event(
        first_tick,
        _event(ReducerEventKind.TIMER_FIRED, 6, timer_id=first_tick.hold_timer_id, fired_monotonic_ms=7_200),
    ).state
    intermission = reduce_event(live, _event(ReducerEventKind.INTERMISSION, 4)).state
    before_fault = reduce_event(
        initial, _event(ReducerEventKind.CONFIG_CHANGE_REQUESTED, 1),
    ).state
    fault = reduce_event(
        before_fault, _event(ReducerEventKind.CONFIG_CHANGE_REQUESTED, 3),
    ).state
    cases = ((initial, 0), (awaiting, 5_000), (live, 5_250), (held, 5_900),
             (expired, 7_200), (intermission, 6_000), (fault, 0))
    for index, (state, monotonic_ms) in enumerate(cases):
        snapshot = project_snapshot(state, _context(stream_seq=index, monotonic_ms=monotonic_ms))
        report = validate_snapshot(snapshot)
        assert report.is_valid, (index, report.issues)
