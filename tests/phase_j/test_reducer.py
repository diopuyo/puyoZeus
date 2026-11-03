"""Phase J reducerのR01〜R10契約試験。"""

from __future__ import annotations

from typing import Any

import pytest

from src.phase_j.contracts import (
    EvaluationAvailability,
    EvaluationMode,
    FaultCode,
    HoldReason,
    RuntimeTier,
    UpdateReason,
)
from src.phase_j.display_projector import PublicationContext, project_snapshot
from src.phase_j.reducer import (
    Bootstrap,
    DisplayState,
    MatchPhase,
    ReducerEvent,
    ReducerEventKind,
    ReducerIntentKind,
    ReducerState,
    initial_state,
    reduce_event,
)
from src.phase_j.validator import validate_snapshot


def _state(*, session_id: str = "session-1") -> ReducerState:
    return initial_state(Bootstrap(session_id=session_id, capture_session_id="capture-1"))


def _event(
    kind: ReducerEventKind,
    seq: int,
    *,
    digest: str | None = None,
    **payload: Any,
) -> ReducerEvent:
    return ReducerEvent(
        kind=kind,
        event_seq=seq,
        content_digest=digest or f"event:{seq}:{kind.value}",
        payload=payload,
    )


def _boundary(state: ReducerState, seq: int = 1, match_id: str = "match-1") -> ReducerState:
    event = _event(ReducerEventKind.FORMAL_BOUNDARY, seq, match_id=match_id)
    return reduce_event(state, event).state


def _observation(
    seq: int,
    *,
    capture_seq: int,
    input_digest: str,
    trusted: bool = True,
    stable: bool = True,
    capture_session_id: str = "capture-1",
    source_ms: int = 1_000,
    monotonic_ms: int = 5_000,
) -> ReducerEvent:
    return _event(
        ReducerEventKind.OBSERVATION_AVAILABLE,
        seq,
        capture_session_id=capture_session_id,
        capture_seq=capture_seq,
        capture_digest=f"capture:{capture_session_id}:{capture_seq}",
        source_available_frame=capture_seq * 10,
        source_available_ms=source_ms,
        captured_monotonic_ms=monotonic_ms - 10,
        observed_monotonic_ms=monotonic_ms,
        recognition_status="trusted" if trusted else "untrusted",
        recognition_reasons=[] if trusted else ["board_unstable"],
        p1_stable=stable,
        p2_stable=stable,
        p1_board_provenance="confirmed" if stable else "unknown",
        p2_board_provenance="confirmed" if stable else "unknown",
        unresolved_physics=[],
        physical_prediction_used=False,
        input_digest=input_digest,
    )


def _prediction(state: ReducerState, seq: int, *, completed_ms: int = 5_100) -> ReducerEvent:
    lane = state.practical_lane
    return _event(
        ReducerEventKind.PREDICTION_COMPLETED,
        seq,
        lane="practical",
        request_id=lane.request_id,
        input_generation=lane.input_generation,
        input_digest=lane.input_digest,
        completed_monotonic_ms=completed_ms,
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


def _best_action_prediction(state: ReducerState, seq: int) -> ReducerEvent:
    lane = state.best_action_lane
    return _event(
        ReducerEventKind.PREDICTION_COMPLETED,
        seq,
        lane="best_action",
        request_id=lane.request_id,
        input_generation=lane.input_generation,
        input_digest=lane.input_digest,
        completed_monotonic_ms=5_100,
        evaluation={
            "calculation_latency_ms": 90,
            "p1_position_value": 12,
            "p1_position_value_low": 5,
            "p1_position_value_high": 20,
            "aggregation_profile_id": "aggregate-v1",
            "search_profile_id": "search-v1",
            "searched_depth": 4,
            "searched_nodes": 120,
        },
    )


def _pending_state(mode: EvaluationMode = EvaluationMode.PRACTICAL) -> ReducerState:
    state = initial_state(
        Bootstrap(session_id="session-1", capture_session_id="capture-1", mode=mode)
    )
    state = _boundary(state)
    return reduce_event(state, _observation(2, capture_seq=1, input_digest="input-1")).state


def _ready_state() -> ReducerState:
    state = _pending_state()
    return reduce_event(state, _prediction(state, 3)).state


def _terminal_evidence(
    state: ReducerState, seq: int, *, winner: str = "1P",
    result_code: str = "p1_win", evidence_kind: str = "visual_result_logo_bilateral_2x2",
) -> ReducerEvent:
    return _event(
        ReducerEventKind.TERMINAL_EVIDENCE, seq,
        match_id=state.match_id, capture_session_id=state.capture_session_id,
        asset_bundle_id=state.asset_bundle_id, source_available_frame=12,
        source_available_ms=1_200, captured_monotonic_ms=5_990,
        observed_monotonic_ms=6_000, evidence_kind=evidence_kind,
        winner=winner, result_code=result_code,
    )


def _physical_ready_state() -> ReducerState:
    state = _boundary(_state())
    observation = _observation(2, capture_seq=1, input_digest="physical-input")
    observed_payload = dict(observation.payload)
    observed_payload.update(
        p1_board_provenance="physics_projected",
        p2_board_provenance="confirmed",
        physical_prediction_used=True,
    )
    observation = ReducerEvent(
        observation.kind, observation.event_seq, observation.content_digest, observed_payload
    )
    pending = reduce_event(state, observation).state
    prediction = _prediction(pending, 3)
    prediction_payload = dict(prediction.payload)
    prediction_payload["evaluation"] = dict(prediction_payload["evaluation"])
    prediction_payload["evaluation"]["origin"] = "physical_prediction"
    return reduce_event(
        pending,
        ReducerEvent(prediction.kind, prediction.event_seq, prediction.content_digest, prediction_payload),
    ).state


def _assert_projected_valid(state: ReducerState, stream_seq: int = 20) -> None:
    context = PublicationContext(stream_seq, "2026-09-04T00:00:00Z", 6_000)
    report = validate_snapshot(project_snapshot(state, context))
    assert report.is_valid, report.issues


def test_r01_bootstrap_is_hidden_before_any_event() -> None:
    state = _state()
    assert state.display_state is DisplayState.HIDDEN
    assert state.match_phase is MatchPhase.PRE_MATCH
    assert state.reducer_revision == state.match_state_seq == state.input_generation == 0
    assert state.practical_lane.availability is EvaluationAvailability.UNAVAILABLE
    assert "stream_seq" not in ReducerState.__dataclass_fields__


def test_r02_only_trusted_bilateral_stable_input_issues_generation_and_job() -> None:
    state = _boundary(_state())
    untrusted = reduce_event(
        state,
        _observation(2, capture_seq=1, input_digest="bad", trusted=False, stable=False),
    )
    assert untrusted.state.input_generation == state.input_generation
    assert all(intent.kind is not ReducerIntentKind.START_JOB for intent in untrusted.intents)

    stable = reduce_event(state, _observation(2, capture_seq=1, input_digest="input-1"))
    assert stable.state.input_generation == state.input_generation + 1
    assert stable.state.practical_lane.availability is EvaluationAvailability.PENDING
    assert any(intent.kind is ReducerIntentKind.START_JOB for intent in stable.intents)

    unchanged = reduce_event(stable.state, _observation(3, capture_seq=2, input_digest="input-1"))
    assert unchanged.state.input_generation == stable.state.input_generation
    assert all(intent.kind is not ReducerIntentKind.START_JOB for intent in unchanged.intents)


@pytest.mark.parametrize(
    ("seq", "digest", "expected"),
    [
        (3, "event-3", FaultCode.SEQUENCE_GAP),
        (0, "event-0", FaultCode.SEQUENCE_REVERSED),
        (1, "different-content", FaultCode.ID_CONTENT_CONFLICT),
    ],
)
def test_r03_event_sequence_faults_fail_closed(seq: int, digest: str, expected: FaultCode) -> None:
    state = reduce_event(_state(), _event(ReducerEventKind.CONFIG_CHANGE_REQUESTED, 1)).state
    reduction = reduce_event(state, _event(ReducerEventKind.CONFIG_CHANGE_REQUESTED, seq, digest=digest))
    assert reduction.state.match_phase is MatchPhase.INTEGRITY_FAULT
    assert expected in reduction.state.fault_codes
    assert reduction.state.practical_lane.availability is EvaluationAvailability.UNAVAILABLE
    assert reduction.state.last_event_seq == seq
    assert reduction.state.last_event_digest == digest
    assert reduction.state.hold_timer_deadline_monotonic_ms is None
    assert reduction.state.hold_timer_tick == 0


def test_r03_capture_gap_enters_quality_hold_without_replacing_confirmed_input() -> None:
    state = _boundary(_state())
    state = reduce_event(state, _observation(2, capture_seq=1, input_digest="input-1")).state
    reduction = reduce_event(state, _observation(3, capture_seq=3, input_digest="input-2"))
    assert reduction.accepted is False
    assert reduction.state.display_state is DisplayState.HOLD
    assert reduction.state.input_digest == "input-1"
    assert [item.value for item in reduction.state.recognition_reasons] == ["capture_gap"]
    assert reduction.intents[0].payload["reason"] == "capture_sequence_gap"


def test_r03_mid_match_asset_change_is_integrity_fault() -> None:
    state = _boundary(_state())
    event = _observation(2, capture_seq=1, input_digest="input-1")
    payload = dict(event.payload)
    payload["asset_bundle_id"] = "unexpected-bundle"
    changed = ReducerEvent(event.kind, event.event_seq, event.content_digest, payload)
    reduction = reduce_event(state, changed)
    assert reduction.state.fault_codes == (FaultCode.ASSET_BUNDLE_CHANGED_MID_MATCH,)
    assert reduction.state.display_state is DisplayState.HIDDEN


def test_r04_formal_boundary_and_intermission_clear_match_scoped_state() -> None:
    ready = _ready_state()
    held = reduce_event(
        ready,
        _event(
            ReducerEventKind.HOLD_REASON_ADDED,
            4,
            reason="calculation_pending",
            source_available_ms=1_200,
            monotonic_ms=5_200,
        ),
    ).state
    boundary = reduce_event(held, _event(ReducerEventKind.FORMAL_BOUNDARY, 5, match_id="match-2")).state
    assert boundary.match_id == "match-2"
    assert boundary.match_state_seq > held.match_state_seq
    assert boundary.display_state is DisplayState.HIDDEN
    assert boundary.hold_reasons == ()
    assert boundary.practical_lane.availability is EvaluationAvailability.UNAVAILABLE
    assert boundary.hold_timer_deadline_monotonic_ms is None
    assert boundary.hold_timer_tick == 0
    boundary_snapshot = project_snapshot(
        boundary,
        PublicationContext(9, "2026-09-04T00:00:00Z", 9_000),
    ).to_mapping()
    assert boundary_snapshot["timing"]["source_available_frame"] is None
    assert boundary_snapshot["timing"]["source_available_ms"] is None
    assert boundary_snapshot["timing"]["last_confirmed_age_ms"] is None
    assert boundary_snapshot["input"]["p1_board_provenance"] == "unknown"
    assert boundary_snapshot["input"]["p2_board_provenance"] == "unknown"
    assert boundary_snapshot["input"]["unresolved_physics"] == []
    assert boundary_snapshot["input"]["physical_prediction_used"] is False

    intermission = reduce_event(boundary, _event(ReducerEventKind.INTERMISSION, 6)).state
    assert intermission.match_phase is MatchPhase.INTERMISSION
    assert intermission.display_state is DisplayState.HIDDEN
    assert intermission.source_available_ms is None
    assert intermission.hold_timer_deadline_monotonic_ms is None


def test_r05_untrusted_observation_freezes_last_confirmed_evaluation() -> None:
    ready = _ready_state()
    previous_values = dict(ready.practical_lane.values)
    reduction = reduce_event(
        ready,
        _observation(4, capture_seq=2, input_digest="untrusted", trusted=False, stable=False),
    )
    assert reduction.state.display_state is DisplayState.HOLD
    assert reduction.state.input_digest == "input-1"
    assert dict(reduction.state.practical_lane.values) == previous_values
    assert reduction.state.physical_prediction_used is False


def test_r06_timer_republishes_then_expires_without_mixing_clocks() -> None:
    ready = _ready_state()
    held = reduce_event(
        ready,
        _event(
            ReducerEventKind.HOLD_REASON_ADDED,
            4,
            reason="calculation_pending",
            source_available_ms=30_500,
            monotonic_ms=5_000,
        ),
    ).state
    assert held.hold_started_source_ms == 30_500
    assert held.hold_started_monotonic_ms == 5_000

    stale = reduce_event(
        held,
        _event(ReducerEventKind.TIMER_FIRED, 5, timer_id="old", fired_monotonic_ms=6_000),
    )
    assert stale.accepted is False
    first = reduce_event(
        stale.state,
        _event(ReducerEventKind.TIMER_FIRED, 6, timer_id=held.hold_timer_id, fired_monotonic_ms=6_000),
    )
    assert first.state.update_reason is UpdateReason.TIMER_ELAPSED
    assert first.state.hold_expired is False
    assert any(intent.kind is ReducerIntentKind.PUBLISH_SNAPSHOT for intent in first.intents)

    expired = reduce_event(
        first.state,
        _event(ReducerEventKind.TIMER_FIRED, 7, timer_id=first.state.hold_timer_id, fired_monotonic_ms=7_000),
    )
    assert expired.state.display_state is DisplayState.HOLD
    assert expired.state.update_reason is UpdateReason.HOLD_EXPIRED
    assert expired.state.practical_lane.availability is EvaluationAvailability.PENDING
    assert expired.state.hold_reasons == (HoldReason.CALCULATION_PENDING,)


def test_r07_hold_reason_priority_and_resolution_are_deterministic() -> None:
    state = _boundary(_state())
    first = reduce_event(
        state,
        _event(
            ReducerEventKind.HOLD_REASON_ADDED,
            2,
            reason="calculation_pending",
            source_available_ms=100,
            monotonic_ms=1_000,
        ),
    ).state
    second = reduce_event(
        first,
        _event(
            ReducerEventKind.HOLD_REASON_ADDED,
            3,
            reason="terminal_confirmation_pending",
            monotonic_ms=1_100,
        ),
    ).state
    assert second.hold_reasons == (
        HoldReason.TERMINAL_CONFIRMATION_PENDING,
        HoldReason.CALCULATION_PENDING,
    )
    resolved = reduce_event(
        second,
        _event(ReducerEventKind.HOLD_REASON_RESOLVED, 4, reason="terminal_confirmation_pending"),
    ).state
    assert resolved.hold_reasons == (HoldReason.CALCULATION_PENDING,)


def test_r08_disconnect_new_capture_session_and_stale_frame_rejection() -> None:
    state = _boundary(_state())
    state = reduce_event(state, _observation(2, capture_seq=1, input_digest="input-1")).state
    disconnected = reduce_event(
        state,
        _event(ReducerEventKind.CAPTURE_STATUS, 3, status="disconnected"),
    ).state
    assert disconnected.display_state is DisplayState.HIDDEN
    assert disconnected.capture_connected is False
    assert disconnected.practical_lane.availability is EvaluationAvailability.UNAVAILABLE
    assert disconnected.best_action_lane.availability is EvaluationAvailability.UNAVAILABLE
    assert disconnected.p1_board_provenance.value == "unknown"
    assert disconnected.physical_prediction_used is False
    assert disconnected.source_available_ms is None
    _assert_projected_valid(disconnected)

    connected = reduce_event(
        disconnected,
        _event(ReducerEventKind.CAPTURE_STATUS, 4, status="connected", capture_session_id="capture-2"),
    ).state
    assert connected.capture_session_id == "capture-2"
    stale = reduce_event(
        connected,
        _observation(5, capture_seq=2, input_digest="old", capture_session_id="capture-1"),
    )
    assert stale.accepted is False
    assert stale.state.capture_session_id == "capture-2"
    assert stale.state.display_state is DisplayState.HIDDEN


def test_r09_config_is_applied_only_at_next_formal_boundary() -> None:
    state = _boundary(_state())
    requested = reduce_event(
        state,
        _event(
            ReducerEventKind.CONFIG_CHANGE_REQUESTED,
            2,
            mode="both",
            tier="high_accuracy",
            tier_profile_id="high-v1",
            asset_bundle_id="bundle-2",
        ),
    ).state
    assert requested.mode is EvaluationMode.PRACTICAL
    assert requested.tier is RuntimeTier.STANDARD
    assert requested.update_reason is UpdateReason.CONFIG_PENDING

    applied = reduce_event(
        requested,
        _event(ReducerEventKind.FORMAL_BOUNDARY, 3, match_id="match-2"),
    ).state
    assert applied.mode is EvaluationMode.BOTH
    assert applied.tier is RuntimeTier.HIGH_ACCURACY
    assert applied.tier_profile_id == "high-v1"
    assert applied.asset_bundle_id == "bundle-2"
    assert not applied.pending_config


def test_r10_same_event_sequence_produces_identical_states() -> None:
    events = (
        _event(ReducerEventKind.FORMAL_BOUNDARY, 1, match_id="match-1"),
        _observation(2, capture_seq=1, input_digest="input-1"),
    )
    left = _state()
    right = _state()
    left_snapshots = []
    right_snapshots = []
    for index, event in enumerate(events, start=1):
        left = reduce_event(left, event).state
        right = reduce_event(right, event).state
        context = PublicationContext(index, "2026-09-04T00:00:00Z", index * 1_000)
        left_snapshots.append(project_snapshot(left, context).to_mapping())
        right_snapshots.append(project_snapshot(right, context).to_mapping())
    assert left == right
    assert left_snapshots == right_snapshots


def test_new_session_resets_all_sequences_and_invalidates_old_state() -> None:
    ready = _ready_state()
    event = _event(
        ReducerEventKind.NEW_SESSION,
        99,
        session_id="session-2",
        capture_session_id="capture-2",
    )
    reduction = reduce_event(ready, event)
    assert reduction.state.session_id == "session-2"
    assert reduction.state.reducer_revision == 0
    assert reduction.state.match_state_seq == reduction.state.input_generation == 0
    assert reduction.state.display_state is DisplayState.HIDDEN
    assert any(intent.kind is ReducerIntentKind.INVALIDATE_JOBS for intent in reduction.intents)


def test_prediction_resolves_only_calculation_pending_while_other_hold_remains() -> None:
    ready = _pending_state()
    recognizing = reduce_event(
        ready,
        _event(
            ReducerEventKind.HOLD_REASON_ADDED,
            3,
            reason="recognition_unreliable",
            source_available_ms=1_100,
            monotonic_ms=5_200,
        ),
    ).state
    held = reduce_event(
        recognizing,
        _event(ReducerEventKind.HOLD_REASON_ADDED, 4, reason="calculation_pending", monotonic_ms=5_250),
    ).state
    reduction = reduce_event(held, _prediction(held, 5, completed_ms=5_300))
    assert reduction.state.display_state is DisplayState.HOLD
    assert reduction.state.hold_reasons == (HoldReason.RECOGNITION_UNRELIABLE,)
    assert reduction.state.update_reason is UpdateReason.HOLD_REASON_CHANGED
    assert reduction.state.hold_timer_id == held.hold_timer_id
    assert all(intent.kind is not ReducerIntentKind.CANCEL_TIMER for intent in reduction.intents)
    _assert_projected_valid(reduction.state)


def test_prediction_clears_calculation_only_hold_and_cancels_timer() -> None:
    ready = _pending_state()
    held = reduce_event(
        ready,
        _event(
            ReducerEventKind.HOLD_REASON_ADDED,
            3,
            reason="calculation_pending",
            source_available_ms=1_100,
            monotonic_ms=5_200,
        ),
    ).state
    reduction = reduce_event(held, _prediction(held, 4, completed_ms=5_300))
    assert reduction.state.display_state is DisplayState.LIVE
    assert reduction.state.hold_reasons == ()
    assert reduction.state.hold_started_source_ms is None
    assert reduction.state.hold_timer_id is None
    assert any(intent.kind is ReducerIntentKind.CANCEL_TIMER for intent in reduction.intents)
    _assert_projected_valid(reduction.state)


def _latched_state(phase: MatchPhase) -> ReducerState:
    ready = _ready_state()
    if phase is MatchPhase.INTERMISSION:
        return reduce_event(ready, _event(ReducerEventKind.INTERMISSION, 4)).state
    gap = _event(ReducerEventKind.CONFIG_CHANGE_REQUESTED, 5, digest="gap")
    return reduce_event(ready, gap).state


@pytest.mark.parametrize("phase", [MatchPhase.INTERMISSION, MatchPhase.INTEGRITY_FAULT])
@pytest.mark.parametrize(
    "event_kind",
    [ReducerEventKind.OBSERVATION_AVAILABLE, ReducerEventKind.PREDICTION_COMPLETED],
)
def test_intermission_and_integrity_fault_are_latched_until_boundary(
    phase: MatchPhase,
    event_kind: ReducerEventKind,
) -> None:
    latched = _latched_state(phase)
    seq = int(latched.last_event_seq) + 1
    event = (
        _observation(seq, capture_seq=2, input_digest="blocked")
        if event_kind is ReducerEventKind.OBSERVATION_AVAILABLE
        else _prediction(latched, seq)
    )
    rejected = reduce_event(latched, event)
    assert rejected.accepted is False
    assert rejected.state.match_phase is phase
    assert rejected.state.display_state is DisplayState.HIDDEN
    assert rejected.intents[0].kind is ReducerIntentKind.REJECT_EVENT
    assert rejected.intents[0].payload["reason"] == f"{phase.value}_latched"
    _assert_projected_valid(rejected.state)
    boundary = reduce_event(
        rejected.state,
        _event(ReducerEventKind.FORMAL_BOUNDARY, seq + 1, match_id="match-2"),
    )
    assert boundary.state.match_phase is MatchPhase.PRE_MATCH


@pytest.mark.parametrize("phase", [MatchPhase.INTERMISSION, MatchPhase.INTEGRITY_FAULT])
def test_new_session_is_the_other_legal_latch_exit(phase: MatchPhase) -> None:
    latched = _latched_state(phase)
    restarted = reduce_event(
        latched,
        _event(ReducerEventKind.NEW_SESSION, 99, session_id="session-2"),
    )
    assert restarted.accepted is True
    assert restarted.state.session_id == "session-2"
    assert restarted.state.match_phase is MatchPhase.PRE_MATCH
    assert restarted.state.fault_codes == ()


def test_observation_accepts_new_capture_session_once_and_rejects_old_frames() -> None:
    ready = _ready_state()
    generation = ready.input_generation
    switched = reduce_event(
        ready,
        _observation(
            4,
            capture_seq=1,
            input_digest="input-1",
            capture_session_id="capture-2",
        ),
    )
    assert switched.accepted is True
    assert switched.state.capture_session_id == "capture-2"
    assert switched.state.retired_capture_session_ids == ("capture-1",)
    assert switched.state.input_generation == generation + 1
    assert switched.state.practical_lane.availability is EvaluationAvailability.PENDING
    invalidation = [item for item in switched.intents if item.kind is ReducerIntentKind.INVALIDATE_JOBS]
    assert invalidation[0].payload["reason"] == "new_capture_session"

    stale = reduce_event(
        switched.state,
        _observation(5, capture_seq=2, input_digest="old", capture_session_id="capture-1"),
    )
    assert stale.accepted is False
    assert stale.state.capture_session_id == "capture-2"
    assert stale.state.input_generation == switched.state.input_generation
    assert stale.intents[0].payload["reason"] == "stale_capture_session"
    _assert_projected_valid(stale.state)


def test_worker_event_remains_explicitly_unimplemented() -> None:
    reduction = reduce_event(_state(), _event(ReducerEventKind.WORKER_FAULT, 1))
    assert reduction.accepted is False
    assert reduction.state.match_phase is MatchPhase.PRE_MATCH
    assert len(reduction.intents) == 1
    assert reduction.intents[0].kind is ReducerIntentKind.REJECT_EVENT
    assert reduction.intents[0].payload["reason"] == "not_implemented_in_j1"


def test_terminal_rejects_non_allowlisted_or_direction_mismatched_evidence() -> None:
    state = _ready_state()
    candidate = reduce_event(
        state, _terminal_evidence(state, 4, evidence_kind="death_candidate"),
    )
    assert candidate.accepted is False
    assert candidate.intents[0].payload["reason"] == "terminal_evidence_not_allowlisted"

    state = _ready_state()
    mismatch = reduce_event(
        state, _terminal_evidence(state, 4, winner="1P", result_code="p2_win"),
    )
    assert mismatch.accepted is False
    assert mismatch.intents[0].payload["reason"] == "terminal_direction_invalid"


def test_terminal_fact_invalidates_jobs_and_projects_exact_probability() -> None:
    state = _ready_state()
    reduction = reduce_event(state, _terminal_evidence(state, 4))

    assert reduction.accepted is True
    assert reduction.state.match_phase is MatchPhase.TERMINAL
    assert reduction.state.display_state is DisplayState.TERMINAL_FACT
    assert reduction.state.terminal_timer_deadline_monotonic_ms == 7_000
    assert reduction.state.practical_lane.values["p1_win_probability"] == 1.0
    assert reduction.state.practical_lane.values["advantage_score"] == 100.0
    assert [intent.kind for intent in reduction.intents] == [
        ReducerIntentKind.INVALIDATE_JOBS,
        ReducerIntentKind.SCHEDULE_TIMER,
        ReducerIntentKind.PUBLISH_SNAPSHOT,
    ]
    _assert_projected_valid(reduction.state)


def test_terminal_timer_transitions_to_result_and_boundary_clears_fact() -> None:
    terminal = reduce_event(
        _ready_state(), _terminal_evidence(_ready_state(), 4, winner="2P", result_code="p2_win"),
    ).state
    result = reduce_event(
        terminal,
        _event(
            ReducerEventKind.TIMER_FIRED, 5,
            timer_id=terminal.terminal_timer_id, fired_monotonic_ms=7_000,
        ),
    ).state
    assert result.match_phase is MatchPhase.RESULT
    assert result.display_state is DisplayState.RESULT
    assert result.practical_lane.availability is EvaluationAvailability.UNAVAILABLE
    _assert_projected_valid(result)

    boundary = reduce_event(
        result, _event(ReducerEventKind.FORMAL_BOUNDARY, 6, match_id="match-2"),
    ).state
    assert boundary.terminal_state.value == "none"
    assert boundary.terminal_winner is None
    assert boundary.display_state is DisplayState.HIDDEN
    _assert_projected_valid(boundary)


@pytest.mark.parametrize("state_name", ["empty_tokens", "already_completed"])
def test_prediction_requires_queued_job_and_non_null_tokens(state_name: str) -> None:
    state = _state() if state_name == "empty_tokens" else _ready_state()
    seq = 1 if state.last_event_seq is None else state.last_event_seq + 1
    reduction = reduce_event(state, _prediction(state, seq))
    assert reduction.accepted is False
    assert reduction.intents[0].kind is ReducerIntentKind.REJECT_EVENT
    assert reduction.intents[0].payload["reason"] == "stale_prediction"
    assert reduction.state.practical_lane == state.practical_lane
    _assert_projected_valid(reduction.state)


@pytest.mark.parametrize(
    ("case", "capture_seq", "digest_override", "reason", "fault"),
    [
        ("forward", 2, None, None, None),
        ("gap", 3, None, "capture_sequence_gap", None),
        ("reversed", 0, None, "capture_sequence_reversed", None),
        ("duplicate", 1, None, "duplicate_capture", None),
        ("conflict", 1, "conflicting-content", None, FaultCode.ID_CONTENT_CONFLICT),
    ],
)
def test_capture_sequence_cases_are_auditable_while_adapter_replacement_is_j4(
    case: str,
    capture_seq: int,
    digest_override: str | None,
    reason: str | None,
    fault: FaultCode | None,
) -> None:
    # adapter固有の置換・欠落counter連携はJ4へ延期し、J1ではcapture通番だけを正本にする。
    state = _ready_state()
    event = _observation(4, capture_seq=capture_seq, input_digest="input-1")
    if digest_override is not None:
        payload = dict(event.payload)
        payload["capture_digest"] = digest_override
        event = ReducerEvent(event.kind, event.event_seq, event.content_digest, payload)
    reduction = reduce_event(state, event)
    reject_reasons = [
        item.payload["reason"]
        for item in reduction.intents
        if item.kind is ReducerIntentKind.REJECT_EVENT
    ]
    assert (reason in reject_reasons) if reason is not None else not reject_reasons
    assert (fault in reduction.state.fault_codes) if fault is not None else not reduction.state.fault_codes
    if fault is None:
        assert reduction.accepted is (case == "forward")
    else:
        assert reduction.state.display_state is DisplayState.HIDDEN
    _assert_projected_valid(reduction.state)


def test_hold_reason_without_any_source_time_is_atomically_rejected() -> None:
    state = _state()
    event = _event(
        ReducerEventKind.HOLD_REASON_ADDED,
        1,
        reason="calculation_pending",
        monotonic_ms=100,
    )
    reduction = reduce_event(state, event)
    assert reduction.accepted is False
    assert reduction.state.display_state is state.display_state
    assert reduction.state.source_available_ms is None
    assert reduction.state.hold_reasons == ()
    assert reduction.intents[0].payload["reason"] == "hold_source_time_unavailable"


def test_hold_reason_preserves_zero_as_a_valid_source_time() -> None:
    state = _state()
    reduction = reduce_event(
        state,
        _event(
            ReducerEventKind.HOLD_REASON_ADDED,
            1,
            reason="calculation_pending",
            source_available_ms=0,
            monotonic_ms=100,
        ),
    )
    assert reduction.accepted is True
    assert reduction.state.hold_started_source_ms == 0
    assert reduction.state.source_available_ms == 0
    _assert_projected_valid(reduction.state)


@pytest.mark.parametrize("invalid_case", ["missing", "enum", "type"])
def test_domain_invalid_payload_fails_closed_with_valid_public_snapshot(
    invalid_case: str,
) -> None:
    state = _ready_state()
    if invalid_case == "missing":
        event = _event(ReducerEventKind.FORMAL_BOUNDARY, 4)
    elif invalid_case == "enum":
        event = _event(
            ReducerEventKind.HOLD_REASON_ADDED,
            4,
            reason="unknown-hold-reason",
            monotonic_ms=5_200,
        )
    else:
        event = _event(
            ReducerEventKind.TIMER_FIRED,
            4,
            timer_id="invalid",
            fired_monotonic_ms="not-an-integer",
        )
    reduction = reduce_event(state, event)
    assert reduction.state.match_phase is MatchPhase.INTEGRITY_FAULT
    assert FaultCode.SCHEMA_INVALID in reduction.state.fault_codes
    _assert_projected_valid(reduction.state)


def test_invalid_event_envelope_is_rejected_without_mutating_state() -> None:
    state = _ready_state()
    event = ReducerEvent(
        kind="invalid-kind",  # type: ignore[arg-type]
        event_seq=4,
        content_digest="invalid-envelope",
    )
    reduction = reduce_event(state, event)
    assert reduction.accepted is False
    assert reduction.state is state
    assert reduction.intents[0].payload["reason"] == "invalid_event_envelope"
    _assert_projected_valid(reduction.state)


def test_both_mode_allows_best_action_to_complete_before_practical() -> None:
    pending = _pending_state(EvaluationMode.BOTH)
    reduction = reduce_event(pending, _best_action_prediction(pending, 3))
    assert reduction.accepted is True
    assert reduction.state.display_state is DisplayState.LIVE
    assert reduction.state.practical_lane.availability is EvaluationAvailability.PENDING
    assert reduction.state.best_action_lane.availability is EvaluationAvailability.AVAILABLE
    assert reduction.state.practical_lane.input_generation == reduction.state.best_action_lane.input_generation
    assert reduction.state.practical_lane.input_digest == reduction.state.best_action_lane.input_digest
    _assert_projected_valid(reduction.state)


def test_resolving_last_hold_restores_legal_physical_prediction_display() -> None:
    physical = _physical_ready_state()
    assert physical.display_state is DisplayState.PHYSICAL_PREDICTION
    held = reduce_event(
        physical,
        _event(
            ReducerEventKind.HOLD_REASON_ADDED,
            4,
            reason="physics_ambiguous",
            monotonic_ms=5_200,
        ),
    ).state
    resolved = reduce_event(
        held,
        _event(ReducerEventKind.HOLD_REASON_RESOLVED, 5, reason="physics_ambiguous"),
    )
    assert resolved.state.display_state is DisplayState.PHYSICAL_PREDICTION
    assert resolved.state.hold_reasons == ()
    _assert_projected_valid(resolved.state)


def test_resolving_hold_with_insufficient_provenance_never_restores_live() -> None:
    physical = _physical_ready_state()
    held = reduce_event(
        physical,
        _observation(
            4,
            capture_seq=2,
            input_digest="untrusted",
            trusted=False,
            stable=False,
        ),
    ).state
    resolved = reduce_event(
        held,
        _event(ReducerEventKind.HOLD_REASON_RESOLVED, 5, reason="recognition_unreliable"),
    )
    assert resolved.state.display_state is DisplayState.AWAITING
    assert resolved.state.practical_lane.availability is EvaluationAvailability.UNAVAILABLE
    _assert_projected_valid(resolved.state)


@pytest.mark.parametrize("case", ["not_holding", "reason_not_registered"])
def test_resolving_inactive_hold_reason_is_explicitly_rejected(case: str) -> None:
    state = _ready_state()
    if case == "reason_not_registered":
        state = reduce_event(
            state,
            _event(
                ReducerEventKind.HOLD_REASON_ADDED,
                4,
                reason="recognition_unreliable",
                monotonic_ms=5_200,
            ),
        ).state
        seq = 5
    else:
        seq = 4
    reduction = reduce_event(
        state,
        _event(ReducerEventKind.HOLD_REASON_RESOLVED, seq, reason="calculation_pending"),
    )
    assert reduction.accepted is False
    assert reduction.state.display_state is state.display_state
    assert reduction.state.hold_reasons == state.hold_reasons
    assert reduction.intents[0].payload["reason"] == "hold_reason_not_active"
