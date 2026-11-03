"""Phase J schedulerのS01〜S05/S08合成契約試験。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from src.phase_j.contracts import WorkerHealth
from src.phase_j.scheduler import (
    NUMERIC_PUBLISH_INTERVAL_NS,
    SAFETY_EVENTS_BYPASS_NUMERIC_GATE,
    CommitRejectReason,
    CommitToken,
    EvaluationKind,
    JobRequest,
    JobResult,
    NumericGateState,
    SchedulerIntentKind,
    SchedulerState,
    complete_job,
    flush_numeric_candidate,
    initial_scheduler_state,
    invalidate_all,
    offer_numeric_candidate,
    set_worker_health,
    submit_job,
)


class _FakeClock:
    def __init__(self, now_ns: int = 0) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


def _token(
    kind: EvaluationKind = EvaluationKind.PRACTICAL,
    *,
    generation: int = 1,
    match_id: str = "match-1",
) -> CommitToken:
    return CommitToken(
        session_id="session-1",
        match_id=match_id,
        capture_session_id="capture-1",
        capture_seq=generation * 10,
        causal_cutoff_digest=f"cutoff-{generation}",
        input_generation=generation,
        evaluation_kind=kind,
        tier_profile_id="standard-v1",
        asset_bundle_id="assets-v1",
        deadline_monotonic_ns=2_000_000_000,
    )


def _request(
    kind: EvaluationKind = EvaluationKind.PRACTICAL,
    *,
    generation: int = 1,
    match_id: str = "match-1",
) -> JobRequest:
    token = _token(kind, generation=generation, match_id=match_id)
    return JobRequest(
        request_id=f"{kind.value}-{match_id}-{generation}",
        token=token,
        input_digest=f"input-{generation}",
        input_snapshot={"generation": generation, "cells": [1, 2]},
    )


def _result(request: JobRequest, *, value: int = 10) -> JobResult:
    return JobResult(
        evaluation_kind=request.token.evaluation_kind,
        request_id=request.request_id,
        token=request.token,
        input_digest=request.input_digest,
        values={"value": value},
    )


def _running(request: JobRequest) -> SchedulerState:
    return submit_job(initial_scheduler_state(), request).state


def test_scheduler_contracts_are_immutable_and_slotted() -> None:
    request = _request()
    result = _result(request)
    with pytest.raises(FrozenInstanceError):
        request.token.input_generation = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        request.input_snapshot["generation"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        result.values["value"] = 20  # type: ignore[index]
    assert not hasattr(initial_scheduler_state(), "__dict__")


@pytest.mark.parametrize(
    "field_name",
    [
        "session_id", "match_id", "capture_session_id", "causal_cutoff_digest",
        "tier_profile_id", "asset_bundle_id",
    ],
)
def test_commit_token_rejects_empty_ids_and_hashes(field_name: str) -> None:
    with pytest.raises(ValueError):
        replace(_token(), **{field_name: " "})


@pytest.mark.parametrize("field_name", ["capture_seq", "input_generation", "deadline_monotonic_ns"])
@pytest.mark.parametrize("invalid", [True, -1])
def test_commit_token_rejects_bool_or_negative_sequences(
    field_name: str,
    invalid: bool | int,
) -> None:
    with pytest.raises(ValueError):
        replace(_token(), **{field_name: invalid})


def test_invalid_evaluation_kind_cannot_fall_through_to_best_action_lane() -> None:
    with pytest.raises(ValueError):
        replace(_token(), evaluation_kind="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        replace(_request(), token="invalid-token")  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["request_id", "input_digest"])
def test_job_request_rejects_empty_identity(field_name: str) -> None:
    with pytest.raises(ValueError):
        replace(_request(), **{field_name: ""})


@pytest.mark.parametrize("kind", [EvaluationKind.PRACTICAL, EvaluationKind.BEST_ACTION])
def test_s01_each_lane_holds_one_running_and_latest_one_waiting(
    kind: EvaluationKind,
) -> None:
    first, second, latest = (_request(kind, generation=index) for index in (1, 2, 3))
    state = submit_job(initial_scheduler_state(), first).state
    state = submit_job(state, second).state
    decision = submit_job(state, latest)
    lane = decision.state.practical if kind is EvaluationKind.PRACTICAL else decision.state.best_action
    assert lane.running == first
    assert lane.waiting == latest
    assert decision.replaced_request_id == second.request_id


def test_s02_waiting_replacement_requests_running_cancel_without_trusting_cancel() -> None:
    first, second, latest = (_request(generation=index) for index in (1, 2, 3))
    state = submit_job(initial_scheduler_state(), first).state
    state = submit_job(state, second).state
    decision = submit_job(state, latest)
    assert decision.state.practical.running == first
    assert decision.state.practical.waiting == latest
    assert decision.intents[0].kind is SchedulerIntentKind.CANCEL_JOB
    assert decision.intents[0].request_id == first.request_id
    assert decision.replaced_request_id == second.request_id


TOKEN_MISMATCH_CASES = (
    ("session_id", "session-2", CommitRejectReason.SESSION_ID_MISMATCH),
    ("match_id", "match-2", CommitRejectReason.MATCH_ID_MISMATCH),
    ("capture_session_id", "capture-2", CommitRejectReason.CAPTURE_SESSION_ID_MISMATCH),
    ("capture_seq", 99, CommitRejectReason.CAPTURE_SEQ_MISMATCH),
    ("causal_cutoff_digest", "cutoff-other", CommitRejectReason.CAUSAL_CUTOFF_MISMATCH),
    ("input_generation", 2, CommitRejectReason.INPUT_GENERATION_MISMATCH),
    ("evaluation_kind", EvaluationKind.BEST_ACTION, CommitRejectReason.EVALUATION_KIND_MISMATCH),
    ("tier_profile_id", "high-v1", CommitRejectReason.TIER_PROFILE_MISMATCH),
    ("asset_bundle_id", "assets-v2", CommitRejectReason.ASSET_BUNDLE_MISMATCH),
    ("deadline_monotonic_ns", 3_000_000_000, CommitRejectReason.DEADLINE_MISMATCH),
)


@pytest.mark.parametrize(("field_name", "changed", "reason"), TOKEN_MISMATCH_CASES)
def test_s03_every_commit_token_field_mismatch_is_typed_and_rejected(
    field_name: str,
    changed: Any,
    reason: CommitRejectReason,
) -> None:
    request = _request()
    current = replace(request.token, **{field_name: changed})
    decision = complete_job(_running(request), _result(request), current, 1_000_000_000)
    assert decision.reducer_candidate is None
    assert decision.reject_reason is reason


@pytest.mark.parametrize(
    ("field_name", "reason"),
    [
        ("request_id", CommitRejectReason.REQUEST_ID_MISMATCH),
        ("input_digest", CommitRejectReason.INPUT_DIGEST_MISMATCH),
    ],
)
def test_s03_request_identity_mismatch_is_rejected(
    field_name: str,
    reason: CommitRejectReason,
) -> None:
    request = _request()
    result = replace(_result(request), **{field_name: "other"})
    decision = complete_job(_running(request), result, request.token, 1_000_000_000)
    assert decision.reject_reason is reason
    assert decision.reducer_candidate is None


def test_s03_later_semantically_unchanged_capture_head_does_not_invalidate_token() -> None:
    request = _request()
    state = _running(request)
    # backend headが進んでも、current_tokenはgeneration発行時snapshotのまま照合する。
    decision = complete_job(state, _result(request), request.token, 1_500_000_000)
    assert decision.reject_reason is None
    assert decision.reducer_candidate == _result(request)
    assert not hasattr(decision, "hub")


def test_s03_missing_token_is_rejected_without_none_equality() -> None:
    request = _request()
    result = replace(_result(request), token=None)
    decision = complete_job(_running(request), result, request.token, 1_000_000_000)
    assert decision.reject_reason is CommitRejectReason.MISSING_TOKEN
    assert decision.reducer_candidate is None


def test_s03_deadline_exceeded_and_restarting_worker_are_rejected() -> None:
    request = _request()
    expired = complete_job(_running(request), _result(request), request.token, 2_000_000_001)
    assert expired.reject_reason is CommitRejectReason.DEADLINE_EXCEEDED

    running = _running(request)
    restarting = set_worker_health(
        running, EvaluationKind.PRACTICAL, WorkerHealth.RESTARTING
    )
    rejected = complete_job(restarting.state, _result(request), request.token, 1_000_000_000)
    assert restarting.intents[0].kind is SchedulerIntentKind.CANCEL_JOB
    assert rejected.reject_reason is CommitRejectReason.WORKER_RESTARTING


@pytest.mark.parametrize(
    "health",
    [WorkerHealth.STARTING, WorkerHealth.DEGRADED, WorkerHealth.DISABLED],
)
def test_non_healthy_worker_result_is_rejected_and_latest_waiting_recovers(
    health: WorkerHealth,
) -> None:
    running, waiting = _request(generation=1), _request(generation=2)
    state = submit_job(initial_scheduler_state(), running).state
    state = submit_job(state, waiting).state
    unhealthy = set_worker_health(state, EvaluationKind.PRACTICAL, health).state

    rejected = complete_job(
        unhealthy, _result(running), running.token, 1_000_000_000,
    )
    assert rejected.reject_reason is CommitRejectReason.WORKER_UNHEALTHY
    assert rejected.reducer_candidate is None
    assert rejected.state.practical.running is None
    assert rejected.state.practical.waiting == waiting
    assert rejected.intents == ()

    recovered = set_worker_health(
        rejected.state, EvaluationKind.PRACTICAL, WorkerHealth.HEALTHY,
    )
    assert recovered.state.practical.running == waiting
    assert recovered.state.practical.waiting is None
    assert recovered.intents[0].kind is SchedulerIntentKind.START_JOB


def test_restart_to_healthy_promotes_latest_waiting_with_start_intent() -> None:
    running, waiting = _request(generation=1), _request(generation=2)
    state = submit_job(initial_scheduler_state(), running).state
    state = submit_job(state, waiting).state
    restarting = set_worker_health(
        state, EvaluationKind.PRACTICAL, WorkerHealth.RESTARTING
    ).state
    rejected = complete_job(
        restarting, _result(running), running.token, 1_000_000_000
    )
    assert rejected.reject_reason is CommitRejectReason.WORKER_RESTARTING
    assert rejected.state.practical.running is None
    healthy = set_worker_health(
        rejected.state, EvaluationKind.PRACTICAL, WorkerHealth.HEALTHY
    )
    assert healthy.state.practical.running == waiting
    assert healthy.state.practical.waiting is None
    assert healthy.intents[0].kind is SchedulerIntentKind.START_JOB


def test_restart_to_healthy_promotes_waiting_without_result_callback() -> None:
    running, waiting = _request(generation=1), _request(generation=2)
    state = submit_job(initial_scheduler_state(), running).state
    state = submit_job(state, waiting).state
    restarting = set_worker_health(
        state, EvaluationKind.PRACTICAL, WorkerHealth.RESTARTING
    )
    assert restarting.state.practical.running is None
    assert restarting.state.practical.waiting == waiting
    assert restarting.intents[0].kind is SchedulerIntentKind.CANCEL_JOB

    healthy = set_worker_health(
        restarting.state, EvaluationKind.PRACTICAL, WorkerHealth.HEALTHY
    )
    assert healthy.state.practical.running == waiting
    assert healthy.state.practical.waiting is None
    assert healthy.intents[0].kind is SchedulerIntentKind.START_JOB


def test_late_old_result_after_restart_without_waiting_is_invalidated() -> None:
    request = _request()
    state = _running(request)
    restarting = set_worker_health(
        state, EvaluationKind.PRACTICAL, WorkerHealth.RESTARTING
    ).state
    healthy = set_worker_health(
        restarting, EvaluationKind.PRACTICAL, WorkerHealth.HEALTHY
    ).state
    delayed = complete_job(healthy, _result(request), request.token, 1_000_000_000)
    assert delayed.reject_reason is CommitRejectReason.INVALIDATED_JOB
    assert delayed.reducer_candidate is None


def test_input_digest_mismatch_finishes_running_slot_and_promotes_latest() -> None:
    running, waiting = _request(generation=1), _request(generation=2)
    state = submit_job(initial_scheduler_state(), running).state
    state = submit_job(state, waiting).state
    corrupt = replace(_result(running), input_digest="corrupt")
    decision = complete_job(state, corrupt, running.token, 1_000_000_000)
    assert decision.reject_reason is CommitRejectReason.INPUT_DIGEST_MISMATCH
    assert decision.reducer_candidate is None
    assert decision.state.practical.running == waiting
    assert decision.state.practical.waiting is None
    assert decision.intents[0].kind is SchedulerIntentKind.START_JOB


def test_s03_invalidated_job_is_rejected() -> None:
    request = _request()
    invalidated = invalidate_all(_running(request), "capture_disconnected")
    decision = complete_job(
        invalidated.state, _result(request), request.token, 1_000_000_000
    )
    assert decision.reject_reason is CommitRejectReason.INVALIDATED_JOB
    assert decision.reducer_candidate is None


def test_s04_old_boundary_result_never_becomes_reducer_candidate() -> None:
    old = _request(match_id="match-1")
    cleared = invalidate_all(_running(old), "formal_boundary").state
    current = _request(generation=2, match_id="match-2")
    state = submit_job(cleared, current).state
    decision = complete_job(state, _result(old), current.token, 1_000_000_000)
    assert decision.reject_reason is CommitRejectReason.MATCH_ID_MISMATCH
    assert decision.reducer_candidate is None
    assert decision.state.practical.running == current


def test_s05_best_action_saturation_does_not_block_practical_commit() -> None:
    best_running = _request(EvaluationKind.BEST_ACTION, generation=1)
    best_waiting = _request(EvaluationKind.BEST_ACTION, generation=2)
    practical = _request(EvaluationKind.PRACTICAL, generation=3)
    state = submit_job(initial_scheduler_state(), best_running).state
    state = submit_job(state, best_waiting).state
    practical_decision = submit_job(state, practical)
    assert practical_decision.state.practical.running == practical
    assert practical_decision.state.best_action.running == best_running
    assert practical_decision.state.best_action.waiting == best_waiting
    completed = complete_job(
        practical_decision.state,
        _result(practical),
        practical.token,
        1_000_000_000,
    )
    assert completed.reducer_candidate == _result(practical)
    assert completed.state.best_action == practical_decision.state.best_action


def test_s08_numeric_candidates_are_latest_wins_and_limited_to_two_hz() -> None:
    clock = _FakeClock()
    first, second, latest = (_result(_request(generation=index)) for index in (1, 2, 3))
    emitted = offer_numeric_candidate(NumericGateState(), first, clock)
    assert emitted.reducer_candidate == first
    clock.now_ns = 100_000_000
    held = offer_numeric_candidate(emitted.state, second, clock)
    clock.now_ns = 200_000_000
    held = offer_numeric_candidate(held.state, latest, clock)
    clock.now_ns = NUMERIC_PUBLISH_INTERVAL_NS - 1
    assert flush_numeric_candidate(held.state, clock).reducer_candidate is None
    clock.now_ns = NUMERIC_PUBLISH_INTERVAL_NS
    flushed = flush_numeric_candidate(held.state, clock)
    assert flushed.reducer_candidate == latest
    assert flushed.state.pending_candidate is None


def test_s08_best_action_bypasses_gate_without_changing_pending_practical_ab() -> None:
    first = _result(_request(generation=1))
    older = _result(_request(generation=2))
    latest = _result(_request(generation=3))
    best = _result(_request(EvaluationKind.BEST_ACTION, generation=99))

    control_clock = _FakeClock()
    control = offer_numeric_candidate(NumericGateState(), first, control_clock)
    control_clock.now_ns = 100_000_000
    control = offer_numeric_candidate(control.state, latest, control_clock)
    control_clock.now_ns = NUMERIC_PUBLISH_INTERVAL_NS
    control = offer_numeric_candidate(control.state, older, control_clock)

    mixed_clock = _FakeClock()
    mixed = offer_numeric_candidate(NumericGateState(), first, mixed_clock)
    mixed_clock.now_ns = 100_000_000
    mixed = offer_numeric_candidate(mixed.state, latest, mixed_clock)
    pending_before = mixed.state
    mixed_clock.now_ns = 200_000_000
    bypassed = offer_numeric_candidate(mixed.state, best, mixed_clock)
    assert bypassed.reducer_candidate == best
    assert bypassed.state == pending_before
    mixed_clock.now_ns = NUMERIC_PUBLISH_INTERVAL_NS
    mixed = offer_numeric_candidate(bypassed.state, older, mixed_clock)
    assert mixed.reducer_candidate == latest
    assert mixed == control


def test_s08_safety_events_explicitly_bypass_scheduler_numeric_gate() -> None:
    assert {
        "formal_boundary",
        "hold_started",
        "hold_expired",
        "integrity_fault",
        "terminal_confirmed",
        "worker_fault",
        "capture_status",
        "health",
    } <= SAFETY_EVENTS_BYPASS_NUMERIC_GATE


def test_scheduler_replay_is_deterministic() -> None:
    first, latest = _request(generation=1), _request(generation=2)
    left = submit_job(initial_scheduler_state(), first)
    right = submit_job(initial_scheduler_state(), first)
    left = submit_job(left.state, latest)
    right = submit_job(right.state, latest)
    assert left == right
