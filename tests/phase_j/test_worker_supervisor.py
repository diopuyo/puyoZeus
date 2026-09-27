"""Phase J worker supervisorのT06/deadline/restart純粋状態試験。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.phase_j.contracts import WorkerHealth
from src.phase_j.scheduler import EvaluationKind
from src.phase_j.worker_supervisor import (
    TIER_CHANGE_SUGGESTION_TIMEOUTS,
    WorkerSupervisorEvent,
    WorkerSupervisorEventKind,
    WorkerSupervisorIntentKind,
    WorkerSupervisorRejectReason,
    WorkerSupervisorSignalKind,
    WorkerSupervisorState,
    bootstrap_worker_supervisor,
    reduce_worker_event,
)

TOKEN_A = "worker-token-a"
TOKEN_B = "worker-token-b"
TOKEN_C = "worker-token-c"
PROFILE = "standard-v1"
HARD_DEADLINE_NS = 500_000_000
GRACE_DEADLINE_NS = 550_000_000


def _bootstrap(
    lane: EvaluationKind = EvaluationKind.PRACTICAL,
) -> WorkerSupervisorState:
    return bootstrap_worker_supervisor(lane, PROFILE, TOKEN_A).state


def _event(
    seq: int,
    kind: WorkerSupervisorEventKind,
    *,
    lane: EvaluationKind = EvaluationKind.PRACTICAL,
    token: str | None = TOKEN_A,
    **kwargs: object,
) -> WorkerSupervisorEvent:
    return WorkerSupervisorEvent(seq, lane, kind, worker_token=token, **kwargs)


def _healthy(
    lane: EvaluationKind = EvaluationKind.PRACTICAL,
) -> WorkerSupervisorState:
    state = _bootstrap(lane)
    spawned = _event(1, WorkerSupervisorEventKind.WORKER_SPAWNED, lane=lane)
    state = reduce_worker_event(state, spawned).state
    prewarm = _event(
        2, WorkerSupervisorEventKind.PREWARM_COMPLETED,
        lane=lane, prewarm_succeeded=True,
    )
    return reduce_worker_event(state, prewarm).state


def _running(
    lane: EvaluationKind = EvaluationKind.PRACTICAL,
    *,
    start_seq: int = 3,
    token: str = TOKEN_A,
) -> WorkerSupervisorState:
    state = _healthy(lane)
    started = _event(
        start_seq, WorkerSupervisorEventKind.JOB_STARTED,
        lane=lane, token=token, request_id="request-1",
        hard_deadline_monotonic_ns=HARD_DEADLINE_NS,
        termination_grace_deadline_monotonic_ns=GRACE_DEADLINE_NS,
    )
    return reduce_worker_event(state, started).state


def _restart_after_timeout(
    state: WorkerSupervisorState,
    *,
    timeout_seq: int,
    restart_seq: int,
    replacement: str,
) -> WorkerSupervisorState:
    timed_out = reduce_worker_event(
        state,
        _event(
            timeout_seq, WorkerSupervisorEventKind.TIMER_OBSERVED,
            token=None, observed_monotonic_ns=HARD_DEADLINE_NS,
        ),
    )
    restarted = reduce_worker_event(
        timed_out.state,
        _event(
            restart_seq, WorkerSupervisorEventKind.TIMER_OBSERVED,
            token=None, observed_monotonic_ns=GRACE_DEADLINE_NS,
            replacement_worker_token=replacement,
        ),
    )
    return restarted.state


def test_t06_bootstrap_emits_spawn_but_disabled_lane_never_spawns() -> None:
    enabled = bootstrap_worker_supervisor(EvaluationKind.PRACTICAL, PROFILE, TOKEN_A)
    disabled = bootstrap_worker_supervisor(EvaluationKind.BEST_ACTION, PROFILE, None, enabled=False)
    assert enabled.state.health is WorkerHealth.STARTING
    assert [intent.kind for intent in enabled.intents] == [WorkerSupervisorIntentKind.SPAWN_WORKER]
    assert disabled.state.health is WorkerHealth.DISABLED
    assert disabled.intents == ()
    rejected = reduce_worker_event(
        disabled.state,
        _event(1, WorkerSupervisorEventKind.TIMER_OBSERVED, lane=EvaluationKind.BEST_ACTION,
               token=None, observed_monotonic_ns=HARD_DEADLINE_NS),
    )
    assert rejected.reject_reason is WorkerSupervisorRejectReason.LANE_DISABLED
    assert not any(intent.kind is WorkerSupervisorIntentKind.SPAWN_WORKER for intent in rejected.intents)


def test_t06_only_same_token_successful_prewarm_makes_worker_healthy() -> None:
    state = _bootstrap()
    spawned = reduce_worker_event(state, _event(1, WorkerSupervisorEventKind.WORKER_SPAWNED))
    assert spawned.state.health is WorkerHealth.STARTING
    assert spawned.intents[0].kind is WorkerSupervisorIntentKind.PREWARM_WORKER
    stale = reduce_worker_event(
        spawned.state,
        _event(2, WorkerSupervisorEventKind.PREWARM_COMPLETED,
               token="old-token", prewarm_succeeded=True),
    )
    assert stale.reject_reason is WorkerSupervisorRejectReason.STALE_WORKER_TOKEN
    completed = reduce_worker_event(
        stale.state,
        _event(3, WorkerSupervisorEventKind.PREWARM_COMPLETED, prewarm_succeeded=True),
    )
    assert completed.state.health is WorkerHealth.HEALTHY
    assert completed.signals[0].kind is WorkerSupervisorSignalKind.RESUME_LATEST_WAITING


@pytest.mark.parametrize("health", [WorkerHealth.STARTING, WorkerHealth.RESTARTING])
def test_t06_starting_and_restarting_reject_jobs_and_results(health: WorkerHealth) -> None:
    state = _bootstrap()
    state = WorkerSupervisorState(
        state.lane, health, True, state.tier_profile_id, state.worker_token,
    )
    job = reduce_worker_event(
        state,
        _event(
            1, WorkerSupervisorEventKind.JOB_STARTED, request_id="request-1",
            hard_deadline_monotonic_ns=HARD_DEADLINE_NS,
            termination_grace_deadline_monotonic_ns=GRACE_DEADLINE_NS,
        ),
    )
    result = reduce_worker_event(
        job.state,
        _event(2, WorkerSupervisorEventKind.RESULT_RECEIVED, request_id="request-1"),
    )
    assert job.reject_reason is WorkerSupervisorRejectReason.WORKER_NOT_READY
    assert result.reject_reason is WorkerSupervisorRejectReason.WORKER_NOT_READY
    assert result.signals == ()


def test_healthy_worker_accepts_job_and_trusts_matching_result() -> None:
    state = _running()
    result = reduce_worker_event(
        state,
        _event(4, WorkerSupervisorEventKind.RESULT_RECEIVED, request_id="request-1"),
    )
    assert result.accepted
    assert result.state.active_request_id is None
    assert result.state.consecutive_timeouts == 0
    assert result.signals[0].kind is WorkerSupervisorSignalKind.RESULT_TRUSTED


def test_hard_deadline_cancels_then_grace_terminates_and_spawns_new_token() -> None:
    state = _running()
    before = reduce_worker_event(
        state,
        _event(4, WorkerSupervisorEventKind.TIMER_OBSERVED,
               token=None, observed_monotonic_ns=HARD_DEADLINE_NS - 1),
    )
    assert before.intents == () and before.state.health is WorkerHealth.HEALTHY
    timeout = reduce_worker_event(
        before.state,
        _event(5, WorkerSupervisorEventKind.TIMER_OBSERVED,
               token=None, observed_monotonic_ns=HARD_DEADLINE_NS),
    )
    assert timeout.state.health is WorkerHealth.DEGRADED
    assert [intent.kind for intent in timeout.intents] == [WorkerSupervisorIntentKind.CANCEL_JOB]
    assert timeout.signals[0].kind is WorkerSupervisorSignalKind.WORKER_FAULT
    restart = reduce_worker_event(
        timeout.state,
        _event(6, WorkerSupervisorEventKind.TIMER_OBSERVED, token=None,
               observed_monotonic_ns=GRACE_DEADLINE_NS, replacement_worker_token=TOKEN_B),
    )
    assert restart.state.health is WorkerHealth.RESTARTING
    assert restart.state.worker_token == TOKEN_B
    assert [intent.kind for intent in restart.intents] == [
        WorkerSupervisorIntentKind.TERMINATE_WORKER,
        WorkerSupervisorIntentKind.SPAWN_WORKER,
    ]
    assert [intent.worker_token for intent in restart.intents] == [TOKEN_A, TOKEN_B]


def test_t06_callbackless_restart_recovers_only_after_new_worker_prewarm() -> None:
    restarting = _restart_after_timeout(
        _running(), timeout_seq=4, restart_seq=5, replacement=TOKEN_B,
    )
    spawned = reduce_worker_event(
        restarting,
        _event(6, WorkerSupervisorEventKind.WORKER_SPAWNED, token=TOKEN_B),
    )
    assert spawned.state.health is WorkerHealth.RESTARTING
    assert spawned.intents[0].kind is WorkerSupervisorIntentKind.PREWARM_WORKER
    prewarmed = reduce_worker_event(
        spawned.state,
        _event(7, WorkerSupervisorEventKind.PREWARM_COMPLETED,
               token=TOKEN_B, prewarm_succeeded=True),
    )
    assert prewarmed.state.health is WorkerHealth.HEALTHY
    assert prewarmed.signals[0].kind is WorkerSupervisorSignalKind.RESUME_LATEST_WAITING


@pytest.mark.parametrize(
    ("kind", "kwargs"),
    [
        (WorkerSupervisorEventKind.WORKER_SPAWNED, {}),
        (WorkerSupervisorEventKind.PREWARM_COMPLETED, {"prewarm_succeeded": True}),
        (WorkerSupervisorEventKind.RESULT_RECEIVED, {"request_id": "request-1"}),
    ],
)
def test_old_worker_token_callbacks_and_results_are_rejected(
    kind: WorkerSupervisorEventKind,
    kwargs: dict[str, object],
) -> None:
    restarting = _restart_after_timeout(
        _running(), timeout_seq=4, restart_seq=5, replacement=TOKEN_B,
    )
    rejected = reduce_worker_event(
        restarting,
        _event(6, kind, token=TOKEN_A, **kwargs),
    )
    assert rejected.reject_reason is WorkerSupervisorRejectReason.STALE_WORKER_TOKEN
    assert rejected.state.health is WorkerHealth.RESTARTING


def test_worker_fault_requests_hold_and_recovers_without_exit_callback() -> None:
    state = _running()
    fault = reduce_worker_event(
        state,
        _event(4, WorkerSupervisorEventKind.WORKER_FAULT,
               termination_grace_deadline_monotonic_ns=GRACE_DEADLINE_NS),
    )
    assert fault.state.health is WorkerHealth.DEGRADED
    assert fault.intents[0].kind is WorkerSupervisorIntentKind.CANCEL_JOB
    assert fault.signals[0].kind is WorkerSupervisorSignalKind.WORKER_FAULT
    restarted = reduce_worker_event(
        fault.state,
        _event(5, WorkerSupervisorEventKind.TIMER_OBSERVED, token=None,
               observed_monotonic_ns=GRACE_DEADLINE_NS, replacement_worker_token=TOKEN_B),
    )
    assert restarted.state.health is WorkerHealth.RESTARTING


def test_failed_prewarm_stays_unready_until_replacement_prewarm_succeeds() -> None:
    state = _bootstrap()
    spawned = reduce_worker_event(state, _event(1, WorkerSupervisorEventKind.WORKER_SPAWNED))
    failed = reduce_worker_event(
        spawned.state,
        _event(2, WorkerSupervisorEventKind.PREWARM_COMPLETED,
               prewarm_succeeded=False,
               termination_grace_deadline_monotonic_ns=GRACE_DEADLINE_NS),
    )
    assert failed.state.health is WorkerHealth.DEGRADED
    restarted = reduce_worker_event(
        failed.state,
        _event(3, WorkerSupervisorEventKind.TIMER_OBSERVED, token=None,
               observed_monotonic_ns=GRACE_DEADLINE_NS, replacement_worker_token=TOKEN_B),
    )
    spawned_b = reduce_worker_event(
        restarted.state,
        _event(4, WorkerSupervisorEventKind.WORKER_SPAWNED, token=TOKEN_B),
    )
    ready = reduce_worker_event(
        spawned_b.state,
        _event(5, WorkerSupervisorEventKind.PREWARM_COMPLETED,
               token=TOKEN_B, prewarm_succeeded=True),
    )
    assert ready.state.health is WorkerHealth.HEALTHY


def test_consecutive_timeouts_never_change_tier_profile_automatically() -> None:
    first_restart = _restart_after_timeout(
        _running(), timeout_seq=4, restart_seq=5, replacement=TOKEN_B,
    )
    spawned = reduce_worker_event(
        first_restart,
        _event(6, WorkerSupervisorEventKind.WORKER_SPAWNED, token=TOKEN_B),
    )
    ready = reduce_worker_event(
        spawned.state,
        _event(7, WorkerSupervisorEventKind.PREWARM_COMPLETED,
               token=TOKEN_B, prewarm_succeeded=True),
    )
    second_job = reduce_worker_event(
        ready.state,
        _event(8, WorkerSupervisorEventKind.JOB_STARTED, token=TOKEN_B,
               request_id="request-2", hard_deadline_monotonic_ns=HARD_DEADLINE_NS,
               termination_grace_deadline_monotonic_ns=GRACE_DEADLINE_NS),
    )
    second_timeout = reduce_worker_event(
        second_job.state,
        _event(9, WorkerSupervisorEventKind.TIMER_OBSERVED,
               token=None, observed_monotonic_ns=HARD_DEADLINE_NS),
    )
    assert second_timeout.state.consecutive_timeouts == TIER_CHANGE_SUGGESTION_TIMEOUTS
    assert second_timeout.state.tier_profile_id == PROFILE
    assert second_timeout.intents[-1].kind is WorkerSupervisorIntentKind.SUGGEST_TIER_CHANGE
    assert second_timeout.intents[-1].tier_profile_id == PROFILE


def test_duplicate_and_reversed_events_are_explicitly_rejected() -> None:
    state = _bootstrap()
    accepted = reduce_worker_event(state, _event(10, WorkerSupervisorEventKind.WORKER_SPAWNED))
    duplicate = reduce_worker_event(
        accepted.state, _event(10, WorkerSupervisorEventKind.WORKER_SPAWNED),
    )
    reversed_event = reduce_worker_event(
        accepted.state, _event(9, WorkerSupervisorEventKind.WORKER_SPAWNED),
    )
    assert duplicate.reject_reason is WorkerSupervisorRejectReason.DUPLICATE_EVENT
    assert reversed_event.reject_reason is WorkerSupervisorRejectReason.REVERSED_EVENT
    assert duplicate.state == accepted.state == reversed_event.state


def test_practical_and_best_action_supervisors_are_independent() -> None:
    practical = _healthy(EvaluationKind.PRACTICAL)
    best_action = _bootstrap(EvaluationKind.BEST_ACTION)
    practical_fault = reduce_worker_event(
        practical,
        _event(3, WorkerSupervisorEventKind.WORKER_FAULT,
               termination_grace_deadline_monotonic_ns=GRACE_DEADLINE_NS),
    )
    assert practical_fault.state.health is WorkerHealth.DEGRADED
    assert best_action.health is WorkerHealth.STARTING
    wrong_lane = reduce_worker_event(
        best_action,
        _event(1, WorkerSupervisorEventKind.WORKER_SPAWNED,
               lane=EvaluationKind.PRACTICAL),
    )
    assert wrong_lane.reject_reason is WorkerSupervisorRejectReason.LANE_MISMATCH
    assert wrong_lane.state == best_action


def test_same_event_sequence_is_deterministic_and_state_is_frozen() -> None:
    events = (
        _event(1, WorkerSupervisorEventKind.WORKER_SPAWNED),
        _event(2, WorkerSupervisorEventKind.PREWARM_COMPLETED, prewarm_succeeded=True),
        _event(3, WorkerSupervisorEventKind.JOB_STARTED, request_id="request-1",
               hard_deadline_monotonic_ns=HARD_DEADLINE_NS,
               termination_grace_deadline_monotonic_ns=GRACE_DEADLINE_NS),
        _event(4, WorkerSupervisorEventKind.TIMER_OBSERVED,
               token=None, observed_monotonic_ns=HARD_DEADLINE_NS),
    )

    def replay() -> WorkerSupervisorState:
        state = _bootstrap()
        for event in events:
            state = reduce_worker_event(state, event).state
        return state

    assert replay() == replay()
    with pytest.raises(FrozenInstanceError):
        replay().health = WorkerHealth.HEALTHY  # type: ignore[misc]
