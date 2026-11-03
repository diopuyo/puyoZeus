"""Phase J worker再生成を副作用なしで決めるlane別状態機械。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from .contracts import WorkerHealth
from .scheduler import EvaluationKind

TIER_CHANGE_SUGGESTION_TIMEOUTS = 2


class WorkerSupervisorEventKind(StrEnum):
    """外部runtimeから注入されるworker事象。"""

    WORKER_SPAWNED = "worker_spawned"
    PREWARM_COMPLETED = "prewarm_completed"
    JOB_STARTED = "job_started"
    RESULT_RECEIVED = "result_received"
    WORKER_FAULT = "worker_fault"
    TIMER_OBSERVED = "timer_observed"


class WorkerSupervisorIntentKind(StrEnum):
    """runtimeが実行する副作用。"""

    SPAWN_WORKER = "spawn_worker"
    PREWARM_WORKER = "prewarm_worker"
    CANCEL_JOB = "cancel_job"
    TERMINATE_WORKER = "terminate_worker"
    SUGGEST_TIER_CHANGE = "suggest_tier_change"


class WorkerSupervisorSignalKind(StrEnum):
    """scheduler/reducerへ渡す型付き通知。"""

    WORKER_FAULT = "worker_fault"
    RESUME_LATEST_WAITING = "resume_latest_waiting"
    RESULT_TRUSTED = "result_trusted"


class WorkerSupervisorRejectReason(StrEnum):
    """事象を状態へ採用しなかった理由。"""

    DUPLICATE_EVENT = "duplicate_event"
    REVERSED_EVENT = "reversed_event"
    LANE_MISMATCH = "lane_mismatch"
    LANE_DISABLED = "lane_disabled"
    STALE_WORKER_TOKEN = "stale_worker_token"
    WORKER_NOT_READY = "worker_not_ready"
    JOB_ALREADY_RUNNING = "job_already_running"
    REQUEST_MISMATCH = "request_mismatch"
    INVALID_EVENT = "invalid_event"
    MISSING_REPLACEMENT_TOKEN = "missing_replacement_token"
    UNEXPECTED_EVENT = "unexpected_event"


@dataclass(frozen=True, slots=True)
class WorkerSupervisorEvent:
    """kind別payloadを持つ、lane内単調通番の入力event。"""

    event_seq: int
    lane: EvaluationKind
    kind: WorkerSupervisorEventKind
    worker_token: str | None = None
    request_id: str | None = None
    hard_deadline_monotonic_ns: int | None = None
    termination_grace_deadline_monotonic_ns: int | None = None
    observed_monotonic_ns: int | None = None
    prewarm_succeeded: bool | None = None
    replacement_worker_token: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerSupervisorIntent:
    """外部adapterが解釈するだけの副作用指示。"""

    kind: WorkerSupervisorIntentKind
    lane: EvaluationKind
    worker_token: str
    request_id: str | None = None
    tier_profile_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerSupervisorSignal:
    """scheduler/reducerへ渡す純粋な連携signal。"""

    kind: WorkerSupervisorSignalKind
    lane: EvaluationKind
    worker_token: str
    request_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerSupervisorState:
    """一つの評価laneと一つの現行workerだけを管理する。"""

    lane: EvaluationKind
    health: WorkerHealth
    enabled: bool
    tier_profile_id: str
    worker_token: str | None
    last_event_seq: int | None = None
    active_request_id: str | None = None
    hard_deadline_monotonic_ns: int | None = None
    termination_grace_deadline_monotonic_ns: int | None = None
    cancel_requested: bool = False
    consecutive_timeouts: int = 0

    @property
    def accepts_jobs(self) -> bool:
        return self.enabled and self.health is WorkerHealth.HEALTHY

    @property
    def trusts_results(self) -> bool:
        return self.accepts_jobs


@dataclass(frozen=True, slots=True)
class WorkerSupervisorReduction:
    """遷移後状態と、外へ渡すintent/signal/rejectをまとめる。"""

    state: WorkerSupervisorState
    intents: tuple[WorkerSupervisorIntent, ...] = ()
    signals: tuple[WorkerSupervisorSignal, ...] = ()
    reject_reason: WorkerSupervisorRejectReason | None = None

    @property
    def accepted(self) -> bool:
        return self.reject_reason is None


def _valid_token(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_time(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _intent(
    state: WorkerSupervisorState,
    kind: WorkerSupervisorIntentKind,
    worker_token: str,
    **kwargs: str | None,
) -> WorkerSupervisorIntent:
    return WorkerSupervisorIntent(kind, state.lane, worker_token, **kwargs)


def _signal(
    state: WorkerSupervisorState,
    kind: WorkerSupervisorSignalKind,
    worker_token: str,
    **kwargs: str | None,
) -> WorkerSupervisorSignal:
    return WorkerSupervisorSignal(kind, state.lane, worker_token, **kwargs)


def bootstrap_worker_supervisor(
    lane: EvaluationKind,
    tier_profile_id: str,
    worker_token: str | None,
    *,
    enabled: bool = True,
) -> WorkerSupervisorReduction:
    """外部採番tokenから初期stateとspawn intentを作る。"""
    if not isinstance(lane, EvaluationKind) or not _valid_token(tier_profile_id):
        raise ValueError("laneとtier_profile_idが不正です")
    if enabled and not _valid_token(worker_token):
        raise ValueError("有効laneには外部採番worker_tokenが必要です")
    if not enabled and worker_token is not None:
        raise ValueError("無効laneへworker_tokenを割り当てられません")
    health = WorkerHealth.STARTING if enabled else WorkerHealth.DISABLED
    state = WorkerSupervisorState(lane, health, enabled, tier_profile_id, worker_token)
    if not enabled or worker_token is None:
        return WorkerSupervisorReduction(state)
    intent = _intent(
        state, WorkerSupervisorIntentKind.SPAWN_WORKER,
        worker_token, tier_profile_id=tier_profile_id,
    )
    return WorkerSupervisorReduction(state, (intent,))


def _reject(
    state: WorkerSupervisorState,
    reason: WorkerSupervisorRejectReason,
) -> WorkerSupervisorReduction:
    return WorkerSupervisorReduction(state, reject_reason=reason)


def _prepare_event(
    state: WorkerSupervisorState,
    event: WorkerSupervisorEvent,
) -> WorkerSupervisorReduction | WorkerSupervisorState:
    if event.lane is not state.lane:
        return _reject(state, WorkerSupervisorRejectReason.LANE_MISMATCH)
    if not _valid_time(event.event_seq):
        return _reject(state, WorkerSupervisorRejectReason.INVALID_EVENT)
    if state.last_event_seq is not None and event.event_seq == state.last_event_seq:
        return _reject(state, WorkerSupervisorRejectReason.DUPLICATE_EVENT)
    if state.last_event_seq is not None and event.event_seq < state.last_event_seq:
        return _reject(state, WorkerSupervisorRejectReason.REVERSED_EVENT)
    sequenced = replace(state, last_event_seq=event.event_seq)
    if not state.enabled:
        return _reject(sequenced, WorkerSupervisorRejectReason.LANE_DISABLED)
    return sequenced


def _check_token(
    state: WorkerSupervisorState,
    worker_token: str | None,
) -> WorkerSupervisorReduction | None:
    if not _valid_token(worker_token) or worker_token != state.worker_token:
        return _reject(state, WorkerSupervisorRejectReason.STALE_WORKER_TOKEN)
    return None


def _handle_spawned(
    state: WorkerSupervisorState,
    event: WorkerSupervisorEvent,
) -> WorkerSupervisorReduction:
    rejected = _check_token(state, event.worker_token)
    if rejected is not None:
        return rejected
    if state.health not in (WorkerHealth.STARTING, WorkerHealth.RESTARTING):
        return _reject(state, WorkerSupervisorRejectReason.UNEXPECTED_EVENT)
    token = state.worker_token
    assert token is not None
    intent = _intent(state, WorkerSupervisorIntentKind.PREWARM_WORKER, token)
    return WorkerSupervisorReduction(state, (intent,))


def _prewarm_failed(
    state: WorkerSupervisorState,
    event: WorkerSupervisorEvent,
) -> WorkerSupervisorReduction:
    grace = event.termination_grace_deadline_monotonic_ns
    if not _valid_time(grace):
        return _reject(state, WorkerSupervisorRejectReason.INVALID_EVENT)
    token = state.worker_token
    assert token is not None
    updated = replace(state, health=WorkerHealth.DEGRADED, termination_grace_deadline_monotonic_ns=grace)
    signal = _signal(
        updated, WorkerSupervisorSignalKind.WORKER_FAULT,
        token, reason="prewarm_failed",
    )
    return WorkerSupervisorReduction(updated, signals=(signal,))


def _handle_prewarm(
    state: WorkerSupervisorState,
    event: WorkerSupervisorEvent,
) -> WorkerSupervisorReduction:
    rejected = _check_token(state, event.worker_token)
    if rejected is not None:
        return rejected
    if state.health not in (WorkerHealth.STARTING, WorkerHealth.RESTARTING):
        return _reject(state, WorkerSupervisorRejectReason.UNEXPECTED_EVENT)
    if not isinstance(event.prewarm_succeeded, bool):
        return _reject(state, WorkerSupervisorRejectReason.INVALID_EVENT)
    if not event.prewarm_succeeded:
        return _prewarm_failed(state, event)
    token = state.worker_token
    assert token is not None
    updated = replace(state, health=WorkerHealth.HEALTHY, termination_grace_deadline_monotonic_ns=None)
    signal = _signal(updated, WorkerSupervisorSignalKind.RESUME_LATEST_WAITING, token)
    return WorkerSupervisorReduction(updated, signals=(signal,))


def _valid_job_timing(event: WorkerSupervisorEvent) -> bool:
    hard = event.hard_deadline_monotonic_ns
    grace = event.termination_grace_deadline_monotonic_ns
    return _valid_time(hard) and _valid_time(grace) and grace > hard


def _handle_job_started(
    state: WorkerSupervisorState,
    event: WorkerSupervisorEvent,
) -> WorkerSupervisorReduction:
    rejected = _check_token(state, event.worker_token)
    if rejected is not None:
        return rejected
    if not state.accepts_jobs:
        return _reject(state, WorkerSupervisorRejectReason.WORKER_NOT_READY)
    if state.active_request_id is not None:
        return _reject(state, WorkerSupervisorRejectReason.JOB_ALREADY_RUNNING)
    if not _valid_token(event.request_id) or not _valid_job_timing(event):
        return _reject(state, WorkerSupervisorRejectReason.INVALID_EVENT)
    updated = replace(
        state,
        active_request_id=event.request_id,
        hard_deadline_monotonic_ns=event.hard_deadline_monotonic_ns,
        termination_grace_deadline_monotonic_ns=event.termination_grace_deadline_monotonic_ns,
        cancel_requested=False,
    )
    return WorkerSupervisorReduction(updated)


def _handle_result(
    state: WorkerSupervisorState,
    event: WorkerSupervisorEvent,
) -> WorkerSupervisorReduction:
    rejected = _check_token(state, event.worker_token)
    if rejected is not None:
        return rejected
    if not state.trusts_results:
        return _reject(state, WorkerSupervisorRejectReason.WORKER_NOT_READY)
    if event.request_id != state.active_request_id or not _valid_token(event.request_id):
        return _reject(state, WorkerSupervisorRejectReason.REQUEST_MISMATCH)
    token = state.worker_token
    assert token is not None
    updated = replace(
        state, active_request_id=None, hard_deadline_monotonic_ns=None,
        termination_grace_deadline_monotonic_ns=None, cancel_requested=False,
        consecutive_timeouts=0,
    )
    signal = _signal(updated, WorkerSupervisorSignalKind.RESULT_TRUSTED, token, request_id=event.request_id)
    return WorkerSupervisorReduction(updated, signals=(signal,))


def _fault_reduction(
    state: WorkerSupervisorState,
    grace_deadline: int,
    reason: str,
) -> WorkerSupervisorReduction:
    token = state.worker_token
    assert token is not None
    updated = replace(
        state, health=WorkerHealth.DEGRADED,
        termination_grace_deadline_monotonic_ns=grace_deadline,
        cancel_requested=state.active_request_id is not None,
    )
    intents = ()
    if state.active_request_id is not None:
        intents = (_intent(
            updated, WorkerSupervisorIntentKind.CANCEL_JOB, token,
            request_id=state.active_request_id, reason=reason,
        ),)
    signal = _signal(updated, WorkerSupervisorSignalKind.WORKER_FAULT, token, reason=reason)
    return WorkerSupervisorReduction(updated, intents, (signal,))


def _handle_fault(
    state: WorkerSupervisorState,
    event: WorkerSupervisorEvent,
) -> WorkerSupervisorReduction:
    rejected = _check_token(state, event.worker_token)
    if rejected is not None:
        return rejected
    grace = event.termination_grace_deadline_monotonic_ns
    if not _valid_time(grace):
        return _reject(state, WorkerSupervisorRejectReason.INVALID_EVENT)
    return _fault_reduction(state, grace, "worker_fault")


def _timeout_reduction(state: WorkerSupervisorState) -> WorkerSupervisorReduction:
    deadline = state.termination_grace_deadline_monotonic_ns
    assert deadline is not None
    reduction = _fault_reduction(state, deadline, "hard_deadline_exceeded")
    timeouts = state.consecutive_timeouts + 1
    updated = replace(reduction.state, consecutive_timeouts=timeouts)
    intents = reduction.intents
    if timeouts >= TIER_CHANGE_SUGGESTION_TIMEOUTS:
        token = updated.worker_token
        assert token is not None
        suggestion = _intent(
            updated, WorkerSupervisorIntentKind.SUGGEST_TIER_CHANGE,
            token, tier_profile_id=updated.tier_profile_id,
            reason="consecutive_timeouts",
        )
        intents = (*intents, suggestion)
    return replace(reduction, state=updated, intents=intents)


def _restart_reduction(
    state: WorkerSupervisorState,
    replacement_token: str | None,
) -> WorkerSupervisorReduction:
    if not _valid_token(replacement_token) or replacement_token == state.worker_token:
        return _reject(state, WorkerSupervisorRejectReason.MISSING_REPLACEMENT_TOKEN)
    old_token = state.worker_token
    assert old_token is not None and replacement_token is not None
    updated = replace(
        state, health=WorkerHealth.RESTARTING, worker_token=replacement_token,
        active_request_id=None, hard_deadline_monotonic_ns=None,
        termination_grace_deadline_monotonic_ns=None, cancel_requested=False,
    )
    intents = (
        _intent(updated, WorkerSupervisorIntentKind.TERMINATE_WORKER, old_token, reason="grace_expired"),
        _intent(
            updated, WorkerSupervisorIntentKind.SPAWN_WORKER,
            replacement_token, tier_profile_id=updated.tier_profile_id,
        ),
    )
    return WorkerSupervisorReduction(updated, intents)


def _handle_timer(
    state: WorkerSupervisorState,
    event: WorkerSupervisorEvent,
) -> WorkerSupervisorReduction:
    now = event.observed_monotonic_ns
    if not _valid_time(now):
        return _reject(state, WorkerSupervisorRejectReason.INVALID_EVENT)
    if state.health is WorkerHealth.DEGRADED:
        grace = state.termination_grace_deadline_monotonic_ns
        if grace is not None and now >= grace:
            return _restart_reduction(state, event.replacement_worker_token)
        return WorkerSupervisorReduction(state)
    if state.health is not WorkerHealth.HEALTHY or state.active_request_id is None:
        return WorkerSupervisorReduction(state)
    hard = state.hard_deadline_monotonic_ns
    if hard is None:
        return _reject(state, WorkerSupervisorRejectReason.INVALID_EVENT)
    if now < hard or state.cancel_requested:
        return WorkerSupervisorReduction(state)
    return _timeout_reduction(state)


def reduce_worker_event(
    state: WorkerSupervisorState,
    event: WorkerSupervisorEvent,
) -> WorkerSupervisorReduction:
    """一eventを決定論的に適用し、実副作用をintentとして返す。"""
    prepared = _prepare_event(state, event)
    if isinstance(prepared, WorkerSupervisorReduction):
        return prepared
    handlers = {
        WorkerSupervisorEventKind.WORKER_SPAWNED: _handle_spawned,
        WorkerSupervisorEventKind.PREWARM_COMPLETED: _handle_prewarm,
        WorkerSupervisorEventKind.JOB_STARTED: _handle_job_started,
        WorkerSupervisorEventKind.RESULT_RECEIVED: _handle_result,
        WorkerSupervisorEventKind.WORKER_FAULT: _handle_fault,
        WorkerSupervisorEventKind.TIMER_OBSERVED: _handle_timer,
    }
    handler = handlers.get(event.kind)
    if handler is None:
        return _reject(prepared, WorkerSupervisorRejectReason.INVALID_EVENT)
    return handler(prepared, event)


__all__ = [
    "TIER_CHANGE_SUGGESTION_TIMEOUTS",
    "WorkerSupervisorEvent",
    "WorkerSupervisorEventKind",
    "WorkerSupervisorIntent",
    "WorkerSupervisorIntentKind",
    "WorkerSupervisorReduction",
    "WorkerSupervisorRejectReason",
    "WorkerSupervisorSignal",
    "WorkerSupervisorSignalKind",
    "WorkerSupervisorState",
    "bootstrap_worker_supervisor",
    "reduce_worker_event",
]
