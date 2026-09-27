"""Phase J reducer・scheduler・SnapshotHubを直列化するSingleWriter runtime。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Lock
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

from .contracts import OverlaySnapshot
from .display_projector import PublicationContext, project_snapshot
from .reducer import (
    DisplayState,
    ReducerEvent,
    ReducerEventKind,
    ReducerIntent,
    ReducerIntentKind,
    ReducerState,
    reduce_event,
)
from .scheduler import (
    CommitRejectReason,
    CommitToken,
    EvaluationKind,
    JobRequest,
    JobResult,
    NumericGateState,
    SchedulerIntent,
    SchedulerIntentKind,
    SchedulerState,
    complete_job as scheduler_complete_job,
    flush_numeric_candidate,
    initial_scheduler_state,
    invalidate_all,
    offer_numeric_candidate,
    submit_job,
)
from .snapshot_hub import PublishResult, SnapshotHub
from .validator import ValidationIssue, ValidationReport, make_fail_closed_snapshot


class RuntimeIntentKind(StrEnum):
    START_JOB = "start_job"
    CANCEL_JOB = "cancel_job"
    INVALIDATE_JOBS = "invalidate_jobs"
    SCHEDULE_TIMER = "schedule_timer"
    CANCEL_TIMER = "cancel_timer"
    REJECT_EVENT = "reject_event"


class RuntimeRejectReason(StrEnum):
    MISSING_JOB_CONTEXT = "missing_job_context"
    JOB_DEADLINE_EXCEEDED = "job_deadline_exceeded"
    WORKER_UNHEALTHY = "worker_unhealthy"
    RESULT_REQUIRES_COMMIT_GATE = "result_requires_commit_gate"
    SESSION_ROTATION_REQUIRED = "session_rotation_required"
    REDUCER_REJECTED = "reducer_rejected"


RejectReason: TypeAlias = RuntimeRejectReason | CommitRejectReason | str
_GATE_RESET_REASONS = frozenset(
    {"formal_boundary", "intermission", "new_session", "new_capture_session"}
)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return deepcopy(value)


def _freeze_optional_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    frozen = _freeze_value(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - 内部不変条件
        raise TypeError("job inputの凍結結果がmappingではありません")
    return frozen


def _require_nonnegative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name}はboolでない非負整数である必要があります")


@dataclass(frozen=True, slots=True)
class RuntimeInputContext:
    """event適用時刻と、job開始に必要な不変入力を外部から受け取る。"""

    published_at_utc: str
    publish_monotonic_ms: int
    now_monotonic_ns: int
    causal_cutoff_digest: str | None = None
    deadline_monotonic_ns: int | None = None
    input_snapshot: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.published_at_utc, str) or not self.published_at_utc:
            raise ValueError("published_at_utcは非空文字列である必要があります")
        _require_nonnegative_int("publish_monotonic_ms", self.publish_monotonic_ms)
        _require_nonnegative_int("now_monotonic_ns", self.now_monotonic_ns)
        if self.deadline_monotonic_ns is not None:
            _require_nonnegative_int("deadline_monotonic_ns", self.deadline_monotonic_ns)
        object.__setattr__(self, "input_snapshot", _freeze_optional_mapping(self.input_snapshot))


@dataclass(frozen=True, slots=True)
class RuntimeCommitContext:
    """worker結果をreducer event化する識別子・時刻。全て呼出し側が注入する。"""

    event_seq: int
    content_digest: str
    completed_monotonic_ms: int
    published_at_utc: str
    publish_monotonic_ms: int
    now_monotonic_ns: int

    def __post_init__(self) -> None:
        for name in ("event_seq", "completed_monotonic_ms", "publish_monotonic_ms", "now_monotonic_ns"):
            _require_nonnegative_int(name, getattr(self, name))
        if not isinstance(self.content_digest, str) or not self.content_digest:
            raise ValueError("content_digestは非空文字列である必要があります")
        if not isinstance(self.published_at_utc, str) or not self.published_at_utc:
            raise ValueError("published_at_utcは非空文字列である必要があります")


@dataclass(frozen=True, slots=True)
class RuntimeFlushContext:
    """保留practicalをpublishする外部timer入力。"""

    event_seq: int
    content_digest: str
    published_at_utc: str
    publish_monotonic_ms: int
    now_monotonic_ns: int

    def __post_init__(self) -> None:
        for name in ("event_seq", "publish_monotonic_ms", "now_monotonic_ns"):
            _require_nonnegative_int(name, getattr(self, name))
        if not self.content_digest or not self.published_at_utc:
            raise ValueError("flushのdigestとUTC時刻は非空である必要があります")


@dataclass(frozen=True, slots=True)
class RuntimeIntent:
    """runtime外部で実行する型付き副作用。"""

    kind: RuntimeIntentKind
    payload: Mapping[str, Any] = field(default_factory=dict)
    request: JobRequest | None = None

    def __post_init__(self) -> None:
        frozen = _freeze_value(self.payload)
        object.__setattr__(self, "payload", frozen)


@dataclass(frozen=True, slots=True)
class RuntimeStep:
    reducer_state: ReducerState
    scheduler_state: SchedulerState
    snapshot: OverlaySnapshot
    intents: tuple[RuntimeIntent, ...] = ()
    publish_result: PublishResult | None = None
    accepted: bool = True
    reject_reason: RejectReason | None = None
    fail_closed: bool = False


@dataclass(frozen=True, slots=True)
class _RuntimeCheckpoint:
    reducer_state: ReducerState
    scheduler_state: SchedulerState
    numeric_gate_state: NumericGateState
    pending_completed_monotonic_ms: int | None


class RuntimePublishError(RuntimeError):
    """Hub本体がfail-closed再試行も受理できない。"""

    def __init__(self, snapshot: OverlaySnapshot, cause: Exception) -> None:
        self.snapshot = snapshot
        self.cause = cause
        super().__init__("SnapshotHubがfail-closed snapshotも受理できません")


def _publication_context(context: RuntimeInputContext | RuntimeCommitContext | RuntimeFlushContext) -> PublicationContext:
    return PublicationContext(0, context.published_at_utc, context.publish_monotonic_ms)


def _runtime_report(message: str) -> ValidationReport:
    issue = ValidationIssue("RUNTIME", "$", message)
    return ValidationReport((issue,))


def _scheduler_runtime_intents(intents: tuple[SchedulerIntent, ...]) -> tuple[RuntimeIntent, ...]:
    translated: list[RuntimeIntent] = []
    for intent in intents:
        kind = (
            RuntimeIntentKind.START_JOB
            if intent.kind is SchedulerIntentKind.START_JOB
            else RuntimeIntentKind.CANCEL_JOB
        )
        payload = {
            "lane": intent.evaluation_kind.value,
            "request_id": intent.request_id,
            "reason": intent.reason,
        }
        translated.append(RuntimeIntent(kind, payload, request=intent.request))
    return tuple(translated)


def _current_token(state: SchedulerState, kind: EvaluationKind) -> CommitToken | None:
    lane = state.practical if kind is EvaluationKind.PRACTICAL else state.best_action
    request = lane.waiting or lane.running
    return request.token if request is not None else None


class SingleWriterRuntime:
    """一つのlock内で状態遷移、commit判定、射影、publishを完了する。"""

    def __init__(
        self,
        reducer_state: ReducerState,
        hub: SnapshotHub,
        scheduler_state: SchedulerState | None = None,
        numeric_gate_state: NumericGateState | None = None,
    ) -> None:
        self._lock = Lock()
        self._reducer_state = reducer_state
        self._scheduler_state = scheduler_state or initial_scheduler_state()
        self._numeric_gate_state = numeric_gate_state or NumericGateState()
        self._pending_completed_monotonic_ms: int | None = None
        self._fail_closed_latch: RuntimeRejectReason | None = None
        self._hub = hub

    @property
    def reducer_state(self) -> ReducerState:
        with self._lock:
            return self._reducer_state

    @property
    def scheduler_state(self) -> SchedulerState:
        with self._lock:
            return self._scheduler_state

    @property
    def numeric_gate_state(self) -> NumericGateState:
        with self._lock:
            return self._numeric_gate_state

    @property
    def latest(self) -> OverlaySnapshot:
        with self._lock:
            return self._hub.latest

    @property
    def fail_closed_latch(self) -> RuntimeRejectReason | None:
        with self._lock:
            return self._fail_closed_latch

    def _job_request(
        self,
        intent: ReducerIntent,
        context: RuntimeInputContext,
    ) -> tuple[JobRequest | None, RuntimeRejectReason | None]:
        payload = intent.payload
        try:
            kind = EvaluationKind(str(payload["lane"]))
            deadline = context.deadline_monotonic_ns
            if deadline is None:
                return None, RuntimeRejectReason.MISSING_JOB_CONTEXT
            if deadline < context.now_monotonic_ns:
                return None, RuntimeRejectReason.JOB_DEADLINE_EXCEEDED
            required = (
                self._reducer_state.match_id, self._reducer_state.capture_session_id,
                self._reducer_state.last_capture_seq, context.causal_cutoff_digest,
                context.input_snapshot, payload.get("request_id"),
                payload.get("input_generation"), payload.get("input_digest"),
            )
            if any(item is None for item in required):
                return None, RuntimeRejectReason.MISSING_JOB_CONTEXT
            token = self._make_token(kind, payload, context, deadline)
            request = JobRequest(str(payload["request_id"]), token, str(payload["input_digest"]), context.input_snapshot)
            return request, None
        except (KeyError, TypeError, ValueError):
            return None, RuntimeRejectReason.MISSING_JOB_CONTEXT

    def _make_token(
        self,
        kind: EvaluationKind,
        payload: Mapping[str, Any],
        context: RuntimeInputContext,
        deadline: int,
    ) -> CommitToken:
        state = self._reducer_state
        return CommitToken(
            session_id=state.session_id,
            match_id=str(state.match_id),
            capture_session_id=str(state.capture_session_id),
            capture_seq=int(state.last_capture_seq),
            causal_cutoff_digest=str(context.causal_cutoff_digest),
            input_generation=int(payload["input_generation"]),
            evaluation_kind=kind,
            tier_profile_id=state.tier_profile_id,
            asset_bundle_id=state.asset_bundle_id,
            deadline_monotonic_ns=deadline,
        )

    def _invalidate(self, intent: ReducerIntent) -> tuple[RuntimeIntent, ...]:
        reason = str(intent.payload.get("reason", "reducer_invalidation"))
        decision = invalidate_all(self._scheduler_state, reason)
        self._scheduler_state = decision.state
        last_ns = self._numeric_gate_state.last_published_monotonic_ns
        self._numeric_gate_state = NumericGateState(None if reason in _GATE_RESET_REASONS else last_ns)
        self._pending_completed_monotonic_ms = None
        marker = RuntimeIntent(RuntimeIntentKind.INVALIDATE_JOBS, {"reason": reason})
        return (marker, *_scheduler_runtime_intents(decision.intents))

    def _interpret_intents(
        self,
        intents: tuple[ReducerIntent, ...],
        context: RuntimeInputContext | RuntimeCommitContext | RuntimeFlushContext,
    ) -> tuple[tuple[RuntimeIntent, ...], RuntimeRejectReason | None]:
        output: list[RuntimeIntent] = []
        rejection: RuntimeRejectReason | None = None
        for intent in intents:
            if intent.kind is ReducerIntentKind.INVALIDATE_JOBS:
                output.extend(self._invalidate(intent))
            elif intent.kind is ReducerIntentKind.START_JOB:
                if not isinstance(context, RuntimeInputContext):
                    rejection = RuntimeRejectReason.MISSING_JOB_CONTEXT
                    continue
                request, reason = self._job_request(intent, context)
                if request is None:
                    rejection = reason
                    continue
                decision = submit_job(self._scheduler_state, request)
                self._scheduler_state = decision.state
                output.extend(_scheduler_runtime_intents(decision.intents))
            else:
                translated = self._translate_reducer_intent(intent)
                if translated is not None:
                    output.append(translated)
        if rejection is not None:
            output.append(RuntimeIntent(RuntimeIntentKind.REJECT_EVENT, {"reason": rejection.value}))
        return tuple(output), rejection

    @staticmethod
    def _translate_reducer_intent(intent: ReducerIntent) -> RuntimeIntent | None:
        mapping = {
            ReducerIntentKind.SCHEDULE_TIMER: RuntimeIntentKind.SCHEDULE_TIMER,
            ReducerIntentKind.CANCEL_TIMER: RuntimeIntentKind.CANCEL_TIMER,
            ReducerIntentKind.REJECT_EVENT: RuntimeIntentKind.REJECT_EVENT,
        }
        kind = mapping.get(intent.kind)
        return None if kind is None else RuntimeIntent(kind, intent.payload)

    def _publish(
        self,
        context: RuntimeInputContext | RuntimeCommitContext | RuntimeFlushContext,
        forced_reason: RuntimeRejectReason | None = None,
        state: ReducerState | None = None,
    ) -> tuple[PublishResult, bool]:
        candidate = project_snapshot(state or self._reducer_state, _publication_context(context))
        effective_reason = forced_reason or self._fail_closed_latch
        forced = effective_reason is not None
        if effective_reason is not None:
            candidate = make_fail_closed_snapshot(candidate, _runtime_report(effective_reason.value))
        try:
            result = self._hub.publish(candidate)
            return result, forced or result.fail_closed
        except Exception as first_error:
            closed = self._hub_identity_fail_closed(str(first_error))
            try:
                return self._hub.publish(closed), True
            except Exception as second_error:
                raise RuntimePublishError(closed, second_error) from first_error

    def _hub_identity_fail_closed(self, message: str) -> OverlaySnapshot:
        """Hubの現在identityへ載せ替え、拒否候補のidentityを引き継がない。"""
        latest = self._hub.latest
        closed = make_fail_closed_snapshot(latest, _runtime_report(message))
        payload = closed.to_mapping()
        payload["identity"] = dict(latest.identity)
        current_seq = payload["identity"].get("stream_seq")
        if not isinstance(current_seq, int) or isinstance(current_seq, bool):
            raise RuntimePublishError(closed, ValueError("Hub stream_seqが不正です"))
        payload["identity"]["stream_seq"] = current_seq + 1
        return OverlaySnapshot.from_mapping(payload)

    def _checkpoint(self) -> _RuntimeCheckpoint:
        return _RuntimeCheckpoint(
            self._reducer_state,
            self._scheduler_state,
            self._numeric_gate_state,
            self._pending_completed_monotonic_ms,
        )

    def _abort_event(
        self,
        checkpoint: _RuntimeCheckpoint,
        reason: RuntimeRejectReason,
    ) -> tuple[RuntimeIntent, ...]:
        """reducerだけを戻し、旧jobと保留数値は再利用不能にする。"""
        self._reducer_state = checkpoint.reducer_state
        decision = invalidate_all(checkpoint.scheduler_state, reason.value)
        self._scheduler_state = decision.state
        last_ns = checkpoint.numeric_gate_state.last_published_monotonic_ns
        self._numeric_gate_state = NumericGateState(last_ns, None)
        self._pending_completed_monotonic_ms = None
        self._fail_closed_latch = reason
        marker = RuntimeIntent(RuntimeIntentKind.INVALIDATE_JOBS, {"reason": reason.value})
        rejected = RuntimeIntent(RuntimeIntentKind.REJECT_EVENT, {"reason": reason.value})
        return (marker, *_scheduler_runtime_intents(decision.intents), rejected)

    def _step(
        self,
        intents: tuple[RuntimeIntent, ...],
        publish_result: PublishResult | None,
        accepted: bool,
        reason: RejectReason | None,
        fail_closed: bool = False,
    ) -> RuntimeStep:
        snapshot = publish_result.snapshot if publish_result is not None else self._hub.latest
        return RuntimeStep(
            self._reducer_state, self._scheduler_state, snapshot, intents,
            publish_result, accepted, reason, fail_closed,
        )

    def apply_event(self, event: ReducerEvent, context: RuntimeInputContext) -> RuntimeStep:
        """reducer eventからHub publishまでを一つのcritical sectionで処理する。"""
        with self._lock:
            checkpoint = self._checkpoint()
            rotation_latched = self._fail_closed_latch is RuntimeRejectReason.SESSION_ROTATION_REQUIRED
            if event.kind is ReducerEventKind.NEW_SESSION or rotation_latched:
                reason = RuntimeRejectReason.SESSION_ROTATION_REQUIRED
                intents = self._abort_event(checkpoint, reason)
                result, fail_closed = self._publish(context, reason)
                return self._step(intents, result, False, reason, fail_closed)
            if event.kind is ReducerEventKind.PREDICTION_COMPLETED:
                reason = RuntimeRejectReason.RESULT_REQUIRES_COMMIT_GATE
                intents = self._abort_event(checkpoint, reason)
                result, fail_closed = self._publish(context, reason)
                return self._step(intents, result, False, reason, fail_closed)
            reduction = reduce_event(self._reducer_state, event)
            self._reducer_state = reduction.state
            intents, runtime_rejection = self._interpret_intents(reduction.intents, context)
            should_publish = any(item.kind is ReducerIntentKind.PUBLISH_SNAPSHOT for item in reduction.intents)
            if runtime_rejection is not None:
                intents = self._abort_event(checkpoint, runtime_rejection)
                result, fail_closed = self._publish(context, runtime_rejection)
                return self._step(intents, result, False, runtime_rejection, fail_closed)
            started = any(item.kind is RuntimeIntentKind.START_JOB for item in intents)
            if (
                event.kind is ReducerEventKind.OBSERVATION_AVAILABLE
                and reduction.state.display_state is DisplayState.AWAITING
                and started
            ):
                self._fail_closed_latch = None
            result: PublishResult | None = None
            fail_closed = False
            if should_publish:
                result, fail_closed = self._publish(context)
            reducer_reason = next(
                (str(item.payload.get("reason")) for item in reduction.intents if item.kind is ReducerIntentKind.REJECT_EVENT),
                None,
            )
            reason: RejectReason | None = runtime_rejection or reducer_reason
            return self._step(intents, result, reduction.accepted and reason is None, reason, fail_closed)

    def _gate_result(
        self,
        candidate: JobResult,
        context: RuntimeCommitContext,
    ) -> tuple[JobResult | None, int | None]:
        previous = self._numeric_gate_state.pending_candidate
        previous_ms = self._pending_completed_monotonic_ms
        decision = offer_numeric_candidate(self._numeric_gate_state, candidate, lambda: context.now_monotonic_ns)
        self._numeric_gate_state = decision.state
        if decision.state.pending_candidate is candidate:
            self._pending_completed_monotonic_ms = context.completed_monotonic_ms
        elif decision.state.pending_candidate is previous:
            self._pending_completed_monotonic_ms = previous_ms
        else:
            self._pending_completed_monotonic_ms = None
        completed_ms = context.completed_monotonic_ms if decision.reducer_candidate is candidate else previous_ms
        return decision.reducer_candidate, completed_ms

    @staticmethod
    def _prediction_event(
        candidate: JobResult,
        context: RuntimeCommitContext | RuntimeFlushContext,
        completed_monotonic_ms: int,
    ) -> ReducerEvent:
        token = candidate.token
        if token is None:  # scheduler commit済み候補では到達しない。
            raise ValueError("commit候補にtokenがありません")
        return ReducerEvent(
            ReducerEventKind.PREDICTION_COMPLETED,
            context.event_seq,
            context.content_digest,
            {
                "lane": candidate.evaluation_kind.value,
                "request_id": candidate.request_id,
                "input_generation": token.input_generation,
                "input_digest": candidate.input_digest,
                "completed_monotonic_ms": completed_monotonic_ms,
                "evaluation": candidate.values,
            },
        )

    def _commit_candidate(
        self,
        candidate: JobResult,
        context: RuntimeCommitContext | RuntimeFlushContext,
        completed_monotonic_ms: int,
        scheduler_intents: tuple[RuntimeIntent, ...] = (),
    ) -> RuntimeStep:
        event = self._prediction_event(candidate, context, completed_monotonic_ms)
        reduction = reduce_event(self._reducer_state, event)
        self._reducer_state = reduction.state
        reducer_intents, runtime_rejection = self._interpret_intents(reduction.intents, context)
        intents = (*scheduler_intents, *reducer_intents)
        should_publish = any(item.kind is ReducerIntentKind.PUBLISH_SNAPSHOT for item in reduction.intents)
        result, fail_closed = (self._publish(context, runtime_rejection) if should_publish else (None, False))
        reason: RejectReason | None = runtime_rejection
        if not reduction.accepted and reason is None:
            reason = RuntimeRejectReason.REDUCER_REJECTED
        return self._step(intents, result, reduction.accepted and reason is None, reason, fail_closed)

    def complete_job(self, result: JobResult, context: RuntimeCommitContext) -> RuntimeStep:
        """current token再検査からpublishまでをboundaryと同じlockで直列化する。"""
        with self._lock:
            kind = result.evaluation_kind
            current = _current_token(self._scheduler_state, kind) if isinstance(kind, EvaluationKind) else None
            decision = scheduler_complete_job(
                self._scheduler_state, result, current, context.now_monotonic_ns,
            )
            self._scheduler_state = decision.state
            intents = _scheduler_runtime_intents(decision.intents)
            if decision.reducer_candidate is None:
                return self._step(intents, None, False, decision.reject_reason)
            candidate, completed_ms = self._gate_result(decision.reducer_candidate, context)
            if candidate is None:
                return self._step(intents, None, True, None)
            if completed_ms is None:  # pragma: no cover - gate内部不変条件
                return self._step(intents, None, False, RuntimeRejectReason.MISSING_JOB_CONTEXT)
            return self._commit_candidate(candidate, context, completed_ms, intents)

    def flush_practical(self, context: RuntimeFlushContext) -> RuntimeStep:
        """外部timerでのみ2Hz保留中のpractical候補をpublishする。"""
        with self._lock:
            completed_ms = self._pending_completed_monotonic_ms
            decision = flush_numeric_candidate(
                self._numeric_gate_state, lambda: context.now_monotonic_ns,
            )
            self._numeric_gate_state = decision.state
            if decision.reducer_candidate is None:
                return self._step((), None, True, None)
            self._pending_completed_monotonic_ms = None
            if completed_ms is None:  # pragma: no cover - gate内部不変条件
                return self._step((), None, False, RuntimeRejectReason.MISSING_JOB_CONTEXT)
            return self._commit_candidate(decision.reducer_candidate, context, completed_ms)


__all__ = [
    "RuntimeCommitContext",
    "RuntimeFlushContext",
    "RuntimeInputContext",
    "RuntimeIntent",
    "RuntimeIntentKind",
    "RuntimePublishError",
    "RuntimeRejectReason",
    "RuntimeStep",
    "SingleWriterRuntime",
]
