"""Phase J予測jobのlatest-wins schedulingとcommit gate。

安全event（boundary、hold、fault、terminal、health）は本schedulerと2Hz集約の対象外で、
SingleWriterEventLoopが即時処理する。ここではHubへpublishせずreducer受理候補だけを返す。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Callable, Mapping, TypeAlias

from .contracts import WorkerHealth

NUMERIC_PUBLISH_INTERVAL_NS = 500_000_000
SAFETY_EVENTS_BYPASS_NUMERIC_GATE = frozenset(
    {
        "formal_boundary",
        "hold_started",
        "hold_expired",
        "integrity_fault",
        "terminal_candidate",
        "terminal_confirmed",
        "worker_fault",
        "capture_status",
        "health",
    }
)
# safety event即時処理とcommit後publish前boundary割込みはSingleWriter統合試験で継続する。
Clock: TypeAlias = Callable[[], int]


class EvaluationKind(StrEnum):
    PRACTICAL = "practical"
    BEST_ACTION = "best_action"


class SchedulerIntentKind(StrEnum):
    START_JOB = "start_job"
    CANCEL_JOB = "cancel_job"


class CommitRejectReason(StrEnum):
    MISSING_TOKEN = "missing_token"
    INVALIDATED_JOB = "invalidated_job"
    REQUEST_ID_MISMATCH = "request_id_mismatch"
    INPUT_DIGEST_MISMATCH = "input_digest_mismatch"
    SESSION_ID_MISMATCH = "session_id_mismatch"
    MATCH_ID_MISMATCH = "match_id_mismatch"
    CAPTURE_SESSION_ID_MISMATCH = "capture_session_id_mismatch"
    CAPTURE_SEQ_MISMATCH = "capture_seq_mismatch"
    CAUSAL_CUTOFF_MISMATCH = "causal_cutoff_mismatch"
    INPUT_GENERATION_MISMATCH = "input_generation_mismatch"
    EVALUATION_KIND_MISMATCH = "evaluation_kind_mismatch"
    TIER_PROFILE_MISMATCH = "tier_profile_mismatch"
    ASSET_BUNDLE_MISMATCH = "asset_bundle_mismatch"
    DEADLINE_MISMATCH = "deadline_mismatch"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    WORKER_RESTARTING = "worker_restarting"
    WORKER_UNHEALTHY = "worker_unhealthy"


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return deepcopy(value)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = _freeze_value(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - 内部不変条件
        raise TypeError("mappingの凍結結果がmappingではありません")
    return frozen


def _require_nonempty_string(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name}は非空文字列である必要があります")


def _require_nonnegative_int(field_name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name}はboolでない非負整数である必要があります")


@dataclass(frozen=True, slots=True)
class CommitToken:
    """immutable入力snapshotを一意に指すcommit条件。"""

    session_id: str
    match_id: str
    capture_session_id: str
    capture_seq: int
    causal_cutoff_digest: str
    input_generation: int
    evaluation_kind: EvaluationKind
    tier_profile_id: str
    asset_bundle_id: str
    deadline_monotonic_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_kind, EvaluationKind):
            raise ValueError("evaluation_kindはEvaluationKindである必要があります")
        for field_name in (
            "session_id", "match_id", "capture_session_id", "causal_cutoff_digest",
            "tier_profile_id", "asset_bundle_id",
        ):
            _require_nonempty_string(field_name, getattr(self, field_name))
        for field_name in ("capture_seq", "input_generation", "deadline_monotonic_ns"):
            _require_nonnegative_int(field_name, getattr(self, field_name))


@dataclass(frozen=True, slots=True)
class JobRequest:
    """workerへ渡す不変job。"""

    request_id: str
    token: CommitToken
    input_digest: str
    input_snapshot: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.token, CommitToken):
            raise ValueError("tokenはCommitTokenである必要があります")
        _require_nonempty_string("request_id", self.request_id)
        _require_nonempty_string("input_digest", self.input_digest)
        object.__setattr__(self, "input_snapshot", _freeze_mapping(self.input_snapshot))


@dataclass(frozen=True, slots=True)
class JobResult:
    """worker境界から戻る未信頼result。token欠損もfail-closedで表せる。"""

    evaluation_kind: EvaluationKind
    request_id: str | None
    token: CommitToken | None
    input_digest: str | None
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _freeze_mapping(self.values))


@dataclass(frozen=True, slots=True)
class SchedulerIntent:
    kind: SchedulerIntentKind
    evaluation_kind: EvaluationKind
    request: JobRequest | None = None
    request_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class LaneSchedule:
    """laneごとにrunning 1件とwaiting最新1件だけを保持する。"""

    running: JobRequest | None = None
    waiting: JobRequest | None = None
    worker_health: WorkerHealth = WorkerHealth.HEALTHY


@dataclass(frozen=True, slots=True)
class SchedulerState:
    practical: LaneSchedule = field(default_factory=LaneSchedule)
    best_action: LaneSchedule = field(default_factory=LaneSchedule)


@dataclass(frozen=True, slots=True)
class SchedulerDecision:
    """副作用intentまたはreducer受理候補を返す。Hub参照は持たない。"""

    state: SchedulerState
    intents: tuple[SchedulerIntent, ...] = ()
    reducer_candidate: JobResult | None = None
    reject_reason: CommitRejectReason | None = None
    replaced_request_id: str | None = None


@dataclass(frozen=True, slots=True)
class NumericGateState:
    last_published_monotonic_ns: int | None = None
    pending_candidate: JobResult | None = None


@dataclass(frozen=True, slots=True)
class NumericGateDecision:
    state: NumericGateState
    reducer_candidate: JobResult | None = None


_TOKEN_REASONS = (
    ("session_id", CommitRejectReason.SESSION_ID_MISMATCH),
    ("match_id", CommitRejectReason.MATCH_ID_MISMATCH),
    ("capture_session_id", CommitRejectReason.CAPTURE_SESSION_ID_MISMATCH),
    ("capture_seq", CommitRejectReason.CAPTURE_SEQ_MISMATCH),
    ("causal_cutoff_digest", CommitRejectReason.CAUSAL_CUTOFF_MISMATCH),
    ("input_generation", CommitRejectReason.INPUT_GENERATION_MISMATCH),
    ("evaluation_kind", CommitRejectReason.EVALUATION_KIND_MISMATCH),
    ("tier_profile_id", CommitRejectReason.TIER_PROFILE_MISMATCH),
    ("asset_bundle_id", CommitRejectReason.ASSET_BUNDLE_MISMATCH),
    ("deadline_monotonic_ns", CommitRejectReason.DEADLINE_MISMATCH),
)


def initial_scheduler_state() -> SchedulerState:
    return SchedulerState()


def _lane(state: SchedulerState, kind: EvaluationKind) -> LaneSchedule:
    return state.practical if kind is EvaluationKind.PRACTICAL else state.best_action


def _replace_lane(
    state: SchedulerState,
    kind: EvaluationKind,
    lane: LaneSchedule,
) -> SchedulerState:
    field_name = "practical" if kind is EvaluationKind.PRACTICAL else "best_action"
    return replace(state, **{field_name: lane})


def _start_intent(request: JobRequest) -> SchedulerIntent:
    return SchedulerIntent(
        SchedulerIntentKind.START_JOB,
        request.token.evaluation_kind,
        request=request,
        request_id=request.request_id,
    )


def _cancel_intent(request: JobRequest, reason: str) -> SchedulerIntent:
    return SchedulerIntent(
        SchedulerIntentKind.CANCEL_JOB,
        request.token.evaluation_kind,
        request_id=request.request_id,
        reason=reason,
    )


def submit_job(state: SchedulerState, request: JobRequest) -> SchedulerDecision:
    """新jobを配置する。期限済み開始のclock gateはJ2 runtime統合で接続する。"""
    kind = request.token.evaluation_kind
    lane = _lane(state, kind)
    can_start = lane.running is None and lane.worker_health is WorkerHealth.HEALTHY
    if can_start:
        updated = _replace_lane(state, kind, replace(lane, running=request))
        return SchedulerDecision(updated, (_start_intent(request),))
    replaced_id = lane.waiting.request_id if lane.waiting is not None else None
    updated = _replace_lane(state, kind, replace(lane, waiting=request))
    intents = () if lane.running is None else (_cancel_intent(lane.running, "newer_input"),)
    return SchedulerDecision(updated, intents, replaced_request_id=replaced_id)


def start_waiting(state: SchedulerState, kind: EvaluationKind) -> SchedulerDecision:
    """workerが空いたlaneのlatest waitingだけを開始する。"""
    lane = _lane(state, kind)
    can_start = (
        lane.running is None
        and lane.waiting is not None
        and lane.worker_health is WorkerHealth.HEALTHY
    )
    if not can_start:
        return SchedulerDecision(state)
    request = lane.waiting
    updated_lane = replace(lane, running=request, waiting=None)
    return SchedulerDecision(_replace_lane(state, kind, updated_lane), (_start_intent(request),))


def set_worker_health(
    state: SchedulerState,
    kind: EvaluationKind,
    health: WorkerHealth,
) -> SchedulerDecision:
    lane = _lane(state, kind)
    intents: tuple[SchedulerIntent, ...] = ()
    if health is WorkerHealth.RESTARTING and lane.running is not None:
        intents = (_cancel_intent(lane.running, "worker_restarting"),)
    running = None if health is WorkerHealth.RESTARTING else lane.running
    updated_lane = replace(lane, running=running, worker_health=health)
    updated = _replace_lane(state, kind, updated_lane)
    if health is not WorkerHealth.HEALTHY:
        return SchedulerDecision(updated, intents)
    promoted = start_waiting(updated, kind)
    return SchedulerDecision(promoted.state, (*intents, *promoted.intents))


def invalidate_all(state: SchedulerState, reason: str) -> SchedulerDecision:
    """boundary、session/capture切替で両laneのjobを原子的に無効化する。"""
    intents = tuple(
        _cancel_intent(lane.running, reason)
        for lane in (state.practical, state.best_action)
        if lane.running is not None
    )
    practical = replace(state.practical, running=None, waiting=None)
    best_action = replace(state.best_action, running=None, waiting=None)
    return SchedulerDecision(replace(state, practical=practical, best_action=best_action), intents)


def _token_mismatch(
    actual: CommitToken | None,
    expected: CommitToken,
) -> CommitRejectReason | None:
    if actual is None:
        return CommitRejectReason.MISSING_TOKEN
    for field_name, reason in _TOKEN_REASONS:
        if getattr(actual, field_name) != getattr(expected, field_name):
            return reason
    return None


def _commit_reject_reason(
    lane: LaneSchedule,
    result: JobResult,
    current_token: CommitToken | None,
    now_ns: int,
) -> CommitRejectReason | None:
    if lane.worker_health is WorkerHealth.RESTARTING:
        return CommitRejectReason.WORKER_RESTARTING
    if lane.worker_health is not WorkerHealth.HEALTHY:
        return CommitRejectReason.WORKER_UNHEALTHY
    if current_token is None or lane.running is None:
        return CommitRejectReason.INVALIDATED_JOB
    mismatch = _token_mismatch(result.token, current_token)
    if mismatch is not None:
        return mismatch
    if result.token is not None and result.token.evaluation_kind is not result.evaluation_kind:
        return CommitRejectReason.EVALUATION_KIND_MISMATCH
    mismatch = _token_mismatch(result.token, lane.running.token)
    if mismatch is not None:
        return mismatch
    if result.request_id != lane.running.request_id:
        return CommitRejectReason.REQUEST_ID_MISMATCH
    if result.input_digest != lane.running.input_digest:
        return CommitRejectReason.INPUT_DIGEST_MISMATCH
    if now_ns > current_token.deadline_monotonic_ns:
        return CommitRejectReason.DEADLINE_EXCEEDED
    return None


def _finish_matching_job(
    state: SchedulerState,
    result: JobResult,
) -> tuple[SchedulerState, tuple[SchedulerIntent, ...]]:
    """token+requestがrunningなら、不正digest結果でもslotを閉じてlatestを進める。"""
    lane = _lane(state, result.evaluation_kind)
    same_job = (
        lane.running is not None
        and result.request_id == lane.running.request_id
        and result.token == lane.running.token
    )
    if not same_job:
        return state, ()
    cleared = _replace_lane(state, result.evaluation_kind, replace(lane, running=None))
    promoted = start_waiting(cleared, result.evaluation_kind)
    return promoted.state, promoted.intents


def complete_job(
    state: SchedulerState,
    result: JobResult,
    current_token: CommitToken | None,
    now_monotonic_ns: int,
) -> SchedulerDecision:
    """全tokenを照合し、Hubではなくreducerへ渡す候補だけを返す。"""
    if not isinstance(result.evaluation_kind, EvaluationKind):
        return SchedulerDecision(state, reject_reason=CommitRejectReason.EVALUATION_KIND_MISMATCH)
    lane = _lane(state, result.evaluation_kind)
    rejection = _commit_reject_reason(lane, result, current_token, now_monotonic_ns)
    updated, intents = _finish_matching_job(state, result)
    candidate = result if rejection is None else None
    return SchedulerDecision(
        updated,
        intents,
        reducer_candidate=candidate,
        reject_reason=rejection,
    )


def _candidate_generation(candidate: JobResult | None) -> int:
    if candidate is None or candidate.token is None:
        return -1
    return candidate.token.input_generation


def offer_numeric_candidate(
    state: NumericGateState,
    candidate: JobResult,
    clock: Clock,
) -> NumericGateDecision:
    """practical数値だけを最大2Hzへ集約する。安全eventには使用しない。"""
    if candidate.token is None:
        raise ValueError("commit済み数値候補にはtokenが必要です")
    if candidate.evaluation_kind is EvaluationKind.BEST_ACTION:
        return NumericGateDecision(state, candidate)
    now_ns = clock()
    last_ns = state.last_published_monotonic_ns
    due = last_ns is None or now_ns - last_ns >= NUMERIC_PUBLISH_INTERVAL_NS
    if due:
        pending = state.pending_candidate
        latest = (
            candidate
            if _candidate_generation(candidate) >= _candidate_generation(pending)
            else pending
        )
        return NumericGateDecision(NumericGateState(now_ns, None), latest)
    pending = state.pending_candidate
    if _candidate_generation(candidate) >= _candidate_generation(pending):
        pending = candidate
    return NumericGateDecision(replace(state, pending_candidate=pending))


def flush_numeric_candidate(state: NumericGateState, clock: Clock) -> NumericGateDecision:
    """外部timer呼出し時、2Hz期限到達済みのlatest practicalを返す。"""
    if state.pending_candidate is None or state.last_published_monotonic_ns is None:
        return NumericGateDecision(state)
    now_ns = clock()
    if now_ns - state.last_published_monotonic_ns < NUMERIC_PUBLISH_INTERVAL_NS:
        return NumericGateDecision(state)
    return NumericGateDecision(NumericGateState(now_ns, None), state.pending_candidate)


__all__ = [
    "CommitRejectReason",
    "CommitToken",
    "EvaluationKind",
    "JobRequest",
    "JobResult",
    "LaneSchedule",
    "NUMERIC_PUBLISH_INTERVAL_NS",
    "NumericGateDecision",
    "NumericGateState",
    "SAFETY_EVENTS_BYPASS_NUMERIC_GATE",
    "SchedulerDecision",
    "SchedulerIntent",
    "SchedulerIntentKind",
    "SchedulerState",
    "complete_job",
    "flush_numeric_candidate",
    "initial_scheduler_state",
    "invalidate_all",
    "offer_numeric_candidate",
    "set_worker_health",
    "start_waiting",
    "submit_job",
]
