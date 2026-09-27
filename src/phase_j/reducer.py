"""Phase Jの副作用を持たないreducer状態機械。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import (
    BoardProvenance,
    EvaluationAvailability,
    EvaluationMode,
    FaultCode,
    HoldReason,
    PracticalOrigin,
    RecognitionQualityStatus,
    RecognitionReason,
    RuntimeStatus,
    RuntimeTier,
    TerminalEvidenceKind,
    TerminalResultCode,
    TerminalState,
    TerminalWinner,
    UpdateReason,
)

HOLD_REPUBLISH_INTERVAL_MS = 1_000
HOLD_EXPIRY_MS = 2_000
TERMINAL_MIN_DISPLAY_MS = 1_000


class MatchPhase(StrEnum):
    PRE_MATCH = "pre_match"
    ACTIVE = "active"
    TERMINAL = "terminal"
    RESULT = "result"
    INTERMISSION = "intermission"
    INTEGRITY_FAULT = "integrity_fault"


class DisplayState(StrEnum):
    HIDDEN = "hidden"
    AWAITING = "awaiting"
    LIVE = "live"
    PHYSICAL_PREDICTION = "physical_prediction"
    HOLD = "hold"
    TERMINAL_FACT = "terminal_fact"
    RESULT = "result"


class JobState(StrEnum):
    IDLE = "idle"
    QUEUED = "queued"
    RESULT_READY = "result_ready"
    REJECTED = "rejected"


class ReducerEventKind(StrEnum):
    NEW_SESSION = "new_session"
    OBSERVATION_AVAILABLE = "observation_available"
    FORMAL_BOUNDARY = "formal_boundary"
    INTERMISSION = "intermission"
    PREDICTION_COMPLETED = "prediction_completed"
    TERMINAL_EVIDENCE = "terminal_evidence"
    WORKER_FAULT = "worker_fault"
    CAPTURE_STATUS = "capture_status"
    CONFIG_CHANGE_REQUESTED = "config_change_requested"
    TIMER_FIRED = "timer_fired"
    HOLD_REASON_ADDED = "hold_reason_added"
    HOLD_REASON_RESOLVED = "hold_reason_resolved"


class ReducerIntentKind(StrEnum):
    PUBLISH_SNAPSHOT = "publish_snapshot"
    START_JOB = "start_job"
    INVALIDATE_JOBS = "invalidate_jobs"
    SCHEDULE_TIMER = "schedule_timer"
    CANCEL_TIMER = "cancel_timer"
    REJECT_EVENT = "reject_event"


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


def _default_assets() -> dict[str, str]:
    return {
        "app_build_id": "dev",
        "recognition_model_hash": "unloaded",
        "recognition_config_hash": "unloaded",
        "prediction_model_hash": "unloaded",
        "calibration_hash": "unloaded",
    }


@dataclass(frozen=True, slots=True)
class Bootstrap:
    """外部composition rootが起動時に注入する値。"""

    session_id: str
    mode: EvaluationMode = EvaluationMode.PRACTICAL
    tier: RuntimeTier = RuntimeTier.STANDARD
    tier_profile_id: str = "standard-v1"
    asset_bundle_id: str = "unloaded"
    assets: Mapping[str, Any] = field(default_factory=_default_assets)
    capture_session_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "assets", _freeze_mapping(self.assets))


@dataclass(frozen=True, slots=True)
class ReducerEvent:
    """通番と内容digestを持つtagged event。"""

    kind: ReducerEventKind
    event_seq: int
    content_digest: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_mapping(self.payload))


@dataclass(frozen=True, slots=True)
class ReducerIntent:
    """reducerが外部実行系へ要求する副作用の記述。"""

    kind: ReducerIntentKind
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_mapping(self.payload))


@dataclass(frozen=True, slots=True)
class LaneState:
    """一つの評価laneの公開候補とjob識別情報。"""

    availability: EvaluationAvailability = EvaluationAvailability.UNAVAILABLE
    job_state: JobState = JobState.IDLE
    request_id: str | None = None
    input_generation: int | None = None
    input_digest: str | None = None
    completed_monotonic_ms: int | None = None
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _freeze_mapping(self.values))


@dataclass(frozen=True, slots=True)
class ReducerState:
    """Phase J reducerの全状態。stream_seqは保持しない。"""

    session_id: str
    assets: Mapping[str, Any]
    mode: EvaluationMode = EvaluationMode.PRACTICAL
    tier: RuntimeTier = RuntimeTier.STANDARD
    tier_profile_id: str = "standard-v1"
    asset_bundle_id: str = "unloaded"
    reducer_revision: int = 0
    match_state_seq: int = 0
    input_generation: int = 0
    match_id: str | None = None
    match_phase: MatchPhase = MatchPhase.PRE_MATCH
    display_state: DisplayState = DisplayState.HIDDEN
    update_reason: UpdateReason = UpdateReason.INITIAL_SNAPSHOT
    last_event_seq: int | None = None
    last_event_digest: str | None = None
    capture_connected: bool = False
    capture_session_id: str | None = None
    retired_capture_session_ids: tuple[str, ...] = ()
    last_capture_seq: int | None = None
    last_capture_digest: str | None = None
    source_available_frame: int | None = None
    source_available_ms: int | None = None
    source_captured_monotonic_ms: int | None = None
    last_confirmed_monotonic_ms: int | None = None
    input_digest: str | None = None
    p1_board_provenance: BoardProvenance = BoardProvenance.UNKNOWN
    p2_board_provenance: BoardProvenance = BoardProvenance.UNKNOWN
    recognition_status: RecognitionQualityStatus = RecognitionQualityStatus.UNTRUSTED
    recognition_reasons: tuple[RecognitionReason, ...] = (RecognitionReason.CALIBRATION_UNAVAILABLE,)
    unresolved_physics: tuple[str, ...] = ()
    physical_prediction_used: bool = False
    practical_lane: LaneState = field(default_factory=LaneState)
    best_action_lane: LaneState = field(default_factory=LaneState)
    hold_reasons: tuple[HoldReason, ...] = ()
    hold_started_source_ms: int | None = None
    hold_started_monotonic_ms: int | None = None
    hold_expired: bool = False
    hold_timer_id: str | None = None
    hold_timer_deadline_monotonic_ms: int | None = None
    hold_timer_tick: int = 0
    fault_codes: tuple[FaultCode, ...] = ()
    pending_config: Mapping[str, Any] = field(default_factory=dict)
    runtime_status: RuntimeStatus = RuntimeStatus.STARTING
    discarded_jobs_since_match_start: int = 0
    terminal_state: TerminalState = TerminalState.NONE
    terminal_winner: TerminalWinner | None = None
    terminal_evidence_kind: TerminalEvidenceKind | None = None
    terminal_result_code: TerminalResultCode | None = None
    terminal_timer_id: str | None = None
    terminal_timer_deadline_monotonic_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "assets", _freeze_mapping(self.assets))
        object.__setattr__(self, "pending_config", _freeze_mapping(self.pending_config))


@dataclass(frozen=True, slots=True)
class Reduction:
    """純粋な状態遷移結果。"""

    state: ReducerState
    intents: tuple[ReducerIntent, ...] = ()
    accepted: bool = True


def initial_state(bootstrap: Bootstrap) -> ReducerState:
    """HTTP開始前に公開可能なhidden初期状態を作る。"""
    return ReducerState(
        session_id=bootstrap.session_id,
        assets=bootstrap.assets,
        mode=bootstrap.mode,
        tier=bootstrap.tier,
        tier_profile_id=bootstrap.tier_profile_id,
        asset_bundle_id=bootstrap.asset_bundle_id,
        capture_connected=bootstrap.capture_session_id is not None,
        capture_session_id=bootstrap.capture_session_id,
    )


def _intent(kind: ReducerIntentKind, **payload: Any) -> ReducerIntent:
    return ReducerIntent(kind=kind, payload=payload)


def _advance(
    state: ReducerState,
    event: ReducerEvent,
    *,
    match_change: bool = True,
    **changes: Any,
) -> ReducerState:
    return replace(
        state,
        reducer_revision=state.reducer_revision + 1,
        match_state_seq=state.match_state_seq + int(match_change),
        last_event_seq=event.event_seq,
        last_event_digest=event.content_digest,
        **changes,
    )


def _sequence_fault(state: ReducerState, event: ReducerEvent) -> FaultCode | None:
    previous = state.last_event_seq
    if previous is None:
        return None
    if event.event_seq == previous and event.content_digest != state.last_event_digest:
        return FaultCode.ID_CONTENT_CONFLICT
    if event.event_seq <= previous:
        return FaultCode.SEQUENCE_REVERSED
    if event.event_seq > previous + 1:
        return FaultCode.SEQUENCE_GAP
    return None


def _fault_reduction(state: ReducerState, event: ReducerEvent, code: FaultCode) -> Reduction:
    failed = replace(
        state,
        reducer_revision=state.reducer_revision + 1,
        match_state_seq=state.match_state_seq + 1,
        last_event_seq=event.event_seq,
        last_event_digest=event.content_digest,
        input_generation=state.input_generation + 1,
        match_phase=MatchPhase.INTEGRITY_FAULT,
        display_state=DisplayState.HIDDEN,
        update_reason=UpdateReason.INTEGRITY_FAULT,
        practical_lane=LaneState(),
        best_action_lane=LaneState(),
        hold_reasons=(),
        hold_started_source_ms=None,
        hold_started_monotonic_ms=None,
        hold_expired=False,
        hold_timer_id=None,
        hold_timer_deadline_monotonic_ms=None,
        hold_timer_tick=0,
        terminal_state=TerminalState.NONE,
        terminal_winner=None,
        terminal_evidence_kind=None,
        terminal_result_code=None,
        terminal_timer_id=None,
        terminal_timer_deadline_monotonic_ms=None,
        fault_codes=tuple(dict.fromkeys((*state.fault_codes, code))),
        runtime_status=RuntimeStatus.DEGRADED,
    )
    intents = (
        _intent(ReducerIntentKind.INVALIDATE_JOBS, reason=code.value),
        _intent(ReducerIntentKind.PUBLISH_SNAPSHOT),
    )
    return Reduction(failed, intents)


def _active_lanes(mode: EvaluationMode) -> tuple[str, ...]:
    if mode is EvaluationMode.BOTH:
        return ("practical", "best_action")
    return (mode.value,)


def _pending_lane(state: ReducerState, lane: str, generation: int, digest: str) -> LaneState:
    match_part = state.match_id or "awaiting-boundary"
    request_id = f"{lane}:{state.session_id}:{match_part}:{generation}"
    return LaneState(
        availability=EvaluationAvailability.PENDING,
        job_state=JobState.QUEUED,
        request_id=request_id,
        input_generation=generation,
        input_digest=digest,
    )


def _start_job_intents(state: ReducerState) -> tuple[ReducerIntent, ...]:
    intents: list[ReducerIntent] = []
    for lane_name in _active_lanes(state.mode):
        lane = state.practical_lane if lane_name == "practical" else state.best_action_lane
        intents.append(
            _intent(
                ReducerIntentKind.START_JOB,
                lane=lane_name,
                request_id=lane.request_id,
                input_generation=lane.input_generation,
                input_digest=lane.input_digest,
            )
        )
    return tuple(intents)


_HOLD_PRIORITY = {reason: index for index, reason in enumerate(HoldReason)}
_LATCHED_PHASES = frozenset({
    MatchPhase.TERMINAL, MatchPhase.RESULT,
    MatchPhase.INTERMISSION, MatchPhase.INTEGRITY_FAULT,
})
_LATCH_SAFE_EVENTS = frozenset(
    {
        ReducerEventKind.FORMAL_BOUNDARY,
        ReducerEventKind.CAPTURE_STATUS,
        ReducerEventKind.CONFIG_CHANGE_REQUESTED,
        ReducerEventKind.TIMER_FIRED,
    }
)
_J1_UNSUPPORTED_EVENTS = frozenset(
    {ReducerEventKind.WORKER_FAULT}
)


def _ordered_reasons(reasons: tuple[HoldReason, ...]) -> tuple[HoldReason, ...]:
    return tuple(sorted(set(reasons), key=_HOLD_PRIORITY.__getitem__))


def _hold_timer_id(state: ReducerState, started_ms: int, tick: int) -> str:
    return f"hold:{state.session_id}:{started_ms}:{tick}"


def _reject_event(state: ReducerState, event: ReducerEvent, reason: str) -> Reduction:
    discarded = state.discarded_jobs_since_match_start
    if event.kind is ReducerEventKind.PREDICTION_COMPLETED:
        discarded += 1
    consumed = _advance(
        state,
        event,
        match_change=False,
        discarded_jobs_since_match_start=discarded,
    )
    return Reduction(
        consumed,
        (_intent(ReducerIntentKind.REJECT_EVENT, reason=reason),),
        False,
    )


def _enter_hold(
    state: ReducerState,
    event: ReducerEvent,
    reason: HoldReason,
    *,
    source_ms: int,
    monotonic_ms: int,
    changes: Mapping[str, Any] | None = None,
    reset_existing: bool = False,
) -> Reduction:
    starting = reset_existing or state.display_state is not DisplayState.HOLD
    previous_reasons = () if reset_existing else state.hold_reasons
    reasons = _ordered_reasons((*previous_reasons, reason))
    started_source = source_ms if starting else state.hold_started_source_ms
    started_mono = monotonic_ms if starting else state.hold_started_monotonic_ms
    tick = 1 if starting else state.hold_timer_tick
    timer_id = _hold_timer_id(state, int(started_mono), tick)
    deadline = int(started_mono) + HOLD_REPUBLISH_INTERVAL_MS * tick
    update = UpdateReason.HOLD_STARTED if starting else UpdateReason.HOLD_REASON_CHANGED
    hold_changes = dict(changes or {})
    hold_changes.update(
        display_state=DisplayState.HOLD,
        update_reason=update,
        hold_reasons=reasons,
        hold_started_source_ms=started_source,
        hold_started_monotonic_ms=started_mono,
        hold_expired=False if starting else state.hold_expired,
        hold_timer_id=timer_id,
        hold_timer_deadline_monotonic_ms=deadline,
        hold_timer_tick=tick,
    )
    updated = _advance(state, event, **hold_changes)
    intents = [_intent(ReducerIntentKind.PUBLISH_SNAPSHOT)]
    if starting:
        intents.append(_intent(ReducerIntentKind.SCHEDULE_TIMER, timer_id=timer_id, deadline_monotonic_ms=deadline))
    return Reduction(updated, tuple(intents))


def _capture_problem(state: ReducerState, event: ReducerEvent) -> FaultCode | str | None:
    session_id = str(event.payload.get("capture_session_id", ""))
    capture_seq = int(event.payload.get("capture_seq", -1))
    capture_digest = str(event.payload.get("capture_digest", ""))
    if session_id in state.retired_capture_session_ids:
        return "stale_capture_session"
    if session_id != state.capture_session_id:
        return "new_capture_session"
    if state.last_capture_seq is None:
        return None
    if capture_seq == state.last_capture_seq:
        if capture_digest != state.last_capture_digest:
            return FaultCode.ID_CONTENT_CONFLICT
        return "duplicate_capture"
    if capture_seq < state.last_capture_seq:
        return "capture_sequence_reversed"
    if capture_seq > state.last_capture_seq + 1:
        return "capture_sequence_gap"
    return None


def _retired_capture_sessions(state: ReducerState, new_session_id: str) -> tuple[str, ...]:
    candidates = (*state.retired_capture_session_ids, state.capture_session_id)
    return tuple(dict.fromkeys(item for item in candidates if item and item != new_session_id))


def _capture_reset_changes(state: ReducerState) -> dict[str, Any]:
    return {
        "input_generation": state.input_generation + 1,
        "input_digest": None,
        "last_confirmed_monotonic_ms": None,
        "practical_lane": LaneState(),
        "best_action_lane": LaneState(),
        "hold_reasons": (),
        "hold_started_source_ms": None,
        "hold_started_monotonic_ms": None,
        "hold_expired": False,
        "hold_timer_id": None,
        "hold_timer_deadline_monotonic_ms": None,
        "hold_timer_tick": 0,
    }


def _terminal_reset_changes() -> dict[str, Any]:
    return {
        "terminal_state": TerminalState.NONE,
        "terminal_winner": None,
        "terminal_evidence_kind": None,
        "terminal_result_code": None,
        "terminal_timer_id": None,
        "terminal_timer_deadline_monotonic_ms": None,
    }


def _observation_changes(state: ReducerState, event: ReducerEvent) -> dict[str, Any]:
    payload = event.payload
    session_id = str(payload["capture_session_id"])
    return {
        "capture_connected": True,
        "capture_session_id": session_id,
        "retired_capture_session_ids": _retired_capture_sessions(state, session_id),
        "last_capture_seq": int(payload["capture_seq"]),
        "last_capture_digest": str(payload["capture_digest"]),
        "source_available_frame": int(payload["source_available_frame"]),
        "source_available_ms": int(payload["source_available_ms"]),
        "source_captured_monotonic_ms": int(payload["captured_monotonic_ms"]),
        "p1_board_provenance": BoardProvenance(str(payload.get("p1_board_provenance", "unknown"))),
        "p2_board_provenance": BoardProvenance(str(payload.get("p2_board_provenance", "unknown"))),
        "recognition_status": RecognitionQualityStatus(str(payload.get("recognition_status", "untrusted"))),
        "recognition_reasons": tuple(RecognitionReason(str(item)) for item in payload.get("recognition_reasons", ())),
        "unresolved_physics": tuple(str(item) for item in payload.get("unresolved_physics", ())),
        "physical_prediction_used": bool(payload.get("physical_prediction_used", False)),
    }


def _reduce_observation(state: ReducerState, event: ReducerEvent) -> Reduction:
    observed_bundle = str(event.payload.get("asset_bundle_id", state.asset_bundle_id))
    if observed_bundle != state.asset_bundle_id:
        return _fault_reduction(state, event, FaultCode.ASSET_BUNDLE_CHANGED_MID_MATCH)
    problem = _capture_problem(state, event)
    if isinstance(problem, FaultCode):
        return _fault_reduction(state, event, problem)
    if problem in {"stale_capture_session", "duplicate_capture", "capture_sequence_reversed"}:
        return _reject_event(state, event, problem)
    capture_changed = problem == "new_capture_session"
    changes = _observation_changes(state, event)
    source_ms = int(changes["source_available_ms"])
    monotonic_ms = int(event.payload["observed_monotonic_ms"])
    if problem == "capture_sequence_gap":
        changes["recognition_reasons"] = (RecognitionReason.CAPTURE_GAP,)
        reduction = _enter_hold(
            state,
            event,
            HoldReason.RECOGNITION_UNRELIABLE,
            source_ms=source_ms,
            monotonic_ms=monotonic_ms,
            changes=changes,
        )
        audit = _intent(ReducerIntentKind.REJECT_EVENT, reason=problem)
        return Reduction(reduction.state, (audit, *reduction.intents), False)
    trusted = changes["recognition_status"] is RecognitionQualityStatus.TRUSTED
    stable = bool(event.payload.get("p1_stable")) and bool(event.payload.get("p2_stable"))
    if not trusted or not stable:
        reason = HoldReason.RECOGNITION_UNRELIABLE
        if capture_changed:
            changes.update(_capture_reset_changes(state))
        reduction = _enter_hold(
            state,
            event,
            reason,
            source_ms=source_ms,
            monotonic_ms=monotonic_ms,
            changes=changes,
            reset_existing=capture_changed,
        )
        if not capture_changed:
            return reduction
        invalidation = _intent(ReducerIntentKind.INVALIDATE_JOBS, reason="new_capture_session")
        return Reduction(reduction.state, (invalidation, *reduction.intents))
    return _reduce_trusted_observation(state, event, changes, capture_changed=capture_changed)


def _reduce_trusted_observation(
    state: ReducerState,
    event: ReducerEvent,
    changes: Mapping[str, Any],
    *,
    capture_changed: bool = False,
) -> Reduction:
    digest = str(event.payload["input_digest"])
    changes = dict(changes)
    if state.match_id is None and capture_changed:
        changes.update(_capture_reset_changes(state))
    changes["last_confirmed_monotonic_ms"] = int(event.payload["observed_monotonic_ms"])
    if state.match_id is None:
        updated = _advance(state, event, update_reason=UpdateReason.OBSERVATION_UPDATE, **changes)
        intents = [_intent(ReducerIntentKind.PUBLISH_SNAPSHOT)]
        if capture_changed:
            intents.insert(0, _intent(ReducerIntentKind.INVALIDATE_JOBS, reason="new_capture_session"))
        return Reduction(updated, tuple(intents))
    if digest == state.input_digest and not capture_changed:
        updated = _advance(state, event, update_reason=UpdateReason.OBSERVATION_UPDATE, **changes)
        return Reduction(updated, (_intent(ReducerIntentKind.PUBLISH_SNAPSHOT),))
    generation = state.input_generation + 1
    practical = _pending_lane(state, "practical", generation, digest) if state.mode is not EvaluationMode.BEST_ACTION else LaneState()
    best_action = _pending_lane(state, "best_action", generation, digest) if state.mode is not EvaluationMode.PRACTICAL else LaneState()
    updated = _advance(
        state,
        event,
        input_generation=generation,
        input_digest=digest,
        match_phase=MatchPhase.ACTIVE,
        display_state=DisplayState.AWAITING,
        update_reason=UpdateReason.OBSERVATION_UPDATE,
        practical_lane=practical,
        best_action_lane=best_action,
        hold_reasons=(),
        hold_started_source_ms=None,
        hold_started_monotonic_ms=None,
        hold_expired=False,
        hold_timer_id=None,
        hold_timer_deadline_monotonic_ms=None,
        hold_timer_tick=0,
        runtime_status=RuntimeStatus.HEALTHY,
        **changes,
    )
    invalidation_reason = "new_capture_session" if capture_changed else "new_generation"
    intents = (
        _intent(ReducerIntentKind.INVALIDATE_JOBS, reason=invalidation_reason),
        *_start_job_intents(updated),
    )
    return Reduction(updated, (*intents, _intent(ReducerIntentKind.PUBLISH_SNAPSHOT)))


def _boundary_config(state: ReducerState) -> tuple[EvaluationMode, RuntimeTier, str, str, Mapping[str, Any]]:
    pending = state.pending_config
    mode = EvaluationMode(str(pending.get("mode", state.mode.value)))
    tier = RuntimeTier(str(pending.get("tier", state.tier.value)))
    profile = str(pending.get("tier_profile_id", state.tier_profile_id))
    bundle_id = str(pending.get("asset_bundle_id", state.asset_bundle_id))
    assets = pending.get("assets", state.assets)
    return mode, tier, profile, bundle_id, assets


def _reduce_boundary(state: ReducerState, event: ReducerEvent) -> Reduction:
    mode, tier, profile, bundle_id, assets = _boundary_config(state)
    updated = _advance(
        state,
        event,
        match_id=str(event.payload["match_id"]),
        match_phase=MatchPhase.PRE_MATCH,
        display_state=DisplayState.HIDDEN,
        update_reason=UpdateReason.FORMAL_BOUNDARY,
        input_generation=state.input_generation + 1,
        input_digest=None,
        source_available_frame=None,
        source_available_ms=None,
        source_captured_monotonic_ms=None,
        last_confirmed_monotonic_ms=None,
        p1_board_provenance=BoardProvenance.UNKNOWN,
        p2_board_provenance=BoardProvenance.UNKNOWN,
        recognition_status=RecognitionQualityStatus.UNTRUSTED,
        recognition_reasons=(),
        unresolved_physics=(),
        physical_prediction_used=False,
        practical_lane=LaneState(),
        best_action_lane=LaneState(),
        hold_reasons=(),
        hold_started_source_ms=None,
        hold_started_monotonic_ms=None,
        hold_expired=False,
        hold_timer_id=None,
        hold_timer_deadline_monotonic_ms=None,
        hold_timer_tick=0,
        fault_codes=(),
        pending_config={},
        mode=mode,
        tier=tier,
        tier_profile_id=profile,
        asset_bundle_id=bundle_id,
        assets=assets,
        discarded_jobs_since_match_start=0,
        **_terminal_reset_changes(),
    )
    intents = (_intent(ReducerIntentKind.INVALIDATE_JOBS, reason="formal_boundary"), _intent(ReducerIntentKind.PUBLISH_SNAPSHOT))
    return Reduction(updated, intents)


def _reduce_intermission(state: ReducerState, event: ReducerEvent) -> Reduction:
    updated = _advance(
        state,
        event,
        match_phase=MatchPhase.INTERMISSION,
        display_state=DisplayState.HIDDEN,
        update_reason=UpdateReason.FORMAL_BOUNDARY,
        input_generation=state.input_generation + 1,
        input_digest=None,
        source_available_frame=None,
        source_available_ms=None,
        source_captured_monotonic_ms=None,
        last_confirmed_monotonic_ms=None,
        p1_board_provenance=BoardProvenance.UNKNOWN,
        p2_board_provenance=BoardProvenance.UNKNOWN,
        recognition_status=RecognitionQualityStatus.UNTRUSTED,
        recognition_reasons=(),
        unresolved_physics=(),
        physical_prediction_used=False,
        practical_lane=LaneState(),
        best_action_lane=LaneState(),
        hold_reasons=(),
        hold_started_source_ms=None,
        hold_started_monotonic_ms=None,
        hold_expired=False,
        hold_timer_id=None,
        hold_timer_deadline_monotonic_ms=None,
        hold_timer_tick=0,
        **_terminal_reset_changes(),
    )
    intents = (_intent(ReducerIntentKind.INVALIDATE_JOBS, reason="intermission"), _intent(ReducerIntentKind.PUBLISH_SNAPSHOT))
    return Reduction(updated, intents)


def _prediction_display_changes(
    state: ReducerState,
    resolved_display: DisplayState,
) -> tuple[dict[str, Any], tuple[ReducerIntent, ...]]:
    if state.display_state is not DisplayState.HOLD:
        return {
            "display_state": resolved_display,
            "update_reason": UpdateReason.PREDICTION_COMMITTED,
        }, ()
    reasons = _ordered_reasons(
        tuple(item for item in state.hold_reasons if item is not HoldReason.CALCULATION_PENDING)
    )
    if reasons:
        return {
            "display_state": DisplayState.HOLD,
            "update_reason": UpdateReason.HOLD_REASON_CHANGED,
            "hold_reasons": reasons,
        }, ()
    return {
        "display_state": resolved_display,
        "update_reason": UpdateReason.PREDICTION_COMMITTED,
        "hold_reasons": (),
        "hold_started_source_ms": None,
        "hold_started_monotonic_ms": None,
        "hold_expired": False,
        "hold_timer_id": None,
        "hold_timer_deadline_monotonic_ms": None,
        "hold_timer_tick": 0,
    }, (_intent(ReducerIntentKind.CANCEL_TIMER),)


def _reduce_prediction(state: ReducerState, event: ReducerEvent) -> Reduction:
    lane_name = str(event.payload.get("lane", ""))
    lane = state.practical_lane if lane_name == "practical" else state.best_action_lane
    identifiers = (lane.request_id, lane.input_generation, lane.input_digest)
    matches = (
        lane_name in {"practical", "best_action"}
        and lane.job_state is JobState.QUEUED
        and all(item is not None for item in identifiers)
        and event.payload.get("request_id") == lane.request_id
        and event.payload.get("input_generation") == lane.input_generation
        and event.payload.get("input_digest") == lane.input_digest
    )
    if not matches:
        consumed = _advance(state, event, discarded_jobs_since_match_start=state.discarded_jobs_since_match_start + 1)
        return Reduction(consumed, (_intent(ReducerIntentKind.REJECT_EVENT, reason="stale_prediction"),), False)
    ready = replace(
        lane,
        availability=EvaluationAvailability.AVAILABLE,
        job_state=JobState.RESULT_READY,
        completed_monotonic_ms=int(event.payload["completed_monotonic_ms"]),
        values=event.payload["evaluation"],
    )
    origin = str(ready.values.get("origin", PracticalOrigin.MODEL.value))
    display = DisplayState.PHYSICAL_PREDICTION if origin == PracticalOrigin.PHYSICAL_PREDICTION.value else DisplayState.LIVE
    lane_changes = {"practical_lane": ready} if lane_name == "practical" else {"best_action_lane": ready}
    display_changes, intents = _prediction_display_changes(state, display)
    updated = _advance(state, event, **display_changes, **lane_changes)
    return Reduction(updated, (*intents, _intent(ReducerIntentKind.PUBLISH_SNAPSHOT)))


def _expire_lane(lane: LaneState) -> LaneState:
    return LaneState(
        availability=EvaluationAvailability.PENDING,
        job_state=lane.job_state,
        request_id=lane.request_id,
        input_generation=lane.input_generation,
        input_digest=lane.input_digest,
    )


def _terminal_problem(state: ReducerState, event: ReducerEvent) -> FaultCode | str | None:
    payload = event.payload
    if state.match_phase is not MatchPhase.ACTIVE or state.match_id is None:
        return "terminal_phase_not_active"
    if payload.get("match_id") != state.match_id:
        return "terminal_match_mismatch"
    if payload.get("capture_session_id") != state.capture_session_id:
        return "terminal_capture_session_mismatch"
    if payload.get("asset_bundle_id", state.asset_bundle_id) != state.asset_bundle_id:
        return FaultCode.ASSET_BUNDLE_CHANGED_MID_MATCH
    kind = str(payload.get("evidence_kind", ""))
    if kind != TerminalEvidenceKind.VISUAL_RESULT_LOGO_BILATERAL_2X2.value:
        return "terminal_evidence_not_allowlisted"
    winner = str(payload.get("winner", ""))
    expected = "p1_win" if winner == "1P" else "p2_win" if winner == "2P" else ""
    if not expected or payload.get("result_code") != expected:
        return "terminal_direction_invalid"
    current_frame = state.source_available_frame
    current_ms = state.source_available_ms
    if current_frame is None or current_ms is None:
        return FaultCode.CUTOFF_VIOLATION
    if int(payload["source_available_frame"]) < current_frame:
        return FaultCode.CUTOFF_VIOLATION
    if int(payload["source_available_ms"]) < current_ms:
        return FaultCode.CUTOFF_VIOLATION
    return None


def _terminal_lane(
    generation: int, digest: str, winner: TerminalWinner, completed_ms: int,
) -> LaneState:
    p1_wins = winner is TerminalWinner.PLAYER_1
    return LaneState(
        availability=EvaluationAvailability.AVAILABLE,
        job_state=JobState.RESULT_READY,
        input_generation=generation,
        input_digest=digest,
        completed_monotonic_ms=completed_ms,
        values={
            "calculation_latency_ms": 0,
            "p1_win_probability": 1.0 if p1_wins else 0.0,
            "p2_win_probability": 0.0 if p1_wins else 1.0,
            "advantage_score": 100.0 if p1_wins else -100.0,
            "is_even": False,
            "origin": PracticalOrigin.OBSERVED.value,
            "calibration_id": None,
            "evaluated_positions": 1,
        },
    )


def _reduce_terminal_evidence(state: ReducerState, event: ReducerEvent) -> Reduction:
    problem = _terminal_problem(state, event)
    if isinstance(problem, FaultCode):
        return _fault_reduction(state, event, problem)
    if problem is not None:
        return _reject_event(state, event, problem)
    payload = event.payload
    winner = TerminalWinner(str(payload["winner"]))
    result_code = TerminalResultCode(str(payload["result_code"]))
    observed_ms = int(payload["observed_monotonic_ms"])
    generation = state.input_generation + 1
    timer_id = f"terminal:{state.session_id}:{state.match_id}:{generation}"
    updated = _advance(
        state, event, input_generation=generation, input_digest=event.content_digest,
        match_phase=MatchPhase.TERMINAL, display_state=DisplayState.TERMINAL_FACT,
        update_reason=UpdateReason.TERMINAL_CONFIRMED,
        source_available_frame=int(payload["source_available_frame"]),
        source_available_ms=int(payload["source_available_ms"]),
        source_captured_monotonic_ms=int(payload["captured_monotonic_ms"]),
        practical_lane=_terminal_lane(generation, event.content_digest, winner, observed_ms),
        best_action_lane=LaneState(), hold_reasons=(), hold_started_source_ms=None,
        hold_started_monotonic_ms=None, hold_expired=False, hold_timer_id=None,
        hold_timer_deadline_monotonic_ms=None, hold_timer_tick=0,
        terminal_state=TerminalState.CONFIRMED, terminal_winner=winner,
        terminal_evidence_kind=TerminalEvidenceKind.VISUAL_RESULT_LOGO_BILATERAL_2X2,
        terminal_result_code=result_code, terminal_timer_id=timer_id,
        terminal_timer_deadline_monotonic_ms=observed_ms + TERMINAL_MIN_DISPLAY_MS,
    )
    intents = _terminal_intents(state, updated)
    return Reduction(updated, intents)


def _terminal_intents(
    previous: ReducerState, updated: ReducerState,
) -> tuple[ReducerIntent, ...]:
    intents = [_intent(ReducerIntentKind.INVALIDATE_JOBS, reason="terminal_confirmed")]
    if previous.hold_timer_id is not None:
        intents.append(_intent(ReducerIntentKind.CANCEL_TIMER, timer_id=previous.hold_timer_id))
    intents.extend((
        _intent(
            ReducerIntentKind.SCHEDULE_TIMER,
            timer_id=updated.terminal_timer_id,
            deadline_monotonic_ms=updated.terminal_timer_deadline_monotonic_ms,
        ),
        _intent(ReducerIntentKind.PUBLISH_SNAPSHOT),
    ))
    return tuple(intents)


def _reduce_terminal_timer(state: ReducerState, event: ReducerEvent) -> Reduction:
    timer_id = str(event.payload.get("timer_id", ""))
    fired_ms = int(event.payload.get("fired_monotonic_ms", -1))
    deadline = state.terminal_timer_deadline_monotonic_ms
    valid = state.match_phase is MatchPhase.TERMINAL and timer_id == state.terminal_timer_id
    if not valid or deadline is None or fired_ms < deadline:
        return _reject_event(state, event, "stale_terminal_timer")
    updated = _advance(
        state, event, match_phase=MatchPhase.RESULT, display_state=DisplayState.RESULT,
        update_reason=UpdateReason.RESULT_TRANSITION,
        practical_lane=LaneState(), best_action_lane=LaneState(),
        terminal_timer_id=None, terminal_timer_deadline_monotonic_ms=None,
    )
    return Reduction(updated, (_intent(ReducerIntentKind.PUBLISH_SNAPSHOT),))


def _reduce_timer(state: ReducerState, event: ReducerEvent) -> Reduction:
    if state.match_phase in {MatchPhase.TERMINAL, MatchPhase.RESULT}:
        return _reduce_terminal_timer(state, event)
    timer_id = str(event.payload.get("timer_id", ""))
    fired_ms = int(event.payload.get("fired_monotonic_ms", -1))
    deadline = state.hold_timer_deadline_monotonic_ms
    valid = state.display_state is DisplayState.HOLD and timer_id == state.hold_timer_id
    if not valid or deadline is None or fired_ms < deadline:
        consumed = _advance(state, event, match_change=False)
        return Reduction(consumed, (_intent(ReducerIntentKind.REJECT_EVENT, reason="stale_timer"),), False)
    anchor = int(state.hold_started_monotonic_ms)
    expired = fired_ms - anchor >= HOLD_EXPIRY_MS
    tick = state.hold_timer_tick + 1
    next_id = _hold_timer_id(state, anchor, tick)
    changes: dict[str, Any] = {
        "hold_expired": expired or state.hold_expired,
        "hold_timer_tick": tick,
        "hold_timer_id": next_id,
        "hold_timer_deadline_monotonic_ms": anchor + HOLD_REPUBLISH_INTERVAL_MS * tick,
        "update_reason": UpdateReason.HOLD_EXPIRED if expired else UpdateReason.TIMER_ELAPSED,
    }
    if expired:
        if state.mode is not EvaluationMode.BEST_ACTION:
            changes["practical_lane"] = _expire_lane(state.practical_lane)
        if state.mode is not EvaluationMode.PRACTICAL:
            changes["best_action_lane"] = _expire_lane(state.best_action_lane)
    updated = _advance(state, event, **changes)
    schedule = _intent(ReducerIntentKind.SCHEDULE_TIMER, timer_id=next_id, deadline_monotonic_ms=changes["hold_timer_deadline_monotonic_ms"])
    return Reduction(updated, (_intent(ReducerIntentKind.PUBLISH_SNAPSHOT), schedule))


def _hold_recovery_display(state: ReducerState) -> DisplayState:
    practical_active = state.mode is not EvaluationMode.BEST_ACTION
    best_active = state.mode is not EvaluationMode.PRACTICAL
    practical_available = (
        practical_active
        and state.practical_lane.availability is EvaluationAvailability.AVAILABLE
    )
    best_available = (
        best_active
        and state.best_action_lane.availability is EvaluationAvailability.AVAILABLE
    )
    if not practical_available and not best_available:
        return DisplayState.AWAITING
    provenances = (state.p1_board_provenance, state.p2_board_provenance)
    physical_input = (
        state.physical_prediction_used
        and BoardProvenance.PHYSICS_PROJECTED in provenances
    )
    origin = state.practical_lane.values.get("origin") if practical_available else None
    if physical_input and origin == PracticalOrigin.PHYSICAL_PREDICTION.value:
        return DisplayState.PHYSICAL_PREDICTION
    normal_origin = not practical_available or origin in {
        PracticalOrigin.OBSERVED.value,
        PracticalOrigin.MODEL.value,
    }
    if provenances == (BoardProvenance.CONFIRMED, BoardProvenance.CONFIRMED) and normal_origin:
        return DisplayState.LIVE
    return DisplayState.AWAITING


def _hide_unsafe_available_lanes(state: ReducerState) -> dict[str, LaneState]:
    changes: dict[str, LaneState] = {}
    if state.practical_lane.availability is EvaluationAvailability.AVAILABLE:
        changes["practical_lane"] = LaneState()
    if state.best_action_lane.availability is EvaluationAvailability.AVAILABLE:
        changes["best_action_lane"] = LaneState()
    return changes


def _reduce_hold_reason(state: ReducerState, event: ReducerEvent, *, adding: bool) -> Reduction:
    reason = HoldReason(str(event.payload["reason"]))
    if adding:
        event_source = event.payload.get("source_available_ms")
        source_value = event_source if event_source is not None else state.source_available_ms
        if source_value is None:
            return _reject_event(state, event, "hold_source_time_unavailable")
        source_ms = int(source_value)
        monotonic_ms = int(event.payload["monotonic_ms"])
        return _enter_hold(
            state,
            event,
            reason,
            source_ms=source_ms,
            monotonic_ms=monotonic_ms,
            changes={"source_available_ms": source_ms},
        )
    if state.display_state is not DisplayState.HOLD or reason not in state.hold_reasons:
        return _reject_event(state, event, "hold_reason_not_active")
    reasons = _ordered_reasons(tuple(item for item in state.hold_reasons if item is not reason))
    if reasons:
        updated = _advance(state, event, hold_reasons=reasons, update_reason=UpdateReason.HOLD_REASON_CHANGED)
        return Reduction(updated, (_intent(ReducerIntentKind.PUBLISH_SNAPSHOT),))
    display = _hold_recovery_display(state)
    lane_changes = _hide_unsafe_available_lanes(state) if display is DisplayState.AWAITING else {}
    updated = _advance(
        state,
        event,
        display_state=display,
        update_reason=UpdateReason.HOLD_REASON_CHANGED,
        hold_reasons=(),
        hold_started_source_ms=None,
        hold_started_monotonic_ms=None,
        hold_expired=False,
        hold_timer_id=None,
        hold_timer_deadline_monotonic_ms=None,
        hold_timer_tick=0,
        **lane_changes,
    )
    return Reduction(updated, (_intent(ReducerIntentKind.CANCEL_TIMER), _intent(ReducerIntentKind.PUBLISH_SNAPSHOT)))


def _reduce_capture_status(state: ReducerState, event: ReducerEvent) -> Reduction:
    status = str(event.payload["status"])
    requested_session = str(event.payload.get("capture_session_id", ""))
    if status == "connected" and requested_session in state.retired_capture_session_ids:
        return _reject_event(state, event, "stale_capture_session")
    if status == "disconnected":
        updated = _advance(
            state,
            event,
            capture_connected=False,
            input_generation=state.input_generation + 1,
            input_digest=None,
            display_state=DisplayState.HIDDEN,
            update_reason=UpdateReason.WORKER_STATE_CHANGED,
            practical_lane=LaneState(),
            best_action_lane=LaneState(),
            hold_reasons=(),
            hold_started_source_ms=None,
            hold_started_monotonic_ms=None,
            hold_expired=False,
            hold_timer_id=None,
            hold_timer_deadline_monotonic_ms=None,
            hold_timer_tick=0,
            source_available_frame=None,
            source_available_ms=None,
            source_captured_monotonic_ms=None,
            last_confirmed_monotonic_ms=None,
            p1_board_provenance=BoardProvenance.UNKNOWN,
            p2_board_provenance=BoardProvenance.UNKNOWN,
            unresolved_physics=(),
            physical_prediction_used=False,
            recognition_status=RecognitionQualityStatus.UNTRUSTED,
            recognition_reasons=(RecognitionReason.CAPTURE_GAP,),
            runtime_status=RuntimeStatus.DEGRADED,
            **_terminal_reset_changes(),
        )
    else:
        updated = _connected_capture_state(state, event)
    intents = (_intent(ReducerIntentKind.INVALIDATE_JOBS, reason=f"capture_{status}"), _intent(ReducerIntentKind.PUBLISH_SNAPSHOT))
    return Reduction(updated, intents)


def _connected_capture_state(state: ReducerState, event: ReducerEvent) -> ReducerState:
    session_id = str(event.payload["capture_session_id"])
    changed = session_id != state.capture_session_id
    return _advance(
        state,
        event,
        capture_connected=True,
        capture_session_id=session_id,
        retired_capture_session_ids=(
            _retired_capture_sessions(state, session_id)
            if changed else state.retired_capture_session_ids
        ),
        last_capture_seq=None if changed else state.last_capture_seq,
        last_capture_digest=None if changed else state.last_capture_digest,
        input_generation=state.input_generation + int(changed),
        input_digest=None if changed else state.input_digest,
        source_available_frame=None if changed else state.source_available_frame,
        source_available_ms=None if changed else state.source_available_ms,
        source_captured_monotonic_ms=None if changed else state.source_captured_monotonic_ms,
        last_confirmed_monotonic_ms=None if changed else state.last_confirmed_monotonic_ms,
        p1_board_provenance=BoardProvenance.UNKNOWN if changed else state.p1_board_provenance,
        p2_board_provenance=BoardProvenance.UNKNOWN if changed else state.p2_board_provenance,
        recognition_status=RecognitionQualityStatus.UNTRUSTED if changed else state.recognition_status,
        recognition_reasons=() if changed else state.recognition_reasons,
        unresolved_physics=() if changed else state.unresolved_physics,
        physical_prediction_used=False if changed else state.physical_prediction_used,
        display_state=DisplayState.HIDDEN if changed else state.display_state,
        practical_lane=LaneState() if changed else state.practical_lane,
        best_action_lane=LaneState() if changed else state.best_action_lane,
        hold_reasons=() if changed else state.hold_reasons,
        hold_started_source_ms=None if changed else state.hold_started_source_ms,
        hold_started_monotonic_ms=None if changed else state.hold_started_monotonic_ms,
        hold_expired=False if changed else state.hold_expired,
        hold_timer_id=None if changed else state.hold_timer_id,
        hold_timer_deadline_monotonic_ms=None if changed else state.hold_timer_deadline_monotonic_ms,
        hold_timer_tick=0 if changed else state.hold_timer_tick,
        runtime_status=RuntimeStatus.STARTING if changed else state.runtime_status,
        update_reason=UpdateReason.WORKER_STATE_CHANGED,
        **(_terminal_reset_changes() if changed else {}),
    )


def _reduce_config(state: ReducerState, event: ReducerEvent) -> Reduction:
    updated = _advance(
        state,
        event,
        pending_config=event.payload,
        update_reason=UpdateReason.CONFIG_PENDING,
    )
    return Reduction(updated, (_intent(ReducerIntentKind.PUBLISH_SNAPSHOT),))


def _reduce_new_session(state: ReducerState, event: ReducerEvent) -> Reduction:
    bootstrap = Bootstrap(
        session_id=str(event.payload["session_id"]),
        mode=EvaluationMode(str(event.payload.get("mode", state.mode.value))),
        tier=RuntimeTier(str(event.payload.get("tier", state.tier.value))),
        tier_profile_id=str(event.payload.get("tier_profile_id", state.tier_profile_id)),
        asset_bundle_id=str(event.payload.get("asset_bundle_id", state.asset_bundle_id)),
        assets=event.payload.get("assets", state.assets),
        capture_session_id=event.payload.get("capture_session_id"),
    )
    fresh = initial_state(bootstrap)
    intents = (_intent(ReducerIntentKind.INVALIDATE_JOBS, reason="new_session"), _intent(ReducerIntentKind.PUBLISH_SNAPSHOT))
    return Reduction(fresh, intents)


def _event_envelope_valid(event: ReducerEvent) -> bool:
    sequence_valid = (
        isinstance(event.event_seq, int)
        and not isinstance(event.event_seq, bool)
        and event.event_seq >= 0
    )
    return (
        isinstance(event.kind, ReducerEventKind)
        and sequence_valid
        and isinstance(event.content_digest, str)
        and bool(event.content_digest)
        and isinstance(event.payload, Mapping)
    )


def _reduce_valid_event(state: ReducerState, event: ReducerEvent) -> Reduction:
    if event.kind is ReducerEventKind.NEW_SESSION:
        return _reduce_new_session(state, event)
    fault = _sequence_fault(state, event)
    if fault is not None:
        return _fault_reduction(state, event, fault)
    if state.match_phase in _LATCHED_PHASES and event.kind not in _LATCH_SAFE_EVENTS:
        return _reject_event(state, event, f"{state.match_phase.value}_latched")
    if event.kind in _J1_UNSUPPORTED_EVENTS:
        return _reject_event(state, event, "not_implemented_in_j1")
    handlers = {
        ReducerEventKind.OBSERVATION_AVAILABLE: _reduce_observation,
        ReducerEventKind.FORMAL_BOUNDARY: _reduce_boundary,
        ReducerEventKind.INTERMISSION: _reduce_intermission,
        ReducerEventKind.PREDICTION_COMPLETED: _reduce_prediction,
        ReducerEventKind.TERMINAL_EVIDENCE: _reduce_terminal_evidence,
        ReducerEventKind.CAPTURE_STATUS: _reduce_capture_status,
        ReducerEventKind.CONFIG_CHANGE_REQUESTED: _reduce_config,
        ReducerEventKind.TIMER_FIRED: _reduce_timer,
    }
    if event.kind is ReducerEventKind.HOLD_REASON_ADDED:
        return _reduce_hold_reason(state, event, adding=True)
    if event.kind is ReducerEventKind.HOLD_REASON_RESOLVED:
        return _reduce_hold_reason(state, event, adding=False)
    handler = handlers.get(event.kind)
    if handler is None:
        return _reject_event(state, event, "unsupported_event")
    return handler(state, event)


def reduce_event(state: ReducerState, event: ReducerEvent) -> Reduction:
    """domain不正を外へ漏らさずeventを一件だけ純粋に適用する。"""
    if not _event_envelope_valid(event):
        intent = _intent(ReducerIntentKind.REJECT_EVENT, reason="invalid_event_envelope")
        return Reduction(state, (intent,), False)
    try:
        return _reduce_valid_event(state, event)
    except (KeyError, ValueError, TypeError, OverflowError):
        return _fault_reduction(state, event, FaultCode.SCHEMA_INVALID)
