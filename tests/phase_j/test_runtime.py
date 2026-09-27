"""Phase J SingleWriter runtimeのS04/S05/S08統合契約試験。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from threading import Barrier, Event, Thread
from typing import Any

import pytest

pytest.importorskip("jsonschema")

import src.phase_j.runtime as runtime_module
from src.phase_j.contracts import EvaluationMode, OverlaySnapshot, WorkerHealth
from src.phase_j.display_projector import PublicationContext, project_snapshot
from src.phase_j.reducer import Bootstrap, ReducerEvent, ReducerEventKind, initial_state
from src.phase_j.runtime import (
    RuntimeCommitContext,
    RuntimeFlushContext,
    RuntimeInputContext,
    RuntimeIntentKind,
    RuntimeRejectReason,
    SingleWriterRuntime,
)
from src.phase_j.scheduler import (
    CommitRejectReason,
    EvaluationKind,
    JobRequest,
    JobResult,
    NumericGateState,
)
from src.phase_j.snapshot_hub import SnapshotHub
from src.phase_j.validator import validate_snapshot

UTC = "2026-09-04T00:00:00Z"
DEADLINE_NS = 10_000_000_000
FIRST_RESULT_NS = 1_000_000_000
STRESS_ROUNDS = 20


def _event(kind: ReducerEventKind, seq: int, **payload: Any) -> ReducerEvent:
    return ReducerEvent(kind, seq, f"event:{seq}:{kind.value}", payload)


def _observation(
    seq: int,
    capture_seq: int,
    digest: str,
    capture_session_id: str = "capture-1",
) -> ReducerEvent:
    return _event(
        ReducerEventKind.OBSERVATION_AVAILABLE,
        seq,
        capture_session_id=capture_session_id,
        capture_seq=capture_seq,
        capture_digest=f"capture-{capture_seq}",
        source_available_frame=capture_seq * 10,
        source_available_ms=5_000 + capture_seq,
        captured_monotonic_ms=5_000,
        observed_monotonic_ms=5_010,
        recognition_status="trusted",
        recognition_reasons=[],
        p1_stable=True,
        p2_stable=True,
        p1_board_provenance="confirmed",
        p2_board_provenance="confirmed",
        unresolved_physics=[],
        physical_prediction_used=False,
        input_digest=digest,
    )


def _terminal_event(seq: int, winner: str = "1P") -> ReducerEvent:
    result_code = "p1_win" if winner == "1P" else "p2_win"
    return _event(
        ReducerEventKind.TERMINAL_EVIDENCE, seq,
        match_id="match-1", capture_session_id="capture-1",
        asset_bundle_id="unloaded", source_available_frame=11,
        source_available_ms=5_011, captured_monotonic_ms=5_090,
        observed_monotonic_ms=5_100,
        evidence_kind="visual_result_logo_bilateral_2x2",
        winner=winner, result_code=result_code,
    )


def _input_context(
    *,
    now_ns: int = FIRST_RESULT_NS,
    publish_ms: int = 5_020,
    with_job: bool = False,
) -> RuntimeInputContext:
    return RuntimeInputContext(
        UTC,
        publish_ms,
        now_ns,
        causal_cutoff_digest="cutoff-1" if with_job else None,
        deadline_monotonic_ns=DEADLINE_NS if with_job else None,
        input_snapshot={"board": [[1, 2]], "next": [3, 4]} if with_job else None,
    )


def _commit_context(
    seq: int,
    *,
    now_ns: int = FIRST_RESULT_NS,
    publish_ms: int = 5_100,
) -> RuntimeCommitContext:
    return RuntimeCommitContext(
        seq, f"result:{seq}", 5_080, UTC, publish_ms, now_ns,
    )


def _runtime(
    mode: EvaluationMode = EvaluationMode.PRACTICAL,
    hub_type: type[SnapshotHub] = SnapshotHub,
    session_id: str = "session-1",
    capture_session_id: str = "capture-1",
) -> tuple[SingleWriterRuntime, SnapshotHub]:
    state = initial_state(
        Bootstrap(session_id=session_id, capture_session_id=capture_session_id, mode=mode)
    )
    initial = project_snapshot(state, PublicationContext(0, UTC, 0))
    hub = hub_type(initial)
    return SingleWriterRuntime(state, hub), hub


def _start_jobs(
    runtime: SingleWriterRuntime,
    *,
    boundary_seq: int = 1,
    observation_seq: int = 2,
    capture_seq: int = 1,
    digest: str = "input-1",
    now_ns: int = FIRST_RESULT_NS,
    capture_session_id: str = "capture-1",
) -> dict[EvaluationKind, JobRequest]:
    boundary = _event(ReducerEventKind.FORMAL_BOUNDARY, boundary_seq, match_id="match-1")
    runtime.apply_event(boundary, _input_context(now_ns=now_ns))
    step = runtime.apply_event(
        _observation(observation_seq, capture_seq, digest, capture_session_id),
        _input_context(now_ns=now_ns, with_job=True),
    )
    requests = [intent.request for intent in step.intents if intent.kind is RuntimeIntentKind.START_JOB]
    return {request.token.evaluation_kind: request for request in requests if request is not None}


def _next_jobs(
    runtime: SingleWriterRuntime,
    seq: int,
    capture_seq: int,
    digest: str,
    now_ns: int,
) -> dict[EvaluationKind, JobRequest]:
    step = runtime.apply_event(
        _observation(seq, capture_seq, digest),
        _input_context(now_ns=now_ns, publish_ms=5_100 + capture_seq, with_job=True),
    )
    requests = [intent.request for intent in step.intents if intent.kind is RuntimeIntentKind.START_JOB]
    return {request.token.evaluation_kind: request for request in requests if request is not None}


def _values(kind: EvaluationKind) -> dict[str, Any]:
    if kind is EvaluationKind.PRACTICAL:
        return {
            "calculation_latency_ms": 80,
            "p1_win_probability": 0.6,
            "p2_win_probability": 0.4,
            "advantage_score": 20,
            "is_even": False,
            "origin": "model",
            "calibration_id": "calibration-v1",
            "evaluated_positions": 1,
        }
    return {
        "calculation_latency_ms": 90,
        "p1_position_value": 12,
        "p1_position_value_low": 5,
        "p1_position_value_high": 20,
        "aggregation_profile_id": "aggregate-v1",
        "search_profile_id": "search-v1",
        "searched_depth": 4,
        "searched_nodes": 120,
    }


def _result(request: JobRequest) -> JobResult:
    return JobResult(
        request.token.evaluation_kind,
        request.request_id,
        request.token,
        request.input_digest,
        _values(request.token.evaluation_kind),
    )


def test_event_pipeline_builds_immutable_job_and_publishes_awaiting() -> None:
    runtime, hub = _runtime()
    jobs = _start_jobs(runtime)
    request = jobs[EvaluationKind.PRACTICAL]
    assert request.token.capture_seq == 1
    assert request.token.causal_cutoff_digest == "cutoff-1"
    assert hub.latest.display["visibility"] == "hidden"
    assert validate_snapshot(hub.latest).is_valid
    with pytest.raises(TypeError):
        request.input_snapshot["board"] = []  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        request.request_id = "changed"  # type: ignore[misc]


def test_terminal_event_invalidates_runtime_jobs_and_timer_publishes_result() -> None:
    runtime, hub = _runtime()
    jobs = _start_jobs(runtime)
    assert jobs[EvaluationKind.PRACTICAL] is not None

    terminal = runtime.apply_event(
        _terminal_event(3, "2P"), _input_context(publish_ms=5_100),
    )
    assert terminal.accepted
    assert terminal.snapshot.display["status"] == "terminal_fact"
    assert terminal.snapshot.terminal["winner"] == "2P"
    assert terminal.snapshot.evaluations["practical"]["p1_win_probability"] == 0.0
    assert runtime.scheduler_state.practical.running is None
    assert any(intent.kind is RuntimeIntentKind.SCHEDULE_TIMER for intent in terminal.intents)
    assert validate_snapshot(hub.latest).is_valid

    timer_id = runtime.reducer_state.terminal_timer_id
    timer = _event(
        ReducerEventKind.TIMER_FIRED, 4,
        timer_id=timer_id, fired_monotonic_ms=6_100,
    )
    result = runtime.apply_event(timer, _input_context(publish_ms=6_100))
    assert result.accepted
    assert result.snapshot.display["status"] == "result"
    assert result.snapshot.evaluations["practical"]["availability"] == "unavailable"
    assert result.snapshot.terminal["result_code"] == "p2_win"
    assert validate_snapshot(result.snapshot).is_valid


def test_missing_job_context_never_starts_worker_and_is_fail_closed() -> None:
    runtime, hub = _runtime()
    boundary = _event(ReducerEventKind.FORMAL_BOUNDARY, 1, match_id="match-1")
    runtime.apply_event(boundary, _input_context())
    event = _observation(2, 1, "input-1")
    before = runtime.reducer_state
    step = runtime.apply_event(event, _input_context())
    assert not step.accepted
    assert step.reject_reason is RuntimeRejectReason.MISSING_JOB_CONTEXT
    assert not any(intent.kind is RuntimeIntentKind.START_JOB for intent in step.intents)
    assert runtime.reducer_state == before
    assert step.fail_closed
    assert hub.latest.display["status"] == "integrity_fault"
    assert validate_snapshot(hub.latest).is_valid
    retried = runtime.apply_event(event, _input_context(with_job=True))
    assert retried.accepted
    assert runtime.reducer_state.last_event_seq == 2
    assert any(intent.kind is RuntimeIntentKind.START_JOB for intent in retried.intents)


def test_elapsed_deadline_rejects_job_before_start() -> None:
    runtime, _ = _runtime()
    boundary = _event(ReducerEventKind.FORMAL_BOUNDARY, 1, match_id="match-1")
    runtime.apply_event(boundary, _input_context())
    context = replace(
        _input_context(now_ns=DEADLINE_NS + 1, with_job=True),
        deadline_monotonic_ns=DEADLINE_NS,
    )
    event = _observation(2, 1, "input-1")
    before = runtime.reducer_state
    step = runtime.apply_event(event, context)
    assert step.reject_reason is RuntimeRejectReason.JOB_DEADLINE_EXCEEDED
    assert runtime.reducer_state == before
    assert runtime.scheduler_state.practical.running is None
    retried = runtime.apply_event(event, _input_context(with_job=True))
    assert retried.accepted


def test_context_failure_latch_hides_unrelated_publish_until_new_job_starts() -> None:
    runtime, hub = _runtime()
    first = _start_jobs(runtime)[EvaluationKind.PRACTICAL]
    runtime.complete_job(_result(first), _commit_context(3))
    assert runtime.reducer_state.display_state.value == "live"
    failed_event = _observation(4, 2, "input-2")
    failed = runtime.apply_event(failed_event, _input_context())
    assert failed.fail_closed
    assert runtime.reducer_state.display_state.value == "live"
    assert runtime.fail_closed_latch is RuntimeRejectReason.MISSING_JOB_CONTEXT

    unrelated = _event(ReducerEventKind.CONFIG_CHANGE_REQUESTED, 4, mode="practical")
    still_hidden = runtime.apply_event(unrelated, _input_context(publish_ms=5_210))
    assert still_hidden.fail_closed
    assert hub.latest.display["visibility"] == "hidden"
    recovered = runtime.apply_event(
        _observation(5, 2, "input-2"),
        _input_context(publish_ms=5_220, with_job=True),
    )
    assert recovered.accepted
    assert runtime.fail_closed_latch is None
    assert recovered.snapshot.display["status"] == "waiting"
    assert any(intent.kind is RuntimeIntentKind.START_JOB for intent in recovered.intents)


def test_start_context_failure_rolls_back_reducer_and_invalidates_old_work() -> None:
    runtime, hub = _runtime()
    first = _start_jobs(runtime)[EvaluationKind.PRACTICAL]
    gate = NumericGateState(777, _result(first))
    runtime = SingleWriterRuntime(
        runtime.reducer_state,
        hub,
        scheduler_state=runtime.scheduler_state,
        numeric_gate_state=gate,
    )
    event = _observation(3, 2, "input-2")
    before_reducer = runtime.reducer_state
    failed = runtime.apply_event(event, _input_context())
    assert runtime.reducer_state == before_reducer
    assert runtime.scheduler_state.practical.running is None
    assert runtime.scheduler_state.practical.waiting is None
    assert runtime.numeric_gate_state.last_published_monotonic_ns == 777
    assert runtime.numeric_gate_state.pending_candidate is None
    assert [intent.kind for intent in failed.intents] == [
        RuntimeIntentKind.INVALIDATE_JOBS,
        RuntimeIntentKind.CANCEL_JOB,
        RuntimeIntentKind.REJECT_EVENT,
    ]
    assert failed.snapshot.identity["reducer_revision"] == before_reducer.reducer_revision
    assert failed.fail_closed
    hidden_seq = failed.snapshot.identity["stream_seq"]
    old_result = runtime.complete_job(_result(first), _commit_context(3))
    assert old_result.reject_reason is CommitRejectReason.INVALIDATED_JOB
    flush = RuntimeFlushContext(3, "flush:3", UTC, 5_100, DEADLINE_NS)
    assert runtime.flush_practical(flush).publish_result is None
    assert hub.latest.identity["stream_seq"] == hidden_seq
    assert hub.latest.display["visibility"] == "hidden"
    retried = runtime.apply_event(event, _input_context(with_job=True))
    assert retried.accepted
    assert runtime.scheduler_state.practical.running != first
    assert retried.snapshot.display["status"] == "waiting"


def test_s04_boundary_invalidates_old_result_before_hub_publish() -> None:
    runtime, hub = _runtime()
    request = _start_jobs(runtime)[EvaluationKind.PRACTICAL]
    boundary = _event(ReducerEventKind.FORMAL_BOUNDARY, 3, match_id="match-2")
    boundary_step = runtime.apply_event(boundary, _input_context(publish_ms=5_200))
    stream_seq = boundary_step.snapshot.identity["stream_seq"]
    stale = runtime.complete_job(_result(request), _commit_context(4))
    assert not stale.accepted
    assert stale.reject_reason is CommitRejectReason.INVALIDATED_JOB
    assert hub.latest.identity["stream_seq"] == stream_seq
    assert hub.latest.display["visibility"] == "hidden"


def test_s05_current_token_mismatch_is_rejected_without_publish() -> None:
    runtime, hub = _runtime()
    request = _start_jobs(runtime)[EvaluationKind.PRACTICAL]
    mismatched = replace(request.token, causal_cutoff_digest="wrong-cutoff")
    result = replace(_result(request), token=mismatched)
    before = hub.latest.identity["stream_seq"]
    step = runtime.complete_job(result, _commit_context(3))
    assert step.reject_reason is CommitRejectReason.CAUSAL_CUTOFF_MISMATCH
    assert hub.latest.identity["stream_seq"] == before


def test_prediction_event_cannot_bypass_commit_gate() -> None:
    runtime, hub = _runtime()
    request = _start_jobs(runtime)[EvaluationKind.PRACTICAL]
    direct = _event(
        ReducerEventKind.PREDICTION_COMPLETED,
        3,
        lane=EvaluationKind.PRACTICAL.value,
        request_id=request.request_id,
        input_generation=request.token.input_generation,
        input_digest=request.input_digest,
        completed_monotonic_ms=5_100,
        evaluation=_values(EvaluationKind.PRACTICAL),
    )
    step = runtime.apply_event(direct, _input_context(publish_ms=5_100))
    assert not step.accepted
    assert step.reject_reason is RuntimeRejectReason.RESULT_REQUIRES_COMMIT_GATE
    assert runtime.reducer_state.practical_lane.availability.value == "pending"
    assert runtime.scheduler_state.practical.running is None
    assert hub.latest.display["visibility"] == "hidden"
    assert step.fail_closed


def test_nonhealthy_worker_result_is_rejected_before_scheduler_commit() -> None:
    runtime, hub = _runtime()
    request = _start_jobs(runtime)[EvaluationKind.PRACTICAL]
    scheduler = runtime.scheduler_state
    degraded = replace(
        scheduler,
        practical=replace(scheduler.practical, worker_health=WorkerHealth.DEGRADED),
    )
    guarded = SingleWriterRuntime(runtime.reducer_state, hub, scheduler_state=degraded)
    before = hub.latest.identity["stream_seq"]
    step = guarded.complete_job(_result(request), _commit_context(3))
    assert step.reject_reason is CommitRejectReason.WORKER_UNHEALTHY
    assert hub.latest.identity["stream_seq"] == before
    assert guarded.scheduler_state.practical.running is None


def test_practical_is_2hz_gated_and_flush_requires_external_timer() -> None:
    runtime, hub = _runtime()
    first = _start_jobs(runtime)[EvaluationKind.PRACTICAL]
    runtime.complete_job(_result(first), _commit_context(3, now_ns=FIRST_RESULT_NS))
    second = _next_jobs(runtime, 4, 2, "input-2", FIRST_RESULT_NS + 100_000_000)[EvaluationKind.PRACTICAL]
    before = hub.latest.identity["stream_seq"]
    buffered = runtime.complete_job(
        _result(second), _commit_context(5, now_ns=FIRST_RESULT_NS + 200_000_000),
    )
    assert buffered.accepted and buffered.publish_result is None
    assert runtime.numeric_gate_state.pending_candidate is not None
    early = RuntimeFlushContext(5, "flush:5", UTC, 5_200, FIRST_RESULT_NS + 400_000_000)
    assert runtime.flush_practical(early).publish_result is None
    due = replace(early, now_monotonic_ns=FIRST_RESULT_NS + 500_000_000)
    flushed = runtime.flush_practical(due)
    assert flushed.publish_result is not None
    assert hub.latest.identity["stream_seq"] == before + 1
    assert hub.latest.evaluations["practical"]["input_digest"] == "input-2"


def test_best_action_bypasses_gate_without_losing_pending_practical() -> None:
    runtime, hub = _runtime(EvaluationMode.BOTH)
    first = _start_jobs(runtime)
    runtime.complete_job(_result(first[EvaluationKind.PRACTICAL]), _commit_context(3))
    second = _next_jobs(runtime, 4, 2, "input-2", FIRST_RESULT_NS + 100_000_000)
    runtime.complete_job(
        _result(second[EvaluationKind.PRACTICAL]),
        _commit_context(5, now_ns=FIRST_RESULT_NS + 200_000_000),
    )
    pending = runtime.numeric_gate_state.pending_candidate
    best = runtime.complete_job(
        _result(second[EvaluationKind.BEST_ACTION]),
        _commit_context(5, now_ns=FIRST_RESULT_NS + 210_000_000),
    )
    assert best.publish_result is not None
    assert runtime.numeric_gate_state.pending_candidate is pending
    assert hub.latest.evaluations["practical"]["availability"] == "pending"
    assert hub.latest.evaluations["best_action"]["availability"] == "available"
    assert validate_snapshot(hub.latest).is_valid


def test_s08_safety_event_publishes_immediately_while_practical_is_pending() -> None:
    runtime, hub = _runtime()
    first = _start_jobs(runtime)[EvaluationKind.PRACTICAL]
    runtime.complete_job(_result(first), _commit_context(3))
    second = _next_jobs(runtime, 4, 2, "input-2", FIRST_RESULT_NS + 100_000_000)[EvaluationKind.PRACTICAL]
    runtime.complete_job(
        _result(second), _commit_context(5, now_ns=FIRST_RESULT_NS + 200_000_000),
    )
    before = hub.latest.identity["stream_seq"]
    hold = _event(
        ReducerEventKind.HOLD_REASON_ADDED,
        5,
        reason="recognition_unreliable",
        source_available_ms=5_002,
        monotonic_ms=5_200,
    )
    step = runtime.apply_event(hold, _input_context(publish_ms=5_200))
    assert step.publish_result is not None
    assert hub.latest.identity["stream_seq"] == before + 1
    assert hub.latest.display["status"] == "hold"
    assert runtime.numeric_gate_state.pending_candidate is not None


class _FlakyHub(SnapshotHub):
    fail_next = False

    def publish(self, candidate: OverlaySnapshot) -> Any:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("synthetic hub failure")
        return super().publish(candidate)


def test_hub_exception_retries_with_explicit_fail_closed_snapshot() -> None:
    runtime, hub = _runtime(hub_type=_FlakyHub)
    request = _start_jobs(runtime)[EvaluationKind.PRACTICAL]
    runtime.complete_job(_result(request), _commit_context(3))
    assert isinstance(hub, _FlakyHub)
    hub.fail_next = True
    change = _event(ReducerEventKind.CONFIG_CHANGE_REQUESTED, 4, mode="practical")
    step = runtime.apply_event(change, _input_context(publish_ms=5_200))
    assert step.fail_closed
    assert hub.latest.display["status"] == "integrity_fault"
    assert validate_snapshot(hub.latest).is_valid


def test_new_session_requires_new_runtime_and_old_runtime_stays_fail_closed() -> None:
    runtime, hub = _runtime()
    request = _start_jobs(runtime)[EvaluationKind.PRACTICAL]
    runtime.complete_job(_result(request), _commit_context(3))
    old_identity = dict(hub.latest.identity)
    old_state = runtime.reducer_state
    new_session = _event(
        ReducerEventKind.NEW_SESSION,
        4,
        session_id="session-2",
        capture_session_id="capture-2",
    )
    step = runtime.apply_event(new_session, _input_context(publish_ms=5_200))
    identity = dict(hub.latest.identity)
    assert step.fail_closed
    assert not step.accepted
    assert step.reject_reason is RuntimeRejectReason.SESSION_ROTATION_REQUIRED
    assert runtime.reducer_state == old_state
    assert runtime.fail_closed_latch is RuntimeRejectReason.SESSION_ROTATION_REQUIRED
    assert identity["session_id"] == old_identity["session_id"]
    assert identity["reducer_revision"] == old_identity["reducer_revision"]
    assert identity["stream_seq"] == old_identity["stream_seq"] + 1
    assert hub.latest.display["status"] == "integrity_fault"
    assert validate_snapshot(hub.latest).is_valid

    ignored = _event(ReducerEventKind.CONFIG_CHANGE_REQUESTED, 4, mode="practical")
    blocked = runtime.apply_event(ignored, _input_context(publish_ms=5_210))
    assert blocked.reject_reason is RuntimeRejectReason.SESSION_ROTATION_REQUIRED
    assert runtime.reducer_state == old_state
    assert hub.latest.display["visibility"] == "hidden"

    replacement, replacement_hub = _runtime(
        session_id="session-2",
        capture_session_id="capture-2",
    )
    replacement_job = _start_jobs(
        replacement,
        capture_session_id="capture-2",
    )[EvaluationKind.PRACTICAL]
    committed = replacement.complete_job(_result(replacement_job), _commit_context(3))
    assert committed.accepted
    assert replacement_hub.latest.identity["session_id"] == "session-2"
    assert replacement_hub.latest.display["status"] == "live"
    assert validate_snapshot(replacement_hub.latest).is_valid


@pytest.mark.parametrize("mismatch", ["revision", "identity"])
def test_hub_identity_rejections_always_replace_old_live_with_fail_closed(
    mismatch: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, hub = _runtime()
    request = _start_jobs(runtime)[EvaluationKind.PRACTICAL]
    runtime.complete_job(_result(request), _commit_context(3))
    old_identity = dict(hub.latest.identity)
    original = runtime_module.project_snapshot

    def mismatched_projector(*args: Any, **kwargs: Any) -> OverlaySnapshot:
        payload = original(*args, **kwargs).to_mapping()
        payload["identity"]["reducer_revision"] = old_identity["reducer_revision"]
        if mismatch == "identity":
            payload["identity"]["stream_seq"] = old_identity["stream_seq"]
        else:
            payload["identity"]["reducer_revision"] -= 1
        return OverlaySnapshot.from_mapping(payload)

    monkeypatch.setattr(runtime_module, "project_snapshot", mismatched_projector)
    change = _event(ReducerEventKind.CONFIG_CHANGE_REQUESTED, 4, mode="practical")
    step = runtime.apply_event(change, _input_context(publish_ms=5_200))
    assert step.fail_closed
    assert hub.latest.display["visibility"] == "hidden"
    assert hub.latest.identity["stream_seq"] == old_identity["stream_seq"] + 1
    assert validate_snapshot(hub.latest).is_valid


def test_invalid_projected_snapshot_never_leaves_old_live_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, hub = _runtime()
    request = _start_jobs(runtime)[EvaluationKind.PRACTICAL]
    runtime.complete_job(_result(request), _commit_context(3))
    original = runtime_module.project_snapshot

    def invalid_projector(*args: Any, **kwargs: Any) -> OverlaySnapshot:
        payload = original(*args, **kwargs).to_mapping()
        payload["evaluations"]["practical"]["p2_win_probability"] = 0.5
        return OverlaySnapshot.from_mapping(payload)

    monkeypatch.setattr(runtime_module, "project_snapshot", invalid_projector)
    change = _event(ReducerEventKind.CONFIG_CHANGE_REQUESTED, 4, mode="practical")
    step = runtime.apply_event(change, _input_context(publish_ms=5_200))
    assert step.fail_closed and step.publish_result is not None
    assert hub.latest.display["visibility"] == "hidden"
    assert validate_snapshot(hub.latest).is_valid


class _BlockingHub(SnapshotHub):
    def __init__(self, initial: OverlaySnapshot) -> None:
        super().__init__(initial)
        self.block_next = False
        self.entered = Event()
        self.release = Event()
        self.history: list[str] = []

    def publish(self, candidate: OverlaySnapshot) -> Any:
        if self.block_next:
            self.block_next = False
            self.entered.set()
            assert self.release.wait(2)
        result = super().publish(candidate)
        self.history.append(str(result.snapshot.display["status"]))
        return result


def test_commit_publish_critical_section_cannot_be_interrupted_by_boundary() -> None:
    runtime, hub = _runtime(hub_type=_BlockingHub)
    request = _start_jobs(runtime)[EvaluationKind.PRACTICAL]
    assert isinstance(hub, _BlockingHub)
    hub.history.clear()
    hub.block_next = True
    boundary_done = Event()
    result_thread = Thread(target=lambda: runtime.complete_job(_result(request), _commit_context(3)))
    boundary = _event(ReducerEventKind.FORMAL_BOUNDARY, 4, match_id="match-2")
    boundary_thread = Thread(
        target=lambda: (runtime.apply_event(boundary, _input_context(publish_ms=5_200)), boundary_done.set())
    )
    result_thread.start()
    assert hub.entered.wait(2)
    boundary_thread.start()
    assert not boundary_done.wait(0.05)
    hub.release.set()
    result_thread.join(2)
    boundary_thread.join(2)
    assert boundary_done.is_set()
    assert hub.history == ["live", "waiting"]
    assert hub.latest.display["visibility"] == "hidden"


def test_same_injected_inputs_produce_identical_runtime_outputs() -> None:
    left, _ = _runtime()
    right, _ = _runtime()
    left_job = _start_jobs(left)[EvaluationKind.PRACTICAL]
    right_job = _start_jobs(right)[EvaluationKind.PRACTICAL]
    left_step = left.complete_job(_result(left_job), _commit_context(3))
    right_step = right.complete_job(_result(right_job), _commit_context(3))
    assert left_step.reducer_state == right_step.reducer_state
    assert left_step.scheduler_state == right_step.scheduler_state
    assert left_step.snapshot.serialize() == right_step.snapshot.serialize()
    assert left_step.intents == right_step.intents


def test_two_thread_stress_never_exposes_old_result_after_boundary() -> None:
    for _ in range(STRESS_ROUNDS):
        runtime, hub = _runtime()
        request = _start_jobs(runtime)[EvaluationKind.PRACTICAL]
        barrier = Barrier(3)
        errors: list[BaseException] = []

        def run(action: Any) -> None:
            try:
                barrier.wait()
                action()
            except BaseException as exc:  # thread内例外をmainで必ず検査する。
                errors.append(exc)

        result_thread = Thread(
            target=run,
            args=(lambda: runtime.complete_job(_result(request), _commit_context(3)),),
        )
        boundary = _event(ReducerEventKind.FORMAL_BOUNDARY, 4, match_id="match-2")
        boundary_thread = Thread(
            target=run,
            args=(lambda: runtime.apply_event(boundary, _input_context(publish_ms=5_200)),),
        )
        result_thread.start()
        boundary_thread.start()
        barrier.wait()
        result_thread.join(2)
        boundary_thread.join(2)
        assert not errors
        assert hub.latest.display["visibility"] == "hidden"
        assert validate_snapshot(hub.latest).is_valid
