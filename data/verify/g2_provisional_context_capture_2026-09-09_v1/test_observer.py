"""人工producerと実frozen updateを区別してcontext捕捉を検査する。"""
from __future__ import annotations
from contextlib import ExitStack
import copy
from dataclasses import dataclass, replace
from enum import Enum
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "g2_generation_publication_split_2026-09-09_v1"
sys.path.insert(0, str(PRIOR))
FRAME = 35850


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


O = load("_provisional_context_observer_test", ROOT / "observer.py")
H = load("_provisional_context_pb_serializer", O.PB_SOURCE)
S = load("_provisional_context_scope_source", O.SCOPE_SOURCE)
K = load("_provisional_context_connection_source", O.CONNECTION)


class State(Enum):
    STABLE = "stable"
    CHAIN = "chain"


class PB:
    def __init__(self) -> None:
        self.cells = [[{0: 1.0} for _ in range(6)] for _ in range(13)]

    def iter_cells(self) -> Any:
        for row in range(13):
            for col in range(6):
                yield row, col, SimpleNamespace(probs=self.cells[row][col])


@dataclass
class Generation:
    side: str
    reset_epoch: int = 0
    action_revision: int | None = 1


@dataclass
class Side:
    side: str
    state: Any
    confirmed_board: Any
    cnn_board: Any
    inferred_board: Any
    prob_board: Any
    next_pair: Any = (1, 2)
    dnext_pair: Any = None
    board_provenance: str = "observed"
    board_none_reason: str | None = None


def board_value(value: Any) -> Any:
    return None if value is None else {"grid": copy.deepcopy(value)}


def make_side(side: str) -> Side:
    grid = [[0] * 6 for _ in range(13)]
    return Side(side, State.STABLE, grid, copy.deepcopy(grid), copy.deepcopy(grid), PB())


def prior_evidence(state: dict[str, Any], history: Any, value: Side) -> None:
    pb, sm, connection = (state[k] for k in ("hidden_probability_observer", "current_scope_sink", "provisional_current_connection"))
    if (history.frame, value.side) not in pb.expected:
        return
    scope = {"source_id": pb.source_id, "run_id": pb.run_id, "frame_idx": history.frame,
             "time_sec": history.time_sec, "side": value.side}
    pb.rows.append(scope | {"probability": H.probability_value(value.prob_board, PB),
        "confirmed": board_value(value.confirmed_board), "state": "STABLE"})
    sm.rows.append(scope | {"kind": "current_entry_sm_return", "context_after": {"state": "stable"}})
    connection.next_rows[(history.frame, value.side)] = scope | {"kind": "next_enqueue_live_decision", "quiet": True}


def fake_runtime(output: Path) -> tuple[Any, ...]:
    history = SimpleNamespace(frame=None, time_sec=None)
    pb = H.Observer(output, "artificial-source", "artificial-run", PB, board_value,
                    [(FRAME, side) for side in O.SIDES])
    state = {"output": output, "hidden_probability_observer": pb, "sink": SimpleNamespace(epoch=lambda row: 1)}
    state["provisional_current_connection"] = K.Connection(history, state)
    state["current_scope_sink"] = S.Sink(state, history)
    generations = {side: Generation(side) for side in O.SIDES}
    tracker = SimpleNamespace(_pipeline=None, _frame=None, _time=None, generation=generations.__getitem__)
    state["tracker"] = tracker
    def isolate(side: str, result: Side) -> Side:
        if (history.frame, side) in pb.expected:
            state["provisional_current_connection"].rows.append({"frame": history.frame, "side": side,
                "current_stage": "before_confirmed_publication_hold", "current_candidate": None})
        if pipe.held and side == "2P":
            return replace(result, confirmed_board=None, prob_board=None,
                           board_none_reason="discarded_candidate_current_unverified")
        return result
    state["pending_rec"] = SimpleNamespace(isolate_side_result=isolate)
    class Pipeline:
        def update(self, frame_idx: int, time_sec: float, image: Any = None) -> Any:
            history.frame, history.time_sec = frame_idx, time_sec
            tracker._frame, tracker._time = frame_idx, time_sec
            if self.failure is not None:
                raise self.failure
            sides = []
            for side in O.SIDES:
                value = make_side(side)
                prior_evidence(state, history, value)
                sides.append(state["pending_rec"].isolate_side_result(side, value))
            self.last = SimpleNamespace(frame_idx=frame_idx, time_sec=time_sec,
                p1=sides[0], p2=sides[1], is_match_active=True, match_end_locked=self.locked)
            if self.mutate:
                sides[0].prob_board.cells[0][0] = {1: 1.0}
            return self.last
    pipe = Pipeline()
    pipe.held, pipe.locked, pipe.failure, pipe.mutate = False, False, None, False
    pipe._post_match_lockdown_active = False
    tracker._pipeline = pipe
    return SimpleNamespace(RecognitionPipeline=Pipeline), pipe, history, state


@pytest.fixture
def setup(tmp_path: Path) -> Any:
    values = fake_runtime(tmp_path)
    yield values
    for key in ("provisional_current_connection", "current_scope_sink", "hidden_probability_observer"):
        values[-1][key].close()


def call_saved(setup: Any) -> tuple[Any, Any]:
    collector, pipe, history, state = setup
    original_update, original_isolate = collector.RecognitionPipeline.update, state["pending_rec"].isolate_side_result
    with ExitStack() as stack:
        rec = O.install(stack, collector, history, state, expected_frames=[FRAME])
        result = pipe.update(FRAME, FRAME / 60)
        assert result is pipe.last
    assert collector.RecognitionPipeline.update is original_update
    assert state["pending_rec"].isolate_side_result is original_isolate
    return rec, result


@pytest.mark.parametrize("held", (False, True))
def test_normal_hold_actual_return_and_saved_copy(setup: Any, held: bool) -> None:
    setup[1].held = held
    rec, result = call_saved(setup)
    assert not rec.errors and rec.closed and len(rec.rows) == 1
    row = rec.rows[0]
    assert row["update"]["match_end_locked_observed"] is True
    assert row["available_frame"] == FRAME and row["ledger"] == O.disconnected()
    assert all(s["identity"]["isolate_return_is_final"] is True for s in row["sides"].values())
    assert all(s["identity"]["after_probability_unchanged"] is True for s in row["sides"].values())
    assert all(s["identity"]["final_probability_matches_after"] is True for s in row["sides"].values())
    assert (row["sides"]["2P"]["after_hold"]["confirmed"] is None) is held
    result.p1.prob_board.cells[0][0] = {5: 1.0}
    assert row["sides"]["1P"]["final"]["probability"]["cells"][0][0] == [[0, 1.0]]
    O.finish(setup[-1])
    O.verify(rec.output, expected_frames=[FRAME])
    with pytest.raises(ValueError, match="expected_scope"):
        O.verify(rec.output)
    O.write(Path(os.environ["MANUFACTURING_OUTPUT"]) / f"ARTIFICIAL_HOLD_{held}.json", row)


@pytest.mark.parametrize("value", (True, None, 0))
def test_lock_is_raw_not_stable_inference(setup: Any, value: Any) -> None:
    setup[1].locked = value
    rec, _ = call_saved(setup)
    assert rec.rows[0]["update"]["match_end_locked"] is value
    assert "match_end_locked_not_ready" in rec.rows[0]["hold_reasons"]
    O.finish(setup[-1])


def test_missing_postlock_explicit(setup: Any) -> None:
    del setup[1]._post_match_lockdown_active
    rec, _ = call_saved(setup)
    assert rec.rows[0]["update"]["post_match_lockdown_active"] is None
    assert rec.rows[0]["update"]["post_match_lockdown_active_observed"] is False


def test_probability_mutation_is_not_silent(setup: Any) -> None:
    setup[1].mutate = True
    rec, _ = call_saved(setup)
    identity = rec.rows[0]["sides"]["1P"]["identity"]
    assert identity["before_probability_unchanged"] is False
    assert identity["after_probability_unchanged"] is False
    assert identity["final_probability_matches_after"] is False


def test_exception_identity_restore_and_finish_refused(setup: Any) -> None:
    collector, pipe, history, state = setup
    failure = LookupError("actual original exception")
    pipe.failure = failure
    original = collector.RecognitionPipeline.update
    with ExitStack() as stack:
        rec = O.install(stack, collector, history, state)
        with pytest.raises(LookupError) as caught:
            pipe.update(FRAME, FRAME / 60)
        assert caught.value is failure
    assert collector.RecognitionPipeline.update is original and rec.errors
    assert rec.rows[0]["capture_status"] == "FAILED"
    with pytest.raises(ValueError, match="lifetime_or_error"):
        O.finish(state)


def test_no_update_coverage_and_exclusive(setup: Any) -> None:
    collector, _, history, state = setup
    with ExitStack() as stack:
        O.install(stack, collector, history, state)
        with pytest.raises(ValueError, match="reentry"):
            O.install(stack, collector, history, state)
    with pytest.raises(ValueError, match="coverage"):
        O.finish(state)


def test_unbound_pipeline_not_unknown_ledger(setup: Any) -> None:
    collector, pipe, history, state = setup
    state["tracker"]._pipeline = None
    with ExitStack() as stack:
        rec = O.install(stack, collector, history, state)
        with pytest.raises(ValueError, match="unbound"):
            pipe.update(FRAME, FRAME / 60)
    assert rec.errors
    with pytest.raises(ValueError, match="lifetime"):
        O.finish(state)


def test_saved_source_change_rejected(setup: Any) -> None:
    rec, _ = call_saved(setup)
    O.finish(setup[-1])
    rows = copy.deepcopy(rec.rows)
    rows[0]["source_id"] = "different"
    (rec.output / O.SIDECAR).write_text(json.dumps(rows[0]) + "\n")
    receipt = json.loads((rec.output / O.RECEIPT).read_text())
    receipt["sha256"][O.SIDECAR] = O.sha(rec.output / O.SIDECAR)
    (rec.output / O.RECEIPT).write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="provenance"):
        O.verify(rec.output, expected_frames=[FRAME])


def test_outer_scope_keeps_candidate_window_unchanged(setup: Any) -> None:
    collector, pipe, history, state = setup
    outside = 33000
    with ExitStack() as stack:
        rec = O.install(stack, collector, history, state, expected_frames=[outside])
        result = pipe.update(outside, outside / 60)
        assert result is pipe.last
    row = rec.rows[0]
    assert row["candidate_scope_selected"] is False and row["available_frame"] == outside
    assert row["sides"]["1P"]["final"]["probability"]["present"] is True
    assert all(s["pb"] is None and s["sm"] is None and s["candidate_row"] is None for s in row["sides"].values())
    assert state["hidden_probability_observer"].rows == state["current_scope_sink"].rows == []
    assert O.finish(state)["outside_candidate_window_updates"] == 1
    O.verify(rec.output, expected_frames=[outside])
    assert len(O.expected()) == 3624 and O.expected()[0] == 29052 and O.expected()[-1] == 36298
    O.write(Path(os.environ["MANUFACTURING_OUTPUT"]) / "OUTSIDE_CANDIDATE_SCOPE.json", row)


def test_upstream_observer_failure_not_ledger_unknown(setup: Any) -> None:
    setup[-1]["current_scope_sink"].failures.append("actual-upstream-failure")
    rec, _ = call_saved(setup)
    assert rec.errors and rec.rows[0]["capture_status"] == "FAILED"
    assert rec.rows[0]["ledger"] == O.disconnected()
    with pytest.raises(ValueError, match="lifetime_or_error"):
        O.finish(setup[-1])


def test_original_isolate_exception_identity_and_restore(setup: Any, monkeypatch: Any) -> None:
    collector, pipe, history, state = setup
    original_error = OSError("original isolate failed")
    def failure(side: str, value: Any) -> Any:
        raise original_error
    monkeypatch.setattr(state["pending_rec"], "isolate_side_result", failure)
    with ExitStack() as stack:
        rec = O.install(stack, collector, history, state, expected_frames=[FRAME])
        with pytest.raises(OSError) as caught:
            pipe.update(FRAME, FRAME / 60)
        assert caught.value is original_error
    assert state["pending_rec"].isolate_side_result is failure
    assert rec.errors and rec.rows[0]["update"]["exception"]["type"] == "OSError"


def test_wrong_return_clock_preserved_but_receipt_refused(setup: Any, monkeypatch: Any) -> None:
    collector, pipe, history, state = setup
    original = collector.RecognitionPipeline.update
    def wrong_clock(self: Any, *args: Any, **kwargs: Any) -> Any:
        value = original(self, *args, **kwargs)
        value.frame_idx = FRAME + 2
        return value
    monkeypatch.setattr(collector.RecognitionPipeline, "update", wrong_clock)
    rec, result = call_saved(setup)
    assert result.frame_idx == FRAME + 2 and rec.errors
    assert rec.rows[0]["available_frame"] is None
    with pytest.raises(ValueError, match="lifetime_or_error"):
        O.finish(state)


def test_frozen_actual_update_noninterference(monkeypatch: Any, tmp_path: Path) -> None:
    if os.environ.get("CONTEXT_ACTUAL_RUNTIME") != "1":
        pytest.skip("専用runtime processで一度だけ実行する")
    parent = ROOT.parent / "g2_current_scope_capture_2026-09-09_v1"
    R = load("_context_actual_runtime_fixture", parent / "test_runtime.py")
    cache = R.guarded_hash_cache.__wrapped__()
    next(cache)
    try:
        _actual_case(R, monkeypatch, tmp_path)
    finally:
        with pytest.raises(StopIteration):
            next(cache)


def _actual_case(R: Any, monkeypatch: Any, tmp_path: Path) -> None:
    frozen_gen = R.frozen.__wrapped__()
    frozen = next(frozen_gen)
    try:
        prepared = R.prepared.__wrapped__()
        original = R.O.install
        captured = []
        def installed(stack: Any, collector: Any, history: Any, state: Any) -> Any:
            value = original(stack, collector, history, state)
            captured.append(O.install(stack, collector, history, state))
            return value
        off = R.run_version(frozen, prepared, monkeypatch, tmp_path / "off", FRAME, True)
        with monkeypatch.context() as scoped:
            scoped.setattr(R.O, "install", installed)
            on = R.run_version(frozen, prepared, scoped, tmp_path / "on", FRAME, True)
        assert off == on
        rec = captured[0]
        assert rec.closed and not rec.errors and len(rec.rows) == 1
        assert all(row["identity"]["isolate_return_is_final"] is True for row in rec.rows[0]["sides"].values())
        O.write(Path(os.environ["MANUFACTURING_OUTPUT"]) / "ACTUAL_FROZEN_UPDATE.json",
            {"rows": rec.rows, "all_old_values_equal": off == on, "artificial_reader": True,
             "real_video": False, "quality_gate_clear": False})
    finally:
        with pytest.raises(StopIteration):
            next(frozen_gen)
