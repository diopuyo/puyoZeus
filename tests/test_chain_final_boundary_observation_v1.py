"""終端観測hookの実呼出・不変性・例外・時計をCPUだけで確認する。"""

from __future__ import annotations

import ast
import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import chain_final_boundary_observation_v1 as target
from src.board import Board
from src.board_state_machine import BoardState, DetectorSignals, StateContext
from src.state_detectors import ChainPhaseDetector, GravitySettleDetector, TsumoPhaseDetector


FRAME = 35774


def observer(frame: int = FRAME) -> target.BoundaryObserver:
    rec = target.base.Recorder(io.StringIO(), target.base.board_value(Board()))
    rec.begin_frame(frame, frame / 60)
    return target.BoundaryObserver(rec)


def rows(value: target.BoundaryObserver) -> list[dict[str, Any]]:
    return [json.loads(line) for line in value.rec.stream.getvalue().splitlines()]


def context(state: BoardState = BoardState.CHAIN) -> StateContext:
    return StateContext(state=state, frame_idx=FRAME, confirmed_board=Board())


def signals() -> DetectorSignals:
    return DetectorSignals(time_sec=FRAME / 60, cnn_board=Board(), is_match_active=True)


@pytest.mark.parametrize("detector_type", [ChainPhaseDetector, GravitySettleDetector, TsumoPhaseDetector])
def test_actual_detector_return_and_mutation_are_unchanged(detector_type: Any) -> None:
    original, observed = detector_type(), detector_type()
    state = BoardState.GRAVITY_SETTLE if detector_type is GravitySettleDetector else BoardState.CHAIN
    plain_ctx, observed_ctx = context(state), context(state)
    sig, value = signals(), observer()
    value.active_side = "2P"
    expected = original.detect(plain_ctx, sig)
    result = value.detector_wrapper(detector_type.__name__, detector_type.detect)(observed, observed_ctx, sig)
    assert result == expected
    assert plain_ctx.state == observed_ctx.state
    assert plain_ctx.confirmed_board == observed_ctx.confirmed_board
    assert all(getattr(original, field, None) == getattr(observed, field, None)
               for field in target.COUNTERS)
    assert len(rows(value)) == 1 and rows(value)[0]["side"] == "2P"


def test_detector_is_not_called_an_extra_time() -> None:
    calls, value = [], observer()
    value.active_side = "1P"
    def detect(self: Any, ctx: Any, sig: Any) -> str:
        calls.append(sig)
        self._stable_consec += 1
        return "sentinel"
    detector = SimpleNamespace(_stable_consec=2)
    result = value.detector_wrapper("fixture", detect)(detector, context(), signals())
    assert len(calls) == 1 and detector._stable_consec == 3 and result == "sentinel"
    record = rows(value)[0]
    assert record["counters_before"]["_stable_consec"] == 2
    assert record["counters_after"]["_stable_consec"] == 3


def test_outside_window_and_unbound_side_do_not_emit() -> None:
    for value in (observer(target.FIRST_FRAME - 1), observer()):
        value.detector_wrapper("ChainPhaseDetector", ChainPhaseDetector.detect)(
            ChainPhaseDetector(), context(), signals())
        assert rows(value) == []


def test_bad_detector_clock_fails_before_original() -> None:
    value = observer()
    value.active_side = "2P"
    sig = replace(signals(), time_sec=0)
    with pytest.raises(RuntimeError, match="clock"):
        value.detector_wrapper("ChainPhaseDetector", ChainPhaseDetector.detect)(
            ChainPhaseDetector(), context(), sig)
    assert rows(value) == []


def test_helper_keeps_args_return_and_caller_without_guessing_side() -> None:
    calls, value = [], observer()
    def helper(current_next: Any, start_next: Any, flag: bool = True) -> object:
        calls.append((current_next, start_next, flag))
        return False
    result = value.helper_wrapper("exit", helper)((5, 5), (5, 5))
    assert result is False and calls == [((5, 5), (5, 5), True)]
    row = rows(value)[0]
    assert row["side"] is None and row["arguments"]["flag"] is True
    assert row["caller"]["source_path"] == __file__
    assert row["side_inferred_from_call_order"] is False


def pipe_fields() -> SimpleNamespace:
    return SimpleNamespace(**{f"_{field}_2p": None for field in target.PIPELINE_FIELDS})


def test_step_preserves_result_and_detaches_side_after_original_error() -> None:
    value, pipe, ctx = observer(), pipe_fields(), context()
    def fail(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx.state = BoardState.GRAVITY_SETTLE
        assert value.active_side == "2P"
        raise ValueError("original")
    with pytest.raises(ValueError, match="original"):
        value.step_wrapper(fail)(pipe, "2P", FRAME, FRAME / 60, True, Board(), None,
                                 sm=SimpleNamespace(context=ctx))
    assert value.active_side is None and ctx.state == BoardState.GRAVITY_SETTLE
    assert [row["kind"] for row in rows(value)] == ["end_boundary_step_enter"]


def test_step_success_records_before_and_after_without_altering_return() -> None:
    value, pipe, ctx = observer(), pipe_fields(), context()
    expected = SimpleNamespace(state=BoardState.STABLE, confirmed_board=Board())
    def step(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx.state = BoardState.STABLE
        return expected
    result = value.step_wrapper(step)(pipe, "2P", FRAME, FRAME / 60, True, Board(), None,
                                      sm=SimpleNamespace(context=ctx), slide_motion=True)
    assert result is expected and value.active_side is None
    before, after = rows(value)
    assert before["context"]["state"] == BoardState.CHAIN.value
    assert after["context"]["state"] == BoardState.STABLE.value
    assert before["step_kwargs"]["slide_motion"] is True


def test_install_restores_all_seven_methods_after_error() -> None:
    from src import recognition_pipeline as pipeline
    from src import state_detectors
    cls = pipeline.RecognitionPipeline
    targets = [(cls, "_step_side")]
    targets.extend((getattr(state_detectors, name), "detect") for name in target.DETECTORS)
    targets.extend((pipeline, name) for name in target.HELPERS)
    original = [getattr(obj, name) for obj, name in targets]
    with pytest.raises(ValueError), contextlib.ExitStack() as stack:
        target.install(stack, SimpleNamespace(RecognitionPipeline=cls), observer().rec)
        assert all(getattr(obj, name) is not before for (obj, name), before in zip(targets, original))
        raise ValueError("fixture")
    assert all(getattr(obj, name) is before for (obj, name), before in zip(targets, original))


def test_all_functions_are_within_fifty_lines() -> None:
    tree = ast.parse(Path(target.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert node.end_lineno - node.lineno + 1 <= 50, node.name
