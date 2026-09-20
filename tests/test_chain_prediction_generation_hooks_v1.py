"""世代hookの非変更性と順序をCPUで確認する。物理的着手の精度試験ではない。"""

from __future__ import annotations

import ast
import contextlib
import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.chain_prediction_generation_hooks_v1 import (
    IDENTITY_SCOPE, PipelineGenerationRecorder, install_generation_hooks,
)


class FakeMachine:
    """戻り値と副作用を検査できる、最小のstate machine対照。"""

    def __init__(self) -> None:
        self.context = SimpleNamespace(state="stable", board=object())

    def update(self, state: str, *, fail: bool = False) -> Any:
        if fail:
            raise RuntimeError("元update失敗")
        self.context.state = state
        return self.context

    def reset(self, *, fail: bool = False) -> str:
        if fail:
            raise RuntimeError("元reset失敗")
        self.context.state = "stable"
        return "reset_result"


class FakePipeline:
    """fresh loadと時計付きupdateの既存API形を保存する。"""

    def __init__(self, marker: str) -> None:
        self.marker = marker
        self._sm_1p, self._sm_2p = FakeMachine(), FakeMachine()

    @classmethod
    def load_default(cls, marker: str = "original") -> FakePipeline:
        return cls(marker)

    def update(
        self, frame_idx: int, time_sec: float, target: str, *, side: str = "1P",
        fail: bool = False,
    ) -> Any:
        return getattr(self, f"_sm_{side.lower()}").update(target, fail=fail)


def _recorder() -> tuple[PipelineGenerationRecorder, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    return PipelineGenerationRecorder(events.append), events


def test_hooks_preserve_return_context_board_and_unknown_initial_action() -> None:
    rec, events = _recorder()
    with contextlib.ExitStack() as stack:
        install_generation_hooks(stack, FakePipeline, FakeMachine, rec)
        pipe = FakePipeline.load_default(marker="passed_through")
        assert pipe.marker == "passed_through"
        assert rec.generation("1P").action_revision is None
        board = pipe._sm_1p.context.board
        result = pipe.update(10, 1.0, "tsumo_fall")
        assert result is pipe._sm_1p.context and result.board is board
        assert rec.generation("1P").action_revision == 1
        assert rec.generation("2P").action_revision is None
    assert [row["generation_sequence"] for row in events] == [1, 2, 3]
    assert events[-1]["frame_idx"] == 10 and events[-1]["time_sec"] == 1.0
    assert all(row["identity_scope"] == IDENTITY_SCOPE for row in events)
    assert all(row["commit_permission_issued"] is False for row in events)


def test_only_new_tsumo_entry_increments_action_and_reset_is_side_local() -> None:
    rec, _ = _recorder()
    with contextlib.ExitStack() as stack:
        install_generation_hooks(stack, FakePipeline, FakeMachine, rec)
        pipe = FakePipeline.load_default()
        for frame, target in enumerate(("tsumo_fall", "tsumo_fall", "stable", "tsumo_fall")):
            pipe.update(frame, float(frame), target)
        assert rec.generation("1P").action_revision == 2
        assert pipe._sm_1p.reset() == "reset_result"
        assert rec.generation("1P").reset_epoch == 1
        assert rec.generation("1P").action_revision is None
        assert rec.generation("2P").reset_epoch == 0
        pipe.update(4, 4.0, "tsumo_fall", side="2P")
        assert rec.generation("2P").action_revision == 1


@pytest.mark.parametrize("target", ["stable", "chain", "gravity_settle", "ojama_fall"])
def test_other_states_do_not_invent_action(target: str) -> None:
    rec, _ = _recorder()
    with contextlib.ExitStack() as stack:
        install_generation_hooks(stack, FakePipeline, FakeMachine, rec)
        pipe = FakePipeline.load_default()
        pipe.update(10, 1.0, target)
        assert rec.generation("1P").action_revision is None


@pytest.mark.parametrize("method", ["update", "reset"])
def test_original_failures_are_propagated_without_generation_success(method: str) -> None:
    rec, events = _recorder()
    with contextlib.ExitStack() as stack:
        install_generation_hooks(stack, FakePipeline, FakeMachine, rec)
        pipe = FakePipeline.load_default()
        before = rec.generation("1P")
        with pytest.raises(RuntimeError, match="元"):
            if method == "update":
                pipe.update(0, 0.0, "tsumo_fall", fail=True)
            else:
                pipe._sm_1p.reset(fail=True)
        assert rec.generation("1P") == before and len(events) == 2


def test_all_descriptors_restore_after_exception() -> None:
    rec, _ = _recorder()
    targets = ((FakePipeline, "load_default"), (FakePipeline, "update"),
               (FakeMachine, "update"), (FakeMachine, "reset"))
    original = [inspect.getattr_static(cls, name) for cls, name in targets]
    with pytest.raises(RuntimeError):
        with contextlib.ExitStack() as stack:
            install_generation_hooks(stack, FakePipeline, FakeMachine, rec)
            FakePipeline.load_default()
            raise RuntimeError("終了時fault")
    assert [inspect.getattr_static(cls, name) for cls, name in targets] == original


@pytest.mark.parametrize("frame,time", [
    (-1, 1.0), (True, 1.0), (1.0, 1.0), (1, True), (1, -1.0),
    (1, float("nan")), (1, float("inf")), (1, "1.0"),
])
def test_invalid_clock_is_rejected(frame: Any, time: Any) -> None:
    rec, _ = _recorder()
    pipe = FakePipeline.load_default()
    rec.bind_pipeline(pipe)
    with pytest.raises(ValueError):
        rec.begin_frame(pipe, frame, time)


@pytest.mark.parametrize("frame,time", [(9, 1.1), (11, 0.9)])
def test_clock_reversal_is_rejected_independently(frame: int, time: float) -> None:
    rec, _ = _recorder()
    pipe = FakePipeline.load_default()
    rec.bind_pipeline(pipe)
    rec.begin_frame(pipe, 10, 1.0)
    rec.end_frame()
    with pytest.raises(ValueError, match="逆行"):
        rec.begin_frame(pipe, frame, time)


def test_unknown_machines_and_wrong_pipeline_are_not_bound_to_one_player() -> None:
    rec, events = _recorder()
    pipe = FakePipeline.load_default()
    rec.bind_pipeline(pipe)
    unknown = FakeMachine()
    rec.record_reset(unknown)
    rec.record_transition(unknown, "stable", "tsumo_fall")
    assert len(events) == 2
    with pytest.raises(ValueError):
        rec.begin_frame(FakePipeline.load_default(), 0, 0.0)
    with pytest.raises(ValueError):
        rec.generation("3P")
    with pytest.raises(ValueError):
        rec.bind_pipeline(pipe)


def test_generation_snapshot_cannot_be_mutated_and_empty_clock_is_not_filled() -> None:
    rec, events = _recorder()
    pipe = FakePipeline.load_default()
    rec.bind_pipeline(pipe)
    generation = rec.generation("1P")
    with pytest.raises(FrozenInstanceError):
        generation.action_revision = 999  # type: ignore[misc]
    assert events[0]["frame_idx"] is None and events[0]["time_sec"] is None
    with pytest.raises(ValueError, match="時計"):
        rec.record_transition(pipe._sm_1p, "stable", "tsumo_fall")
    assert rec.generation("1P").action_revision is None


def test_recording_failure_is_not_swallowed() -> None:
    def fail_emit(row: dict[str, Any]) -> None:
        raise RuntimeError("保存fault")
    rec = PipelineGenerationRecorder(fail_emit)
    with pytest.raises(RuntimeError, match="保存fault"):
        rec.bind_pipeline(FakePipeline.load_default())


@pytest.mark.parametrize("fail_update", [False, True])
def test_external_reset_after_update_has_unknown_clock(fail_update: bool) -> None:
    """正常/例外のupdate後のresetを直前frameへ偽帰属させない。"""
    rec, events = _recorder()
    with contextlib.ExitStack() as stack:
        install_generation_hooks(stack, FakePipeline, FakeMachine, rec)
        pipe = FakePipeline.load_default()
        if fail_update:
            with pytest.raises(RuntimeError):
                pipe.update(10, 1.0, "tsumo_fall", fail=True)
        else:
            pipe.update(10, 1.0, "tsumo_fall")
        pipe._sm_1p.reset()
        assert events[-1]["frame_idx"] is None and events[-1]["time_sec"] is None
        assert events[-1]["clock_source"] == "outside_update_time_unknown"
        assert rec.generation("1P").reset_epoch == 1


def test_same_frame_reset_order_is_recorded_without_reentrant_update() -> None:
    """同frame順序と、再入拒否後にも外側の時計を保持することを確認する。"""
    rec, events = _recorder()
    pipe = FakePipeline.load_default()
    rec.bind_pipeline(pipe)
    rec.begin_frame(pipe, 10, 1.0)
    rec.record_reset(pipe._sm_1p)
    with pytest.raises(ValueError, match="再入"):
        rec.begin_frame(pipe, 10, 1.0)
    rec.record_transition(pipe._sm_1p, "stable", "tsumo_fall")
    rec.end_frame()
    assert [row["reason"] for row in events[-2:]] == [
        "state_machine_reset_succeeded", "state_machine_tsumo_fall_entry",
    ]
    assert all(row["frame_idx"] == 10 for row in events[-2:])
    assert all(row["clock_source"] == "pipeline_update_input" for row in events[-2:])


def test_hook_functions_are_within_fifty_lines() -> None:
    import scripts.chain_prediction_generation_hooks_v1 as module
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert node.end_lineno is not None
            assert node.end_lineno - node.lineno + 1 <= 50
