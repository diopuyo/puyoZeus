"""連鎖終了後 grace 隔離 shadow の CPU 契約検査。"""

from __future__ import annotations

import ast
import contextlib
import dis
import inspect
import textwrap
from enum import Enum
from types import SimpleNamespace as NS
from typing import Any

import pytest

from scripts import diagnose_video38_c6_pending_commit_shadow_v1 as pending
from scripts import post_chain_grace_shadow_v1 as shadow
from src.board import Board
from src.board_state_machine import BoardState as RealState
from src.recognition_pipeline import RecognitionPipeline


class State(Enum):
    CHAIN = "chain"
    GRAVITY_SETTLE = "gravity_settle"
    STABLE = "stable"


BoardState = State


class Grid:
    def __init__(self, value: int) -> None:
        self.value = value

    def copy(self) -> "Grid":
        return Grid(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Grid) and self.value == other.value


class Pipeline:
    def _stash_and_clear_active_chain(self, side: str) -> None:
        suffix = side.lower()
        active = getattr(self, f"_active_chain_{suffix}", None)
        if active is None:
            return
        setattr(self, f"_last_chain_event_for_settle_{suffix}", active)
        setattr(self, f"_active_chain_{suffix}", None)
        self._rec.emit(mutation(side, self._instance_id))

    def _step_side(self, side: str, frame_idx: int, time_sec: float,
                   prev_state: Any, ctx: Any, _effective_chain_event: Any) -> Any:
        if (
            prev_state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE)
            and ctx.state == BoardState.STABLE
            and _effective_chain_event is not None
        ):
            if side == "1P":
                self._last_chain_event_for_settle_1p = None
            else:
                self._last_chain_event_for_settle_2p = None
            cr = self._chain_sim.simulate(_effective_chain_event.before_board)
            if cr.chain_count > 0 and cr.final_board is not None:
                final = cr.final_board.copy()
                ctx.confirmed_board = final
                ctx.pending_board = final.copy()
                self._chain_verify_pending_1p = {"expected": final}
                target_tsumo = self._tsumo_count_1p
                for color in target_tsumo:
                    target_tsumo[color] -= 1
                self._constraint_valid_1p = True
        landing_pending = self._landing_pending_1p if side == "1P" else self._landing_pending_2p
        if landing_pending is not None and landing_pending[0] == frame_idx and ctx.confirmed_board is not None:
            grace_until = frame_idx + 5
            grace_until_time = time_sec + 0.1
            if side == "1P":
                self._landing_grace_1p = (grace_until, ctx.confirmed_board.copy(), grace_until_time)
                self._landing_pending_1p = None
            else:
                self._landing_grace_2p = (grace_until, ctx.confirmed_board.copy(), grace_until_time)
                self._landing_pending_2p = None
        return ctx.confirmed_board


class Generation:
    def __init__(self, reset: int = 1, action: int | None = 4) -> None:
        self.reset_epoch, self.action_revision = reset, action


class Recorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.frame, self.time_sec = 100, 1.0
        self.handle = NS(instance_id=7, side="2P")
        self.active_handles = {"2P": self.handle}
        transaction = NS(state=NS(value="pending"))
        self.active_candidates = {"2P": {"instance_id": 7, "status": "pending",
                                         "transaction": transaction}}
        self.current = Generation()
        self.generation_recorder = NS(generation=lambda _side: self.current)
        self.snapshot = NS(generation=Generation(), status=NS(value="provisional"))
        self.ledger = NS(snapshot=lambda _handle: self.snapshot)
        self.clock_active = True

    def _clock_active(self) -> bool:
        return self.clock_active

    def emit(self, row: dict[str, Any]) -> None:
        self.rows.append(row.copy())


def mutation(side: str = "2P", instance: int = 7) -> dict[str, Any]:
    return {"kind": "boundary_repair_mutation", "repair": "chain_end",
            "side": side, "before": {"active": {"x": 1}},
            "after": {"active": None}, "evidence": {"instance_id": instance}}


def pipe() -> Pipeline:
    value = Pipeline()
    value._landing_grace_1p = value._landing_grace_2p = None
    value._landing_pending_1p = value._landing_pending_2p = None
    value._active_chain_1p = value._active_chain_2p = None
    value._tsumo_count_1p = {1: 5}
    value._constraint_valid_1p = False
    value._chain_sim = NS(simulate=lambda _board: NS(chain_count=1, final_board=Grid(99)))
    return value


def stash_owner(controller: Any, rec: Recorder, value: Pipeline, instance: int = 7) -> None:
    """実stash呼出scopeと同期emitを通してownerを作る。"""
    original_emit = rec.emit
    rec.emit = lambda row: (original_emit(row), controller.observe(row))[0]
    value._rec, value._instance_id = rec, instance
    rec.generation_recorder._pipeline = value
    value._active_chain_2p = NS(chain_count=13)
    controller.run_stash(Pipeline._stash_and_clear_active_chain, value, "2P")
    rec.emit = original_emit


def pending_step() -> Any:
    return pending._build_transformed_step(Pipeline._step_side, lambda *_args: None)[0]


def test_owner_requires_success_clock_handle_and_current_generation() -> None:
    rec = Recorder()
    controller = shadow.PostChainGraceController(rec, rec.emit)
    stash_owner(controller, rec, pipe())
    assert controller.owners["2P"] == shadow.GraceOwner("2P", 7, 1, 4)
    for change in ("clock", "handle", "generation"):
        other = Recorder()
        if change == "clock":
            other.clock_active = False
        elif change == "handle":
            other.active_handles["2P"] = NS(instance_id=8, side="2P")
        else:
            other.current = Generation(action=5)
        target = shadow.PostChainGraceController(other, other.emit)
        stash_owner(target, other, pipe())
        assert target.owners == {}


def test_unknown_action_does_not_release_but_reset_or_action_change_does() -> None:
    rec, value = Recorder(), pipe()
    controller = shadow.PostChainGraceController(rec, rec.emit)
    stash_owner(controller, rec, value)
    rec.current = Generation(action=None)
    controller.before_step(value, "2P", 100, 1.0)
    assert "2P" in controller.owners
    rec.current = Generation(action=5)
    controller.before_step(value, "2P", 102, 1.1)
    assert "2P" not in controller.owners
    rec.current = Generation(action=4)
    stash_owner(controller, rec, value)
    rec.current = Generation(reset=2, action=None)
    controller.before_step(value, "2P", 104, 1.2)
    assert "2P" not in controller.owners


def test_external_emit_noop_exception_and_invalidated_handle_cannot_own() -> None:
    rec, value = Recorder(), pipe()
    controller = shadow.PostChainGraceController(rec, rec.emit)
    controller.observe(mutation())
    assert controller.owners == {}
    original_emit = rec.emit
    rec.emit = lambda row: (original_emit(row), controller.observe(row))[0]
    controller.run_stash(lambda _pipe, _side: rec.emit(mutation()), value, "2P")
    assert controller.owners == {}
    value._active_chain_2p = NS(chain_count=13)
    rec.generation_recorder._pipeline = value

    def fail(_pipe: Any, _side: str) -> None:
        rec.emit(mutation())
        raise ValueError("stash失敗")

    with pytest.raises(ValueError, match="stash失敗"):
        controller.run_stash(fail, value, "2P")
    assert controller.owners == {}
    rec.snapshot.status = NS(value="invalidated")
    value._rec, value._instance_id = rec, 7
    value._active_chain_2p = NS(chain_count=13)
    controller.run_stash(Pipeline._stash_and_clear_active_chain, value, "2P")
    assert controller.owners == {}


def test_new_generation_stash_expires_old_owner_before_replacement() -> None:
    rec, value = Recorder(), pipe()
    controller = shadow.PostChainGraceController(rec, rec.emit)
    stash_owner(controller, rec, value)
    rec.current = Generation(reset=2, action=5)
    rec.snapshot = NS(generation=Generation(reset=2, action=5),
                      status=NS(value="provisional"))
    rec.handle = NS(instance_id=8, side="2P")
    rec.active_handles["2P"] = rec.handle
    rec.active_candidates["2P"] = {
        "instance_id": 8, "status": "pending",
        "transaction": NS(state=NS(value="pending"))}
    stash_owner(controller, rec, value, instance=8)
    assert controller.owners["2P"] == shadow.GraceOwner("2P", 8, 2, 5)


@pytest.mark.parametrize("replace_same_instance", [False, True])
def test_stash_rejects_handle_or_candidate_replacement(replace_same_instance: bool) -> None:
    rec, value = Recorder(), pipe()
    controller = shadow.PostChainGraceController(rec, rec.emit)
    original_emit = rec.emit
    rec.emit = lambda row: (original_emit(row), controller.observe(row))[0]
    value._active_chain_2p = NS(chain_count=13)

    def replace(pipe: Any, side: str) -> None:
        pipe._active_chain_2p = None
        instance = 7 if replace_same_instance else 8
        rec.active_handles[side] = NS(instance_id=instance, side=side)
        rec.active_candidates[side] = {
            "instance_id": instance, "status": "pending",
            "transaction": NS(state=NS(value="pending"))}
        rec.emit(mutation(side, instance))

    controller.run_stash(replace, value, "2P")
    assert controller.owners == {}


def test_bound_controller_never_clears_or_blocks_another_pipe() -> None:
    rec, value = Recorder(), pipe()
    controller = shadow.PostChainGraceController(rec, rec.emit)
    stash_owner(controller, rec, value)
    other = pipe()
    other._landing_grace_2p = (110, Grid(55), 2.0)
    controller.before_step(other, "2P", 102, 1.1)
    assert other._landing_grace_2p == (110, Grid(55), 2.0)
    assert controller.allow_generation(other, "2P", "2P", 102, 1.1) is True


def test_composite_blocks_only_grace_and_preserves_pending_c6_transform() -> None:
    rec, value = Recorder(), pipe()
    original = Pipeline._step_side
    Pipeline._step_side = pending_step()
    try:
        with contextlib.ExitStack() as stack:
            controller = shadow.install(stack, NS(RecognitionPipeline=Pipeline), rec)
            value._rec, value._instance_id = rec, 7
            rec.generation_recorder._pipeline = value
            value._active_chain_2p = NS(chain_count=13)
            value._stash_and_clear_active_chain("2P")
            value._landing_pending_2p = (100, (2, 3))
            ctx = NS(state=State.STABLE, confirmed_board=Grid(46), pending_board=Grid(46))
            result = value._step_side("2P", 100, 1.0, State.GRAVITY_SETTLE,
                                      ctx, NS(before_board=Grid(1)))
            assert result == Grid(46) and value._landing_grace_2p is None
            assert value._landing_pending_2p is None
            assert value._tsumo_count_1p == {1: 5}
            assert value._constraint_valid_1p is False
            assert rec.step_code is Pipeline._step_side.__code__
            assert controller.transform_receipt == {
                "c6_pending_blocks": 1, "grace_assignments": ["1P", "2P"],
                "landing_pending_modified": False, "counter_modified": False}
        assert Pipeline._step_side is not original
        assert "emit" not in vars(rec)
        assert "step_code" not in vars(rec)
    finally:
        Pipeline._step_side = original


def test_normal_and_other_side_grace_are_unchanged() -> None:
    rec, value = Recorder(), pipe()
    original = Pipeline._step_side
    Pipeline._step_side = pending_step()
    try:
        with contextlib.ExitStack() as stack:
            shadow.install(stack, NS(RecognitionPipeline=Pipeline), rec)
            value._landing_pending_1p = (100, (1, 2))
            ctx = NS(state=State.STABLE, confirmed_board=Grid(55), pending_board=Grid(55))
            value._step_side("1P", 100, 1.0, State.STABLE, ctx, None)
            assert value._landing_grace_1p == (105, Grid(55), 1.1)
    finally:
        Pipeline._step_side = original


def test_release_frame_cannot_recreate_stale_grace() -> None:
    rec, value = Recorder(), pipe()
    controller = shadow.PostChainGraceController(rec, rec.emit)
    stash_owner(controller, rec, value)
    rec.current = Generation(action=5)
    value._landing_grace_2p = (99, Grid(56), 2.0)
    assert controller.allow_generation(value, "2P", "2P", 100, 1.0) is False
    assert value._landing_grace_2p is None and "2P" not in controller.owners
    assert controller.allow_generation(value, "2P", "2P", 102, 1.1) is True


def test_transform_structure_and_function_length() -> None:
    source = inspect.getsource(shadow)
    tree = ast.parse(source)
    too_long = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            if end - node.lineno + 1 > 50:
                too_long.append((node.name, end - node.lineno + 1))
    assert too_long == []
    assert shadow.GUARD_PATHS == (
        "scripts/diagnose_video38_c6_pending_commit_shadow_v1.py",)


def test_composite_grace_store_lines_are_not_incremented_twice() -> None:
    rec = Recorder()
    transformed, _ = shadow._composite_step(
        pending_step(), shadow.PostChainGraceController(rec, rec.emit))
    stores = [instruction for instruction in dis.get_instructions(transformed)
              if instruction.opname == "STORE_ATTR"
              and instruction.argval in ("_landing_grace_1p", "_landing_grace_2p")]
    lines = [instruction.positions.lineno for instruction in stores]
    source, first = inspect.getsourcelines(Pipeline._step_side)
    tree = ast.parse(textwrap.dedent("".join(source)))
    expected = [first + node.lineno - 1 for node in ast.walk(tree)
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Attribute)
                    and target.attr in ("_landing_grace_1p", "_landing_grace_2p")
                    for target in node.targets)]
    assert sorted(lines) == sorted(expected * 2)


def test_real_step_side_consumes_pending_without_creating_owned_grace() -> None:
    rec = Recorder()
    original = RecognitionPipeline._step_side
    RecognitionPipeline._step_side = pending._build_transformed_step(
        original, lambda *_args: None)[0]
    value = RecognitionPipeline(image_reader=NS(), match_state_detector=NS(),
                                score_ocr=None, chain_tracker_1p=None,
                                chain_tracker_2p=None)
    board = Board()
    value._sm_2p.context.state = RealState.GRAVITY_SETTLE
    value._sm_2p.context.confirmed_board = board.copy()
    value._landing_pending_2p = (100, (2, 3))
    try:
        with contextlib.ExitStack() as stack:
            controller = shadow.install(stack, NS(RecognitionPipeline=RecognitionPipeline), rec)
            rec.generation_recorder._pipeline = value
            value._instance_id = 7
            value._active_chain_2p = NS(chain_count=13)

            def inner_stash(pipe: Any, side: str) -> None:
                setattr(pipe, "_active_chain_2p", None)
                rec.emit(mutation(side, 7))

            controller.run_stash(inner_stash, value, "2P")
            assert "2P" in controller.owners
            value._step_side("2P", 100, 1.0, True, board, None,
                             score_d_2p_for_ojama=0, sm=value._sm_2p,
                             gen=value._gen_2p, drift=value._drift_2p,
                             score_tracker=None)
            assert value._landing_pending_2p is None
            assert value._landing_grace_2p is None
            value._sm_1p.context.state = RealState.GRAVITY_SETTLE
            value._sm_1p.context.confirmed_board = board.copy()
            value._landing_pending_1p = (102, (1, 4))
            value._step_side("1P", 102, 1.1, True, board, None,
                             score_d_2p_for_ojama=0, sm=value._sm_1p,
                             gen=value._gen_1p, drift=value._drift_1p,
                             score_tracker=None)
            assert value._landing_pending_1p is None
            assert value._landing_grace_1p is not None
    finally:
        RecognitionPipeline._step_side = original


def test_missing_pending_transform_is_fail_closed() -> None:
    rec = Recorder()
    with pytest.raises(RuntimeError, match="先に接続"):
        shadow._composite_step(Pipeline._step_side,
                               shadow.PostChainGraceController(rec, rec.emit))
