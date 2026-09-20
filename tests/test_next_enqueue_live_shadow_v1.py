"""NEXT実接続のCPU境界検収。人工認識入力であり動画精度・GPU合格ではない。"""
from __future__ import annotations

import ast
import contextlib
import inspect
import os
import sys
import threading
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import next_enqueue_live_shadow_v1 as subject
from scripts import diagnose_video38_accounting_history_v1 as history


@pytest.fixture(scope="module")
def frozen() -> Any:
    saved, paths, directory = dict(sys.modules), list(sys.path), Path.cwd()
    os.chdir(history.base.SNAPSHOT)
    collector = history.base.load_collector()
    try:
        yield collector
    finally:
        os.chdir(directory)
        sys.path[:] = paths
        for name in list(sys.modules):
            if name == "src" or name.startswith("src."):
                sys.modules.pop(name)
        sys.modules.update({name: value for name, value in saved.items()
                            if name == "src" or name.startswith("src.")})


class Recorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def emit(self, row: dict[str, Any]) -> None:
        self.rows.append(row)


def light_pipe(cls: type) -> Any:
    pipe = object.__new__(cls)
    for side in subject.SIDES:
        for name, value in {"pending_tsumo": deque(), "landing_pending": None,
                            "last_consumed_color": None, "last_seen_next": None,
                            "tsumo_count": Counter(), "first_move_sec": None}.items():
            setattr(pipe, f"_{name}_{side.lower()}", value)
    return pipe


@pytest.fixture
def unit(frozen: Any) -> Any:
    cls = frozen.RecognitionPipeline
    pipe, rec = light_pipe(cls), Recorder()
    controller = subject.NextEnqueueController(cls, rec, subject._programs(cls.update.__globals__))
    return pipe, rec, controller


def packet(controller: Any, pipe: Any, frame: int, pair: tuple[int, int],
           *, side: str = "1P", diff: Any = 1.0, threshold: Any = 8.0,
           pulse: Any = False, active: bool = True) -> None:
    controller.begin(pipe, frame, frame / 60)
    try:
        from src.next_slide_detector import SlideMotionResult
        controller.main_return(pipe, frame, frame / 60, actual_next(pair))
        result = SlideMotionResult(pulse, diff, threshold)
        controller.slide_return(pipe, side, frame, frame / 60, result)
        controller.enqueue(pipe, side, frame, frame / 60, active, pair)
    except BaseException:
        controller.end(False)
        raise
    controller.end(True)


def actual_next(pair: Any, other: Any = None) -> Any:
    """固定factoryが返す実immutable DTOをCPU入力でも使用する。"""
    from src.next_detector import NextDetectionBothResult, NextDetectionResult
    second = pair if other is None else other
    return NextDetectionBothResult(NextDetectionResult(*pair, 2, 3), NextDetectionResult(*second, 2, 3))


def seed(unit: Any, side: str = "1P") -> None:
    pipe, _, controller = unit
    packet(controller, pipe, 32696, (5, 5), side=side)


@pytest.mark.parametrize("side", subject.SIDES)
def test_real_body_quiet_only_and_raw_global_progress(unit: Any, side: str) -> None:
    pipe, rec, controller = unit
    seed(unit, side)
    queue = getattr(pipe, f"_pending_tsumo_{side.lower()}")
    packet(controller, pipe, 32698, (2, 3), side=side, diff=63.0)
    assert not queue and getattr(pipe, f"_last_seen_next_{side.lower()}") == (2, 3)
    packet(controller, pipe, 32700, (3, 5), side=side)
    assert list(queue) == [(5, 5)]
    assert getattr(pipe, f"_landing_pending_{side.lower()}") == (32700, (5, 5))
    assert getattr(pipe, f"_tsumo_count_{side.lower()}") == Counter()
    assert rec.rows[-1]["physical_placement_verified"] is False


@pytest.mark.parametrize("changes", [dict(diff=8.0), dict(diff=60.0), dict(pulse=True)])
def test_motion_and_equal_threshold_do_not_enqueue(unit: Any, changes: dict[str, Any]) -> None:
    pipe, _, controller = unit
    seed(unit)
    packet(controller, pipe, 32698, (3, 5), **changes)
    assert not pipe._pending_tsumo_1p


@pytest.mark.parametrize("changes", [dict(diff=True), dict(diff=float("nan")), dict(diff=-1),
    dict(diff=float("inf")), dict(threshold=0), dict(threshold=True), dict(pulse=0)])
def test_invalid_actual_numeric_refused_before_writes(unit: Any, changes: dict[str, Any]) -> None:
    pipe, _, controller = unit
    seed(unit)
    old = controller.instances[id(pipe)].histories["1P"]
    with pytest.raises(ValueError):
        packet(controller, pipe, 32698, (3, 5), **changes)
    assert controller.instances[id(pipe)].histories["1P"] is old
    assert not pipe._pending_tsumo_1p


def test_same_color_does_not_invent_placement(unit: Any) -> None:
    pipe, _, controller = unit
    seed(unit)
    packet(controller, pipe, 32698, (5, 5))
    assert not pipe._pending_tsumo_1p


def test_missing_slide_is_not_false_quiet(unit: Any) -> None:
    pipe, _, controller = unit
    controller.begin(pipe, 32698, 32698 / 60)
    controller.enqueue(pipe, "1P", 32698, 32698 / 60, True, (5, 5))
    controller.end(True)
    assert controller.instances[id(pipe)].histories["1P"].accepted is None


@pytest.mark.parametrize("value", [[5, 5], (True, 5)])
def test_mutable_or_boolean_next_never_enters_history(unit: Any, value: Any) -> None:
    pipe, _, controller = unit
    with pytest.raises(ValueError, match="可変/未対応|immutable"):
        packet(controller, pipe, 32696, value)
    assert controller.instances[id(pipe)].histories["1P"].accepted is None


def test_inactive_and_reset_only_invalidate_accepted_history(unit: Any) -> None:
    pipe, _, controller = unit
    seed(unit)
    packet(controller, pipe, 32698, (5, 5), active=False)
    assert controller.instances[id(pipe)].histories["1P"].accepted is None
    pipe._pending_tsumo_1p.append((2, 2))
    queue = pipe._pending_tsumo_1p
    controller.reset_finished(controller.reset(pipe), True)
    assert pipe._pending_tsumo_1p is queue and list(queue) == [(2, 2)]
    with pytest.raises(ValueError, match="非空"):
        packet(controller, pipe, 32700, (5, 5))


@pytest.mark.parametrize("frame", [32696, 32695, 32702])
def test_clock_duplicate_reverse_gap_refused(unit: Any, frame: int) -> None:
    pipe, _, controller = unit
    seed(unit)
    with pytest.raises(ValueError):
        packet(controller, pipe, frame, (3, 5))


@pytest.mark.parametrize("target_line", [5126, 5129, 5133, 5134])
@pytest.mark.parametrize("error_type", [RuntimeError, MemoryError])
def test_unfinished_native_write_fault_restores_alias_and_retry(unit: Any, target_line: int,
                                                               error_type: type[Exception]) -> None:
    pipe, _, controller = unit
    seed(unit)
    runtime, queue = controller.instances[id(pipe)], pipe._pending_tsumo_1p
    prefix = (4, 4)
    queue.append(prefix)
    old = runtime.histories["1P"]
    state = {"fired": False}
    def fault(frame: Any, event: str, arg: Any) -> Any:
        if (not state["fired"] and event == "line" and frame.f_code.co_name == "native_enqueue"
                and frame.f_lineno == target_line):
            state["fired"] = True
            raise error_type("test_only_one_fault")
        return fault
    previous = sys.gettrace()
    try:
        sys.settrace(fault)
        with pytest.raises(error_type, match="test_only"):
            packet(controller, pipe, 32698, (3, 5))
    finally:
        sys.settrace(previous)
    assert state["fired"] and pipe._pending_tsumo_1p is queue and queue[0] is prefix
    assert list(queue) == [prefix] and runtime.histories["1P"] is old
    assert pipe._landing_pending_1p is None and pipe._last_consumed_color_1p is None
    packet(controller, pipe, 32698, (3, 5))
    assert list(queue) == [prefix, (5, 5)]


@pytest.mark.parametrize("change", ["identity", "prefix", "length", "metadata"])
def test_recovery_never_claims_other_change_restored(unit: Any, change: str) -> None:
    pipe, _, controller = unit
    seed(unit)
    old = controller.instances[id(pipe)].histories["1P"]
    pipe._pending_tsumo_1p.append((4, 4))
    transaction = subject.EnqueueTransaction(pipe, "1P", old, subject.History(accepted=(3, 5)), 32698)
    if change == "identity":
        pipe._pending_tsumo_1p = deque(transaction.queue)
    elif change == "prefix":
        transaction.queue[0] = (2, 2)
    elif change == "length":
        transaction.queue.extend([(2, 2), (3, 3)])
    else:
        pipe._last_consumed_color_1p = (1, 1)
    with pytest.raises(subject.RecoveryFailed):
        transaction.restore()


def test_bounded_deque_rejected_before_eviction(unit: Any) -> None:
    pipe, _, controller = unit
    seed(unit)
    pipe._pending_tsumo_1p = deque([(4, 4)], maxlen=1)
    with pytest.raises(ValueError):
        packet(controller, pipe, 32698, (3, 5))
    assert list(pipe._pending_tsumo_1p) == [(4, 4)]


def test_reentry_and_other_thread_refused(unit: Any) -> None:
    pipe, _, controller = unit
    controller.begin(pipe, 32698, 32698 / 60)
    with pytest.raises(ValueError):
        controller.begin(pipe, 32698, 32698 / 60)
    controller.end(False)
    errors = []
    def other() -> None:
        try:
            controller.begin(pipe, 32698, 32698 / 60)
        except ValueError as error:
            errors.append(str(error))
    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    assert errors


def test_committed_state_not_rolled_back_on_receipt_error(unit: Any, monkeypatch: Any) -> None:
    pipe, rec, controller = unit
    seed(unit)
    def fail(row: Any) -> None:
        raise RuntimeError("after_commit")
    monkeypatch.setattr(rec, "emit", fail)
    with pytest.raises(RuntimeError):
        packet(controller, pipe, 32698, (3, 5))
    assert list(pipe._pending_tsumo_1p) == [(5, 5)]
    assert controller.instances[id(pipe)].histories["1P"].accepted == (3, 5)


class CpuReader:
    def __init__(self, board: Any) -> None:
        self.board = board

    def read_both_boards(self, frame: Any, **kwargs: Any) -> tuple[Any, Any]:
        return self.board.copy(), self.board.copy()


class CpuMatch:
    def __init__(self, state: Any) -> None:
        self.state = state

    def detect(self, frame: Any) -> Any:
        return SimpleNamespace(state=self.state, bg_value=100.0, bg_saturation=50.0, samples=1)


@pytest.fixture
def real(frozen: Any, monkeypatch: Any) -> Any:
    import numpy as np
    from src.board import Board
    from src.match_state import MatchState
    from src.next_detector import NextDetector
    from src.next_slide_detector import NextSlideDetector
    rec, image = Recorder(), np.zeros((1080, 1920, 3), dtype=np.uint8)
    main = object.__new__(NextDetector)
    source = {"pair": (5, 5), "diff": 1.0, "calls": Counter(), "returns": [], "fail": False}
    def detect(detector: Any, frame: Any) -> Any:
        source["calls"]["next"] += 1
        result = actual_next(source["pair"])
        source["returns"].append(result)
        return result
    def slide(detector: Any, previous: Any, current: Any) -> Any:
        source["calls"]["slide"] += 1
        if source["fail"]:
            raise RuntimeError("actual_detector_fixture_failure")
        from src.next_slide_detector import SlideMotionResult
        result = SlideMotionResult(False, source["diff"], 8.0)
        source["returns"].append(result)
        return result
    monkeypatch.setattr(NextDetector, "detect_both", detect)
    monkeypatch.setattr(NextSlideDetector, "update", slide)
    pipe = frozen.RecognitionPipeline(image_reader=CpuReader(Board()), match_state_detector=CpuMatch(MatchState.IN_MATCH),
        score_ocr=None, chain_tracker_1p=None, chain_tracker_2p=None, next_detector=main, stable_frame_count=2)
    pipe._prev_frame = image.copy()
    original_update, original_reset = type(pipe).update, type(pipe).reset
    with contextlib.ExitStack() as stack:
        controller = subject.install(stack, frozen, rec)
        yield pipe, controller, rec, image, source
    assert type(pipe).update is original_update and type(pipe).reset is original_reset


def drive(real: Any, frame: int, pair: tuple[int, int], diff: float = 1.0) -> Any:
    pipe, _, _, image, source = real
    source.update(pair=pair, diff=diff)
    return pipe.update(frame, frame / 60, image)


def test_actual_frozen_update_calls_once_and_preserves_global(real: Any) -> None:
    pipe, controller, rec, _, source = real
    drive(real, 32696, (5, 5))
    drive(real, 32698, (2, 3), 63.0)
    drive(real, 32700, (3, 5))
    assert source["calls"] == Counter(next=3, slide=6)
    assert pipe._last_seen_next_1p == (3, 5) and pipe._last_seen_next_2p == (3, 5)
    assert [row["quiet"] for row in rec.rows] == [True, True, False, False, True, True]
    assert controller.active is None
    assert all(not row["production_permission"] for row in rec.rows)


def test_actual_update_metadata_consumption_not_reinjected(real: Any) -> None:
    pipe, _, _, _, _ = real
    drive(real, 32696, (5, 5))
    drive(real, 32698, (3, 5))
    # 原本P7/graceによって消費済みなら次の同NEXTで戻らない。
    before = (pipe._landing_pending_1p, pipe._landing_pending_2p)
    drive(real, 32700, (3, 5))
    assert (pipe._landing_pending_1p, pipe._landing_pending_2p) == before


def test_actual_slide_exception_is_missing_not_quiet(real: Any) -> None:
    pipe, controller, rec, _, source = real
    drive(real, 32696, (5, 5))
    source["fail"] = True
    drive(real, 32698, (3, 5))
    assert [row["quiet"] for row in rec.rows[-2:]] == [False, False]
    assert controller.instances[id(pipe)].histories["1P"].accepted == (5, 5)


def test_actual_update_partial_fault_restore_and_same_side_retry(real: Any) -> None:
    pipe, controller, _, _, _ = real
    drive(real, 32696, (5, 5))
    queue, old = pipe._pending_tsumo_2p, controller.instances[id(pipe)].histories["2P"]
    fired = []
    def fault(frame: Any, event: str, arg: Any) -> Any:
        if not fired and event == "line" and frame.f_code.co_name == "native_enqueue" and frame.f_lineno == 5147:
            fired.append(True)
            raise RuntimeError("actual_update_test_fault")
        return fault
    previous = sys.gettrace()
    try:
        sys.settrace(fault)
        with pytest.raises(RuntimeError, match="actual_update_test_fault"):
            drive(real, 32698, (3, 5))
    finally:
        sys.settrace(previous)
    assert fired and pipe._pending_tsumo_2p is queue and not queue
    assert controller.instances[id(pipe)].histories["2P"] is old
    assert list(pipe._pending_tsumo_1p) == [(5, 5)]  # 成功sideは戻さない。
    drive(real, 32698, (3, 5))
    assert list(pipe._pending_tsumo_1p) == [(5, 5)]
    assert list(pipe._pending_tsumo_2p) == [(5, 5)]


def test_install_after_history_wrapper_refused(frozen: Any) -> None:
    cls, original = frozen.RecognitionPipeline, frozen.RecognitionPipeline.update
    def wrapper(self: Any, *args: Any) -> Any:
        return original(self, *args)
    with contextlib.ExitStack() as stack:
        stack.callback(setattr, cls, "update", original)
        cls.update = wrapper
        with pytest.raises(RuntimeError, match="元frozen"):
            subject.install(stack, frozen, Recorder())


def test_later_global_patch_reaches_actual_update(real: Any, monkeypatch: Any) -> None:
    pipe, _, _, _, source = real
    from src import recognition_pipeline as module
    original_enum = module.MatchState
    assert type(pipe).update.__globals__ is vars(module)
    monkeypatch.setattr(module, "MatchState", SimpleNamespace(IN_MATCH=object(), NOT_IN_MATCH=original_enum.NOT_IN_MATCH))
    result = drive(real, 32696, (5, 5))
    assert result.is_match_active is False and source["calls"]["next"] == 0
    sentinel = object()
    monkeypatch.setattr(module, "_is_game_event_chain_exit", sentinel)
    assert type(pipe).update.__globals__["_is_game_event_chain_exit"] is sentinel


def test_actual_reset_failure_never_authenticates_epoch(real: Any, monkeypatch: Any) -> None:
    pipe, controller, _, _, _ = real
    drive(real, 32696, (5, 5))
    runtime = controller.instances[id(pipe)]
    epoch = runtime.histories["1P"].epoch
    def fail() -> None:
        raise RuntimeError("original_reset_failed")
    monkeypatch.setattr(pipe._sm_1p, "reset", fail)
    with pytest.raises(RuntimeError, match="original_reset_failed"):
        pipe.reset()
    assert runtime.reset_status == "failed" and runtime.histories["1P"].epoch == epoch
    assert runtime.histories["1P"].accepted is None
    with pytest.raises(ValueError, match="失敗reset"):
        drive(real, 32698, (3, 5))


def test_swallowed_capture_contract_error_still_fails_after_catch(real: Any, monkeypatch: Any) -> None:
    pipe, controller, _, _, _ = real
    original = controller.main_return
    def twice(owner: Any, frame: int, time_sec: float, result: Any) -> Any:
        original(owner, frame, time_sec, result)
        return original(owner, frame, time_sec, result)
    monkeypatch.setattr(controller, "main_return", twice)
    with pytest.raises(ValueError, match="主NEXT二重"):
        drive(real, 32696, (5, 5))
    assert not pipe._pending_tsumo_1p and controller.active is None


def test_actual_result_construction_failure_not_completed(real: Any, monkeypatch: Any) -> None:
    from src import recognition_pipeline as module
    pipe, controller, _, _, _ = real
    def fail(**kwargs: Any) -> Any:
        raise RuntimeError("result_constructor_failed")
    monkeypatch.setattr(module, "PipelineResult", fail)
    with pytest.raises(RuntimeError, match="result_constructor_failed"):
        drive(real, 32696, (5, 5))
    assert controller.instances[id(pipe)].completed is None


def test_actual_settle_metadata_clear_and_collector_drain_once(real: Any, frozen: Any) -> None:
    """SM detectorだけ人工STABLE信号。元update/SM/settle/clear/getter/drainを実行する。"""
    from src.board import Board
    from src.board_state_machine import BoardState
    pipe, _, _, _, _ = real
    drive(real, 32696, (5, 5))
    drive(real, 32698, (3, 5))
    class StableSignal:
        def detect(self, context: Any, signals: Any) -> Any:
            return BoardState.STABLE
    for sm in (pipe._sm_1p, pipe._sm_2p):
        sm._ctx.state = BoardState.TSUMO_FALL
        sm._ctx.confirmed_board = Board()
        sm._detectors = [StableSignal()]
    pipe._reader.board.set(12, 0, 5)
    pipe._reader.board.set(12, 1, 5)
    first_move, queue = pipe._first_move_sec_1p, pipe._pending_tsumo_1p
    drive(real, 32700, (3, 5))
    assert pipe._pending_tsumo_1p is queue and not queue
    assert pipe.tsumo_count("1P") == 1 and pipe._last_consumed_color_1p is None
    assert pipe._landing_pending_1p is None
    calls, state = [], frozen._SideState()
    tracker = SimpleNamespace(on_tsumo_settled=lambda key, time_sec: calls.append((key, time_sec)))
    frozen._drain_ojama_by_tsumo_delta_lean(tracker, "1P", state, pipe.tsumo_count("1P"), 545.0)
    drive(real, 32702, (3, 5))
    frozen._drain_ojama_by_tsumo_delta_lean(tracker, "1P", state, pipe.tsumo_count("1P"), 545.1)
    assert len(calls) == 1 and pipe._last_consumed_color_1p is None
    assert pipe._first_move_sec_1p == (first_move if first_move is not None else 32700 / 60)


def test_install_failure_restores_added_global(frozen: Any, monkeypatch: Any) -> None:
    original = frozen.RecognitionPipeline.update
    def fail(*args: Any) -> Any:
        raise RuntimeError("compile_fixture_failure")
    monkeypatch.setattr(subject, "_transformed", fail)
    with pytest.raises(RuntimeError):
        with contextlib.ExitStack() as stack:
            subject.install(stack, frozen, Recorder())
    assert "__next_live" not in original.__globals__
    assert frozen.RecognitionPipeline.update is original


def test_guard_ast_scope_and_function_lengths() -> None:
    for path, digest in subject.REQUIRED_INPUT_SHA256.items():
        subject.BASE._read_verified(Path(path), digest)
    for path in (Path(subject.__file__), Path(__file__)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert all(node.end_lineno - node.lineno + 1 <= 50 for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))


@pytest.mark.parametrize("value", [SimpleNamespace(), None, (5, 5), True])
def test_actual_main_immutable_type_refuses_foreign_values(unit: Any, value: Any) -> None:
    pipe, _, controller = unit
    controller.begin(pipe, 32698, 32698 / 60)
    with pytest.raises(ValueError, match="immutable"):
        controller.main_return(pipe, 32698, 32698 / 60, value)
    assert controller.active.error is not None and not pipe._pending_tsumo_1p
    controller.end(False)


@pytest.mark.parametrize("side,pair", [("1P", (3, 5)), ("2P", (5, 5))])
def test_other_side_or_changed_value_not_accepted(unit: Any, side: str, pair: Any) -> None:
    from src.next_slide_detector import SlideMotionResult
    pipe, _, controller = unit
    controller.begin(pipe, 32698, 32698 / 60)
    controller.main_return(pipe, 32698, 32698 / 60, actual_next((5, 5), (3, 5)))
    controller.slide_return(pipe, side, 32698, 32698 / 60, SlideMotionResult(False, 1.0, 8.0))
    with pytest.raises(ValueError, match="実NEXT return"):
        controller.enqueue(pipe, side, 32698, 32698 / 60, True, pair)
    assert not getattr(pipe, "_pending_tsumo_" + side.lower())
    controller.end(False)


def test_actual_capture_other_invocation_refused(unit: Any) -> None:
    pipe, _, controller = unit
    controller.begin(pipe, 32698, 32698 / 60)
    with pytest.raises(ValueError, match="clock/instance"):
        controller.main_return(pipe, 32696, 32696 / 60, actual_next((5, 5)))
    controller.end(False)


def test_real_slide_result_builtin_types(frozen: Any) -> None:
    import numpy as np
    from src.next_slide_detector import NextSlideDetector, SlideMotionResult
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = NextSlideDetector().update(image, image)
    assert type(result) is SlideMotionResult and type(result.slide_motion) is bool
    assert type(result.diff_score) is float and type(result.threshold_used) is float


def test_fixed_32494_busy_internal_reset_calls_original_once(real: Any, monkeypatch: Any) -> None:
    """保存32494と同じ空FIFO/score0。OCRは人工入力、元4868 branch/resetは実行。"""
    from src import recognition_pipeline as module
    from src.match_state import MatchState
    pipe, controller, _, _, source = real
    pipe._match_detector.state = MatchState.NOT_IN_MATCH
    pipe._enable_match_start_full_clear = True
    pipe._enable_score_reset_strict = True
    pipe._score_reset_boundary_streak = module.SCORE_RESET_BOUNDARY_DEBOUNCE_FRAMES - 1
    resets = []
    for side in subject.SIDES:
        tracker = SimpleNamespace(last_score=0, reset=lambda: resets.append(True))
        setattr(pipe, "_score_tracker_" + side.lower(), tracker)
    monkeypatch.setattr(pipe, "_update_score_tracker", lambda *args, **kwargs: (0, 0, None, None))
    result = drive(real, subject.FIRST_FRAME, (4, 4))
    runtime = controller.instances[id(pipe)]
    assert len(resets) == 2 and result.is_match_active is False
    assert runtime.reset_status == "software_reset_observed_not_game_proof"
    assert all(value.epoch == 1 for value in runtime.histories.values())
    assert not pipe._pending_tsumo_1p and not pipe._pending_tsumo_2p
    assert source["calls"]["next"] == 0 and runtime.completed == subject.FIRST_FRAME
