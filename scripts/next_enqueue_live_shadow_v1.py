"""固定同期runtimeのNEXT会計だけを接続する開発shadow。本番許可は発行しない。

同sideの未完了enqueueのみ限定撤回する。Counter/first_move/下流消費は原本。
固定非消費traceは許容するが、外部thread/任意callbackへの不可視性、撤回中の
MemoryError/再中断までの一般原子性は保証しない。GPU/初手修復は行わない。
"""
from __future__ import annotations

import ast
import contextlib
import functools
import inspect
import math
import threading
from collections import deque
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from scripts import next_enqueue_freshness_shadow_v1 as cpu

BASE = cpu.base
FIRST_FRAME, LAST_FRAME, STRIDE = BASE.RESET_FRAME, 36298, 2
SIDES = BASE.SIDES
PREFIX = "next_enqueue_live_"
CLOCK_TOLERANCE = 1e-8
PAIR_SIZE = 2
ANCHORS = {"update": 4175, "internal_reset": 4868, "main": 5061,
           "slide_1p": 5077, "slide_2p": 5089, "enqueue_1p": 5117, "enqueue_2p": 5135}
GUARD_PATHS = (
    "scripts/next_enqueue_live_shadow_v1.py",
    "tests/test_next_enqueue_live_shadow_v1.py",
    "scripts/next_enqueue_freshness_shadow_v1.py",
    "tests/test_next_enqueue_freshness_shadow_v1.py",
    "docs/agent_coordination/NEXT_ENQUEUE_LIVE_WIRING_PLAN_2026-09-08.md",
)
REQUIRED_INPUT_SHA256 = {
    str(BASE.PIPELINE): BASE.PIPELINE_SHA,
    str(BASE.ROOT / "scripts/next_enqueue_freshness_shadow_v1.py"):
        "4b05a8254607bd58212344c78a712633d5b49427d42cee0d91c00366fe363ead",
    str(BASE.ROOT / "tests/test_next_enqueue_freshness_shadow_v1.py"):
        "00fd704ef1d5d8227ed52261ba99a611ae7c99bfec282c67428ed661dcd6d4ab",
    str(BASE.ROOT / "docs/agent_coordination/NEXT_ENQUEUE_LIVE_WIRING_PLAN_2026-09-08.md"):
        "e72c16ac572676d3b03893c725d8d334515407eb745fc8fe5411eb47110e6be4",
}


class RecoveryFailed(RuntimeError):
    """当該差分だけの撤回が証明できず、正常完了を拒否する。"""


def _pair(value: Any) -> bool:
    return (type(value) is tuple and len(value) == PAIR_SIZE
            and all(type(color) is int and color in BASE.COLORS for color in value))


def _clock(frame: int, time_sec: float) -> None:
    if (type(frame) is not int or frame < 0 or type(time_sec) not in (int, float)
            or not math.isfinite(time_sec) or abs(time_sec - frame / BASE.FPS) > CLOCK_TOLERANCE):
        raise ValueError("実frame/timeが不正です")


def _actual_next_result(result: Any) -> bool:
    """固定immutable DTOだけ。propertyの再読tuple identityは要求しない。"""
    from src.next_detector import NextDetectionBothResult, NextDetectionResult
    if type(result) is not NextDetectionBothResult:
        return False
    return all(type(value) is NextDetectionResult and all(type(color) is int for color in
               (value.next_top, value.next_bot, value.dnext_top, value.dnext_bot))
               for value in (result.p1, result.p2))


@dataclass(frozen=True)
class History:
    """会計の受理履歴だけ。FIFO/Counter/消費metadataを複製しない。"""

    epoch: int = 0
    accepted: tuple[int, int] | None = None
    clock: int | None = None
    payload: tuple[Any, ...] | None = None


@dataclass
class Runtime:
    """一つのpipelineを同期呼出するthreadに束縛する。"""

    pipe: Any
    histories: dict[str, History]
    thread: int
    busy: bool = False
    completed: int | None = None
    reset_status: str = "not_observed"


@dataclass
class Invocation:
    runtime: Runtime
    frame: int
    time_sec: float
    main: Any = None
    slides: dict[str, Any] | None = None
    error: str | None = None
    accounting_started: bool = False


class EnqueueTransaction:
    """原本enqueue中だけ存在する、末尾一個と属性三種の限定撤回境界。"""

    def __init__(self, pipe: Any, side: str, old: History, new: History, frame: int) -> None:
        suffix = side.lower()
        self.pipe, self.old, self.new, self.accepted = pipe, old, new, old.accepted
        self.queue_name = f"_pending_tsumo_{suffix}"
        self.names = (f"_landing_pending_{suffix}", f"_last_consumed_color_{suffix}")
        self.queue = vars(pipe)[self.queue_name]
        if type(self.queue) is not deque or self.queue.maxlen is not None:
            raise ValueError("通常unbounded dequeだけが対象です")
        self.prefix = tuple(self.queue)
        if not all(_pair(value) for value in self.prefix):
            raise ValueError("FIFO内に可変または未対応pairがあります")
        self.before = tuple(vars(pipe)[name] for name in self.names)
        pending, color = self.before
        if not (color is None or _pair(color)):
            raise ValueError("元colorが不正です")
        if pending is not None and not (type(pending) is tuple and len(pending) == PAIR_SIZE
                and type(pending[0]) is int and _pair(pending[1])):
            raise ValueError("元pendingが不正です")
        self.consumed = old.accepted if old.accepted != new.accepted else None
        self.prepared_landing = (frame, self.consumed) if self.consumed is not None else pending
        self.expected = (self.prepared_landing, self.consumed) if self.consumed is not None else self.before

    def verify(self, *, success: bool) -> None:
        """prefixは色一致でなくobject identityを保ち、他の変更を撤回しない。"""
        queue = vars(self.pipe).get(self.queue_name)
        if queue is not self.queue or len(queue) not in (len(self.prefix), len(self.prefix) + 1):
            raise RecoveryFailed("FIFO identity/lengthが変化しました")
        if any(queue[index] is not value for index, value in enumerate(self.prefix)):
            raise RecoveryFailed("元FIFO prefixが変化しました")
        appended = len(queue) == len(self.prefix) + 1
        if appended and (self.consumed is None or queue[-1] is not self.consumed):
            raise RecoveryFailed("対象でない末尾を撤回しません")
        for name, old, new in zip(self.names, self.before, self.expected, strict=True):
            current = vars(self.pipe).get(name)
            if current is not old and current is not new:
                raise RecoveryFailed("対象でないmetadata変更を撤回しません")
            if success and current is not new:
                raise RecoveryFailed("metadataが予定値に到達していません")
        if success and appended != (self.consumed is not None):
            raise RecoveryFailed("enqueue回数が予定と違います")

    def restore(self) -> None:
        """下流未到達の同side差分だけ撤回。復旧自体の異常も隠さない。"""
        self.verify(success=False)
        if len(self.queue) == len(self.prefix) + 1:
            self.queue.pop()
        for name, value in zip(self.names, self.before, strict=True):
            vars(self.pipe)[name] = value
        if len(self.queue) != len(self.prefix):
            raise RecoveryFailed("FIFO撤回後の長さが不正です")


class NextEnqueueController:
    """実戻りを同invocationへ捕捉し、通常body一回と限定撤回を管理する。"""

    def __init__(self, cls: type, rec: Any, programs: dict[str, Any]) -> None:
        self.cls, self.rec, self.programs = cls, rec, programs
        self.instances: dict[int, Runtime] = {}
        self.active: Invocation | None = None
        self.thread = threading.get_ident()
        self.transform_receipt: dict[str, Any] = {}

    def _runtime(self, pipe: Any) -> Runtime:
        if (type(pipe) is not self.cls or type(vars(pipe)) is not dict
                or self.cls.__setattr__ is not object.__setattr__
                or self.cls.__getattribute__ is not object.__getattribute__):
            raise ValueError("固定通常classとplain属性だけが対象です")
        if threading.get_ident() != self.thread:
            raise ValueError("別threadのpipeline呼出を拒否します")
        for side in SIDES:
            for key in ("pending_tsumo", "landing_pending", "last_consumed_color"):
                name = f"_{key}_{side.lower()}"
                if inspect.getattr_static(self.cls, name, None) is not None:
                    raise ValueError("会計属性descriptorを拒否します")
        if id(pipe) not in self.instances:
            self.instances[id(pipe)] = Runtime(pipe, {side: History() for side in SIDES}, self.thread)
        return self.instances[id(pipe)]

    def begin(self, pipe: Any, frame: int, time_sec: float) -> None:
        _clock(frame, time_sec)
        runtime = self._runtime(pipe)
        if runtime.reset_status in ("pending", "failed"):
            raise ValueError("未完了/失敗reset後の履歴は再利用しません")
        if self.active is not None or runtime.busy:
            raise ValueError("再入updateを拒否します")
        if runtime.completed is not None and frame <= runtime.completed:
            raise ValueError("完了済みupdateの再走を拒否します")
        runtime.busy = True
        self.active = Invocation(runtime, frame, float(time_sec), slides={})

    def end(self, success: bool) -> None:
        invocation = self.active
        if invocation is not None:
            if success:
                invocation.runtime.completed = invocation.frame
            invocation.runtime.busy = False
            self.active = None

    def _invocation(self, pipe: Any, frame: int, time_sec: float) -> Invocation:
        _clock(frame, time_sec)
        value = self.active
        if (value is None or value.runtime.pipe is not pipe or value.frame != frame
                or value.time_sec != time_sec or threading.get_ident() != self.thread):
            raise ValueError("実updateと捕捉clock/instance/threadが違います")
        return value

    def main_return(self, pipe: Any, frame: int, time_sec: float, result: Any) -> Any:
        invocation = self._invocation(pipe, frame, time_sec)
        if invocation.main is not None:
            invocation.error = "主NEXT二重呼出"
            raise ValueError("主NEXTの追加呼出を拒否します")
        if not _actual_next_result(result):
            invocation.error = "主NEXTの固定immutable実返却型が不正"
            raise ValueError(invocation.error)
        invocation.main = result
        return result

    def slide_return(self, pipe: Any, side: str, frame: int, time_sec: float, result: Any) -> Any:
        from src.next_slide_detector import SlideMotionResult
        invocation = self._invocation(pipe, frame, time_sec)
        if side not in SIDES or side in invocation.slides:
            invocation.error = "slide二重呼出/別side"
            raise ValueError("slideのside/呼出回数が不正です")
        if type(result) is not SlideMotionResult:
            invocation.error = "slideの固定immutable実返却型が不正"
            raise ValueError(invocation.error)
        invocation.slides[side] = result
        return result

    def reset(self, pipe: Any) -> Runtime:
        runtime = self._runtime(pipe)
        if runtime.busy:
            invocation = self.active
            frame = inspect.currentframe()
            while frame is not None and not (frame.f_code.co_name == "update"
                    and BASE.Path(frame.f_code.co_filename).resolve() == BASE.PIPELINE.resolve()):
                frame = frame.f_back
            if (invocation is None or invocation.runtime is not runtime or invocation.accounting_started
                    or invocation.main is not None or frame is None or frame.f_lineno != ANCHORS["internal_reset"]):
                raise ValueError("固定pre-NEXT reset以外の内部resetを拒否します")
        runtime.reset_status = "pending"
        runtime.histories = {side: History(value.epoch) for side, value in runtime.histories.items()}
        return runtime

    def reset_finished(self, runtime: Runtime, success: bool) -> None:
        """失敗resetを成功epoch/新ゲームproofへ昇格させない。"""
        if success:
            runtime.histories = {side: History(value.epoch + 1) for side, value in runtime.histories.items()}
        runtime.reset_status = "software_reset_observed_not_game_proof" if success else "failed"

    def _quiet(self, invocation: Invocation, side: str, pair: Any) -> tuple[bool, tuple[Any, ...]]:
        result = invocation.slides.get(side)
        if result is None or invocation.main is None:
            return False, ("missing_actual_call",)
        observed = getattr(invocation.main, "p1" if side == "1P" else "p2").next_pair
        # 実return objectは同invocationへ束縛済み。固定propertyは毎回新tupleを返す。
        if pair is not None and pair != observed:
            raise ValueError("実NEXT returnと入力pairが一致しません")
        pulse, diff, threshold = result.slide_motion, result.diff_score, result.threshold_used
        if (type(pulse) is not bool or type(diff) not in (int, float)
                or type(threshold) not in (int, float) or not math.isfinite(diff)
                or not math.isfinite(threshold) or diff < 0 or threshold <= 0):
            raise ValueError("実slideの有限値契約に違反しています")
        return _pair(pair) and not pulse and diff < threshold, (pulse, diff, threshold)

    def _global_next(self, pipe: Any, side: str, active: bool, pair: Any) -> None:
        """元global代入と同じint変換・色条件・新tuple生成。会計quietは条件にしない。"""
        if active and pair is not None:
            top, bottom = int(pair[0]), int(pair[1])
            if top in BASE.COLORS and bottom in BASE.COLORS:
                vars(pipe)[f"_last_seen_next_{side.lower()}"] = (top, bottom)

    def enqueue(self, pipe: Any, side: str, frame: int, time_sec: float,
                active: bool, pair: Any) -> None:
        invocation = self._invocation(pipe, frame, time_sec)
        if invocation.error is not None:
            raise ValueError(invocation.error)
        invocation.accounting_started = True
        if not FIRST_FRAME <= frame <= LAST_FRAME:
            self.programs["legacy_" + side](pipe, active, frame, pair, None)
            return
        if type(active) is not bool:
            raise ValueError("実activeがboolではありません")
        if pair is not None and (type(pair) is not tuple or len(pair) != PAIR_SIZE
                                 or any(type(color) is not int for color in pair)):
            raise ValueError("会計履歴に可変/未対応NEXTを保持しません")
        runtime, old = invocation.runtime, invocation.runtime.histories[side]
        quiet, detail = self._quiet(invocation, side, pair) if active else (False, ("inactive",))
        payload = (active, pair, detail, old.epoch)
        if old.clock == frame:
            if payload != old.payload:
                raise ValueError("同frame異payloadを拒否します")
            return
        if old.clock is not None and frame - old.clock != STRIDE:
            raise ValueError("会計履歴の欠落/逆行を拒否します")
        accepted = pair if quiet else old.accepted if active else None
        new = History(old.epoch, accepted, frame, payload)
        self._apply(runtime, side, frame, active, pair if quiet else None, old, new)
        self._global_next(pipe, side, active, pair)
        self.rec.emit({"kind": PREFIX + "decision", "side": side, "accepted": accepted,
                       "frame_idx": frame, "time_sec": time_sec, "quiet": quiet,
                       "physical_placement_verified": False, "production_permission": False})

    def _apply(self, runtime: Runtime, side: str, frame: int, active: bool,
               pair: Any, old: History, new: History) -> None:
        if not active:
            runtime.histories[side] = new
            return
        if old.accepted is None and vars(runtime.pipe)[f"_pending_tsumo_{side.lower()}"]:
            raise ValueError("未知履歴を既存非空FIFOへ接合しません")
        transaction = EnqueueTransaction(runtime.pipe, side, old, new, frame)
        try:
            self.programs[side](runtime.pipe, active, frame, pair, transaction)
            transaction.verify(success=True)
            if runtime.histories[side] is not old:
                raise RecoveryFailed("受理historyが途中変更されました")
            runtime.histories[side] = new
        except BaseException:
            if runtime.histories[side] is not old and runtime.histories[side] is not new:
                raise RecoveryFailed("別historyを上書き復旧しません")
            transaction.restore()
            runtime.histories[side] = old
            raise


def _source_tree() -> ast.Module:
    return ast.parse(BASE._read_verified(BASE.PIPELINE, BASE.PIPELINE_SHA))


def _side_body(tree: ast.Module, side: str) -> ast.If:
    first = ANCHORS["enqueue_" + side.lower()]
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.If) and node.lineno == first]
    if len(nodes) != 1:
        raise RuntimeError("元enqueue ASTが一意ではありません")
    return nodes[0]


class _AccountingReferences(ast.NodeTransformer):
    """原本bodyの会計履歴と事前準備tupleだけを限定差替えする。"""

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if node.attr.startswith("_last_seen_next_"):
            return ast.copy_location(ast.Attribute(ast.Name("tx", ast.Load()), "accepted", node.ctx), node)
        return self.generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple) -> ast.AST:
        if ast.unparse(node) == "(frame_idx, consumed)":
            return ast.copy_location(ast.Attribute(ast.Name("tx", ast.Load()), "prepared_landing", ast.Load()), node)
        return self.generic_visit(node)


def _programs(namespace: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for side in SIDES:
        for legacy in (False, True):
            body = _side_body(_source_tree(), side)
            if not legacy:
                body = _AccountingReferences().visit(body)
            pair_name = "next_pair_" + side.lower()
            arguments = ast.arguments([], [ast.arg(name) for name in
                ("self", "is_active", "frame_idx", pair_name, "tx")], None, [], [], None, [])
            valid = ast.Assign([ast.Name("VALID_PUYO_COLORS", ast.Store())],
                               ast.Set([ast.Constant(color) for color in BASE.COLORS]))
            function = ast.FunctionDef("native_enqueue", arguments, [valid, body], [], None)
            module = ast.fix_missing_locations(ast.Module([function], []))
            values = dict(namespace)
            exec(compile(module, str(BASE.PIPELINE), "exec"), values)
            result[("legacy_" if legacy else "") + side] = values["native_enqueue"]
    return result


class _UpdateTransformer(ast.NodeTransformer):
    """元updateの実return捕捉と左右enqueue二箇所だけを接続する。"""

    def __init__(self) -> None:
        self.matched: list[str] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:
        line = node.lineno
        if line not in (ANCHORS["main"], ANCHORS["slide_1p"], ANCHORS["slide_2p"]):
            return self.generic_visit(node)
        text = ast.unparse(node.func)
        expected = {ANCHORS["main"]: "self._next_detector.detect_both",
                    ANCHORS["slide_1p"]: "self._slide_detector_1p.update",
                    ANCHORS["slide_2p"]: "self._slide_detector_2p.update"}[line]
        if text != expected:
            raise RuntimeError("実detector ASTが固定版と違います")
        name = "main_return" if line == ANCHORS["main"] else "slide_return"
        args = [ast.Name("self", ast.Load())]
        if line != ANCHORS["main"]:
            args.append(ast.Constant("1P" if line == ANCHORS["slide_1p"] else "2P"))
        args.extend([ast.Name("frame_idx", ast.Load()), ast.Name("time_sec", ast.Load()), node])
        self.matched.append(name + str(line))
        return ast.copy_location(ast.Call(ast.Attribute(ast.Name("__next_live", ast.Load()), name, ast.Load()), args, []), node)

    def visit_If(self, node: ast.If) -> ast.AST:
        if node.lineno not in (ANCHORS["enqueue_1p"], ANCHORS["enqueue_2p"]):
            return self.generic_visit(node)
        side = "1P" if node.lineno == ANCHORS["enqueue_1p"] else "2P"
        args = [ast.Name("self", ast.Load()), ast.Constant(side), ast.Name("frame_idx", ast.Load()),
                ast.Name("time_sec", ast.Load()), ast.Name("is_active", ast.Load()),
                ast.Name("next_pair_" + side.lower(), ast.Load())]
        self.matched.append("enqueue" + side)
        call = ast.Call(ast.Attribute(ast.Name("__next_live", ast.Load()), "enqueue", ast.Load()), args, [])
        return ast.copy_location(ast.Expr(call), node)


def _transformed(original: Any, controller: NextEnqueueController) -> Any:
    tree = _source_tree()
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RecognitionPipeline")
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "update")
    transformer = _UpdateTransformer()
    function = transformer.visit(function)
    expected = ["main_return" + str(ANCHORS["main"]), "slide_return" + str(ANCHORS["slide_1p"]),
                "slide_return" + str(ANCHORS["slide_2p"]), "enqueue1P", "enqueue2P"]
    if transformer.matched != expected:
        raise RuntimeError("NEXT AST対象数が固定版と違います")
    prelude = ast.parse("__next_live.begin(self, frame_idx, time_sec)\n__next_success = False").body
    success = ast.Assign([ast.Name("__next_success", ast.Store())], ast.Constant(True))
    for index, node in enumerate(function.body):
        if isinstance(node, ast.Return):
            result = ast.Assign([ast.Name("__next_result", ast.Store())], node.value)
            node.value = ast.Name("__next_result", ast.Load())
            function.body[index:index] = [ast.copy_location(result, node), ast.copy_location(success, node)]
            break
    ending = ast.parse("__next_live.end(__next_success)").body
    function.body = prelude + [ast.Try(function.body, [], [], ending)]
    module = ast.fix_missing_locations(ast.Module([function], []))
    values = original.__globals__
    existed, previous = "update" in values, values.get("update")
    try:
        exec(compile(module, original.__code__.co_filename, "exec"), values)
        generated = values["update"]
    finally:
        if existed:
            values["update"] = previous
        else:
            values.pop("update", None)
    controller.transform_receipt = {"matched": transformer.matched, "original_settle_unchanged": True,
        "counter_written_by_adapter": False, "rollback_scope": "unfinished_same_side_enqueue_only",
        "external_thread_atomicity": False, "arbitrary_callback_supported": False,
        "trace_scope": "fixed_nonconsumer_trace_only; native_enqueue_has_source_lines_but_distinct_code"}
    return functools.update_wrapper(generated, original)


def install(stack: contextlib.ExitStack, collector: ModuleType, rec: Any) -> NextEnqueueController:
    """history wrapperがclosureを保持する前の実frozen updateへ接続する。"""
    for path, digest in REQUIRED_INPUT_SHA256.items():
        BASE._read_verified(BASE.Path(path), digest)
    cls, original = collector.RecognitionPipeline, collector.RecognitionPipeline.update
    if (BASE.Path(original.__code__.co_filename).resolve() != BASE.PIPELINE.resolve()
            or original.__code__.co_firstlineno != ANCHORS["update"] or hasattr(original, "__wrapped__")):
        raise RuntimeError("既存wrapperより前の元frozen updateが必要です")
    controller = NextEnqueueController(cls, rec, _programs(original.__globals__))
    if "__next_live" in original.__globals__:
        raise RuntimeError("未知のNEXT transformが既にあります")
    original.__globals__["__next_live"] = controller
    stack.callback(original.__globals__.pop, "__next_live", None)
    transformed, reset = _transformed(original, controller), cls.reset
    def reset_wrapper(pipe: Any, *args: Any, **kwargs: Any) -> Any:
        runtime = controller.reset(pipe)
        try:
            result = reset(pipe, *args, **kwargs)
        except BaseException:
            controller.reset_finished(runtime, False)
            raise
        controller.reset_finished(runtime, True)
        return result
    stack.callback(setattr, cls, "update", original)
    stack.callback(setattr, cls, "reset", reset)
    cls.update, cls.reset = transformed, reset_wrapper
    return controller
