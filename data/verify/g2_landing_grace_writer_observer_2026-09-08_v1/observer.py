"""固定grace writerだけを追加観測する。認識の値・分岐は変更しない。"""
from __future__ import annotations

import contextlib
import functools
import inspect
import sys
from pathlib import Path
from types import FrameType, SimpleNamespace
from typing import Any, Callable

from scripts import diagnose_video38_post_chain_grace_shadow_v1 as lifecycle

PIPELINE_SHA = "6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02"
INFER_SHA = "412a15120db0d6e9c10f38ad8d5fa8dff1dfa05f05f3d1a14b71f4ce10610b82"
LIFECYCLE_SHA = "53bd2f9db63a65bbe87720fc3e530b6fae8a237e4754e655d5d8a1804ccecc08"
PREFIX = "landing_grace_writer_"


def validate_code(entry: Any, original: Callable[..., Any]) -> tuple[Any, Any]:
    """固定sourceと原本codeを同時に束縛し、再compileや別runtimeを拒否する。"""
    step = inspect.unwrap(original)
    infer = inspect.unwrap(step.__globals__["infer_placement"])
    paths = (entry.base.SNAPSHOT / "src/recognition_pipeline.py",
             entry.base.SNAPSHOT / "src/placement_inferrer.py")
    guards = {str(path): digest for path, digest in zip(paths, (PIPELINE_SHA, INFER_SHA), strict=True)}
    guards[str(Path(lifecycle.__file__).resolve())] = LIFECYCLE_SHA
    entry.base.assert_unchanged(guards)
    for function, path in zip((step, infer), paths, strict=True):
        if Path(function.__code__.co_filename).resolve() != path.resolve():
            raise RuntimeError("writer観測対象の実code pathが固定sourceと異なります")
    module = sys.modules[step.__module__]
    if module.__dict__.get("infer_placement") is not step.__globals__["infer_placement"]:
        raise RuntimeError("writerの実globalsが原本と異なります")
    return step.__code__, infer.__code__


class WriterTrace(lifecycle.LifecycleTrace):
    """既存AST範囲traceを再利用し、完了代入の前後を別sinkへ送る。"""
    def __init__(self, entry: Any, rec: Any, sink: Any, codes: tuple[Any, Any]) -> None:
        super().__init__(SimpleNamespace(step_code=codes[0]))
        self.entry, self.clock, self.sink, self.infer_code = entry, rec, sink, codes[1]
        self.scope: dict[str, Any] = {}
        self.previous: Any = None
        self.step_frame: FrameType | None = None
        self.step_calls, self.infer_calls = 0, 0

    def emit(self, suffix: str, **fields: Any) -> None:
        """同scope時計からだけ生成する。source行は認証権ではない。"""
        self._check_clock()
        self.sink.emit({"kind": PREFIX + suffix, "frame_idx": self.scope["frame"],
            "time_sec": self.scope["time"], "side": self.scope["side"], **fields,
            "current_publication_allowed": False, "accounting_commit_allowed": False})

    def _check_clock(self) -> None:
        """外部rec時計と実step時計を厳密照合する。"""
        if not self.scope or (self.clock.frame, self.clock.time_sec) != (self.scope["frame"], self.scope["time"]):
            raise RuntimeError("writer scope clock不一致")

    def _snapshot(self, frame: FrameType) -> dict[str, Any]:
        """mutable値を即コピーし、FIFOを消費も復元もしない。"""
        local, pipe, side = frame.f_locals, self.scope["pipe"], self.scope["side"]
        ctx = local.get("ctx", self.scope["sm"].context)
        grace = getattr(pipe, f"_landing_grace_{side.lower()}")
        board = self.entry.base.board_value
        return {"state": getattr(ctx.state, "value", ctx.state),
            "prev_state": getattr(local.get("prev_state"), "value", None),
            "prev_confirmed": board(local.get("prev_confirmed")),
            "confirmed": board(ctx.confirmed_board), "pending_board": board(ctx.pending_board),
            "grace_present": grace is not None,
            "grace_board": None if grace is None else board(grace[1]),
            "grace_deadline_frame": None if grace is None else grace[0],
            "grace_deadline_time": None if grace is None else grace[2],
            "accounting": self.entry.previous.history.accounting_snapshot(pipe, side),
            "infer_calls_so_far": self.infer_calls,
            "local_evidence": {name: {"present": name in local,
                "value": board(local[name]) if name in local and name != "chain_count"
                else self.entry.base.json_value(local.get(name))}
                for name in ("inferred_landing", "inferred_b", "chain_count")}}

    def _record(self, frame: FrameType, pending: tuple[Any, ...]) -> None:
        """多行代入の範囲を抜けた後だけafterを記録する。"""
        line, end, branch, before = pending
        after = self._snapshot(frame)
        self.emit("branch", branch=lifecycle._actual_branch(branch, after),
            executed_source_line=line, executed_source_end_line=end,
            before=before, after=after)

    def _step_event(self, frame: FrameType, event: str) -> None:
        """実関数本体のenter/returnを必須にし、未到達wrapperを除外する。"""
        self._check_clock()
        if event == "call":
            values = frame.f_locals
            expected = (self.scope["pipe"], self.scope["sm"], self.scope["side"], self.scope["frame"], self.scope["time"])
            actual = tuple(values[name] for name in ("self", "sm", "side", "frame_idx", "time_sec"))
            if actual != expected or self.step_frame is not None:
                raise RuntimeError("writer実stepのinstance/side/時計不一致")
            self.step_frame, self.step_calls = frame, self.step_calls + 1
            self.emit("step_enter", snapshot=self._snapshot(frame))
        self.observe(frame, event)
        if event == "return":
            self.emit("step_return", snapshot=self._snapshot(frame), infer_call_count=self.infer_calls)

    def _inside_step(self, frame: FrameType) -> bool:
        """同実stepの子callであることだけを確認する。"""
        caller = frame.f_back
        while caller is not None:
            if caller is self.step_frame:
                return True
            caller = caller.f_back
        return False

    def _infer_event(self, frame: FrameType, event: str, arg: Any, state: dict[str, Any]) -> None:
        """固定infer本体への実到達・返却・例外を区別する。"""
        if event == "call":
            self.infer_calls += 1
            state["call_index"] = self.infer_calls
            caller = frame.f_back
            self.emit("infer_call", caller_line=None if caller is None else caller.f_lineno,
                      actual_code_path=frame.f_code.co_filename, call_index=state["call_index"])
        elif event == "exception":
            state["exception"] = True
            self.emit("infer_exception", exception_type=arg[0].__name__, message=str(arg[1]), call_index=state["call_index"])
        elif event == "line":
            state["exception"] = False
        elif event == "return":
            self.emit("infer_return", unwinding_exception=state.get("exception", False),
                      returned_board=self.entry.base.board_value(arg), call_index=state["call_index"])

    def trace(self, frame: FrameType, event: str, arg: Any) -> Any:
        """対象code以外は元traceへそのまま委譲し、local traceも保持する。"""
        prior = None if self.previous is None else self.previous(frame, event, arg)
        is_step = frame.f_code is self.rec.step_code
        is_infer = frame.f_code is self.infer_code and self._inside_step(frame)
        if not is_step and not is_infer:
            return prior
        state: dict[str, Any] = {}
        def dispatch(event: str, arg: Any) -> None:
            if is_step:
                self._step_event(frame, event)
            else:
                self._infer_event(frame, event, arg, state)
        dispatch(event, arg)
        def local(current: FrameType, event: str, arg: Any) -> Any:
            nonlocal prior
            if prior is not None:
                next_trace = prior(current, event, arg)
                if next_trace is not None:
                    prior = next_trace
            dispatch(event, arg)
            return local
        return local

    def wrapper(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """対象窓の外側だけでtraceを設置し、失敗もfinallyで復元する。"""
        @functools.wraps(original)
        def step(pipe: Any, side: str, frame_idx: int, time_sec: float,
                 *args: Any, **kwargs: Any) -> Any:
            if not self.entry.selected(frame_idx):
                return original(pipe, side, frame_idx, time_sec, *args, **kwargs)
            if self.scope or side not in self.entry.SIDES:
                raise RuntimeError("nested/別side writer scope")
            self.scope = dict(pipe=pipe, sm=kwargs["sm"], side=side, frame=frame_idx, time=time_sec)
            self.step_calls, self.infer_calls, self.step_frame = 0, 0, None
            self.previous = sys.gettrace()
            try:
                self._check_clock()
                sys.settrace(self.trace)
                result = original(pipe, side, frame_idx, time_sec, *args, **kwargs)
                if self.step_calls != 1:
                    raise RuntimeError("固定step本体の実到達が一回ではありません")
                return result
            except BaseException as exc:
                self.sink.failed(exc, "writer_trace")
                raise
            finally:
                sys.settrace(self.previous)
                self.pending.clear()
                self.scope, self.step_frame = {}, None
        return step


def install(stack: contextlib.ExitStack, entry: Any, collector: Any, rec: Any, sink: Any) -> WriterTrace:
    """既存観測wrapperは保持し、原本codeだけを別sinkへ観測する。"""
    original = collector.RecognitionPipeline._step_side
    observer = WriterTrace(entry, rec, sink, validate_code(entry, original))
    entry.base.patch(stack, collector.RecognitionPipeline, "_step_side", observer.wrapper(original))
    return observer
