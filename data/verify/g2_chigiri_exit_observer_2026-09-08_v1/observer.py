"""実呼出の構造を変えず、ちぎり窓の出口を別 sink に記録する。"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
from enum import Enum
from pathlib import Path
import sys
from types import FrameType, SimpleNamespace
from typing import Any, Callable

FIRST, LAST, STRIDE = 34468, 34484, 2
PREFIX = "chigiri_exit_"
COUNTERS = ("_settle_start_time", "_settle_start_frame", "_stable_consec", "_prev_puyo_count",
            "_consec_count", "_last_frame_idx", "_landed_consec_count", "_last_cnn_board_for_landing")
PARAMETERS = ("min_increase", "max_increase", "consec_threshold", "landed_consec", "min_frames", "max_hold_sec")
SIGNALS = ("is_match_active", "next_pair", "slide_motion", "placement_validated", "score_delta",
           "own_score_delta", "effect_visible", "ojama_top_positive", "chain_max_hold_expired")


def selected(frame: int) -> bool:
    return type(frame) is int and FIRST <= frame <= LAST and frame % STRIDE == 0


def expected() -> list[tuple[int, str]]:
    return [(frame, "1P") for frame in range(FIRST, LAST + 1, STRIDE)]


def code_info(code: Any) -> dict[str, Any]:
    return {"path": code.co_filename, "first_line": code.co_firstlineno,
            "name": code.co_qualname, "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest()}


def register(entry: Any, collector: Any, codes: tuple[Any, Any]) -> dict[Any, str]:
    """実 frozen module の全 detector と SM/resolve の code を登録する。"""
    module = sys.modules[collector.RecognitionPipeline.__module__]
    sm_type = module.BoardStateMachine
    detector_module = sys.modules[module.ChainPhaseDetector.__module__]
    functions = {inspect.unwrap(sm_type.update): "sm", inspect.unwrap(module.resolve_after_placement): "resolve"}
    for value in vars(detector_module).values():
        if inspect.isclass(value) and value.__module__.startswith("src.") and hasattr(value, "detect"):
            functions[inspect.unwrap(value.detect)] = "detector"
    result = {codes[0]: "step", codes[1]: "infer"}
    for function, kind in functions.items():
        path = Path(function.__code__.co_filename).resolve()
        if path.parent != (entry.base.SNAPSHOT / "src").resolve():
            raise RuntimeError("観測 code は固定 src に限定します")
        result[function.__code__] = kind
    return result


def make_trace(writer: Any, entry: Any, collector: Any, rec: Any, sink: Any) -> Any:
    """既存 WriterTrace の scope/時計/復元を継承し、値はコピーだけ行う。"""
    original = collector.RecognitionPipeline._step_side
    codes = writer.validate_code(entry, original)
    registry = register(entry, collector, codes)
    facade = SimpleNamespace(**vars(entry))
    facade.selected = selected

    class ExitTrace(writer.WriterTrace):
        def __init__(self) -> None:
            super().__init__(facade, rec, sink, codes)
            self.calls: dict[str, int] = {}
            self.registry = registry

        def emit(self, suffix: str, **fields: Any) -> None:
            self._check_clock()
            sink.emit({"kind": PREFIX + suffix, "frame_idx": self.scope["frame"],
                "time_sec": self.scope["time"], "side": self.scope["side"], **fields,
                "current_publication_allowed": False, "accounting_commit_allowed": False})

        def trace(self, frame: FrameType, event: str, arg: Any) -> Any:
            return trace_event(self, frame, event, arg)

    return ExitTrace()


def json_value(trace: Any, value: Any) -> Any:
    if hasattr(value, "_grid"):
        return trace.entry.base.board_value(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "_cells"):
        return {"cell_probabilities": [[trace.entry.base.json_value(cell.probs) for cell in row] for row in value._cells]}
    if isinstance(value, (list, tuple)):
        return [json_value(trace, item) for item in value]
    return trace.entry.base.json_value(value)


def context(trace: Any, ctx: Any) -> dict[str, Any]:
    return {"state": json_value(trace, ctx.state), "frame_idx": ctx.frame_idx,
            "confirmed": json_value(trace, ctx.confirmed_board),
            "pending": json_value(trace, ctx.pending_board)}


def inputs(trace: Any, frame: FrameType, kind: str) -> dict[str, Any]:
    """局所変数の欠測と None を区別し、補間や再推論をしない。"""
    local = frame.f_locals
    if kind in ("sm", "detector"):
        ctx = local["self"].context if kind == "sm" else local["ctx"]
        signal = local["signals"]
        result = {"context": context(trace, ctx), "cnn": json_value(trace, signal.cnn_board),
                  "signals": {name: json_value(trace, getattr(signal, name, None)) for name in SIGNALS}}
        if kind == "detector":
            result["counters"] = {name: json_value(trace, getattr(local["self"], name))
                                  for name in COUNTERS if hasattr(local["self"], name)}
            result["parameters"] = {name: json_value(trace, getattr(local["self"], name))
                                     for name in PARAMETERS if hasattr(local["self"], name)}
        return result
    names = ("confirmed_before", "cnn_after", "next_pair", "new_confirmed", "prev_confirmed",
             "inferred_board", "inferred_b", "prev_state", "chain_count")
    return {name: {"present": name in local, "value": json_value(trace, local.get(name))} for name in names}


def step_event(trace: Any, frame: FrameType, event: str, arg: Any, state: dict[str, Any]) -> None:
    trace._check_clock()
    if event == "call":
        values = frame.f_locals
        actual = tuple(values[name] for name in ("self", "sm", "side", "frame_idx", "time_sec"))
        expected_values = tuple(trace.scope[name] for name in ("pipe", "sm", "side", "frame", "time"))
        if actual != expected_values or trace.step_frame is not None:
            raise RuntimeError("実 step 同一性違反")
        trace.step_frame, trace.step_calls, trace.calls = frame, trace.step_calls + 1, {}
        trace.emit("step_enter", snapshot=trace._snapshot(frame), actual_code=code_info(frame.f_code),
                   input_cnn=json_value(trace, values["cnn_board"]))
    elif event == "exception":
        state["unwinding"] = True
        trace.emit("step_exception", exception_type=arg[0].__name__, message=str(arg[1]))
    elif event == "line":
        state["unwinding"] = False
    elif event == "return":
        result = None if arg is None else {name: json_value(trace, getattr(arg, name, None))
            for name in ("state", "confirmed_board", "prob_board", "inferred_board", "landing_diag", "score", "score_delta")}
        trace.emit("step_return", snapshot=trace._snapshot(frame), result=result,
                   calls=dict(trace.calls), unwinding_exception=state.get("unwinding", False))


def child_event(trace: Any, frame: FrameType, event: str, arg: Any, state: dict[str, Any], kind: str) -> None:
    if event == "call":
        if kind == "sm":
            values = frame.f_locals
            if values["self"] is not trace.scope["sm"] or values["frame_idx"] != trace.scope["frame"]:
                raise RuntimeError("別 SM/clock 呼出")
            if values["signals"].time_sec != trace.scope["time"]:
                raise RuntimeError("SM signals clock 不一致")
            actual_codes = [inspect.unwrap(det.detect).__code__ for det in values["self"]._detectors]
            if any(trace.registry.get(code) != "detector" for code in actual_codes):
                raise RuntimeError("未登録 detector を含む実 SM")
        trace.calls[kind] = trace.calls.get(kind, 0) + 1
        if kind == "infer":
            trace.infer_calls = trace.calls[kind]
        state["index"] = trace.calls[kind]
        actual_type = type(frame.f_locals.get("self")).__qualname__
        trace.emit(kind + "_call", call_index=state["index"], actual_type=actual_type,
                   actual_code=code_info(frame.f_code), inputs=inputs(trace, frame, kind))
    elif event == "exception":
        state["unwinding"] = True
        trace.emit(kind + "_exception", call_index=state["index"], exception_type=arg[0].__name__, message=str(arg[1]))
    elif event == "line":
        state["unwinding"] = False
        state["last_line"] = frame.f_lineno
    elif event == "return":
        fields = {"call_index": state["index"], "return_line": frame.f_lineno,
                  "last_line": state.get("last_line"), "unwinding_exception": state.get("unwinding", False),
                  "locals": inputs(trace, frame, kind)}
        if kind == "sm":
            fields.update(returned=context(trace, arg) if arg is not None else None,
                new_state=json_value(trace, frame.f_locals.get("new_state")),
                last_detector_type=type(frame.f_locals.get("det")).__qualname__)
        else:
            fields["returned"] = json_value(trace, arg)
            if kind == "detector":
                names = ("baseline_count", "cur_count", "diff", "same", "effective_landed", "elapsed")
                fields["decision_locals"] = {name: {"present": name in frame.f_locals,
                    "value": json_value(trace, frame.f_locals.get(name))} for name in names}
        trace.emit(kind + "_return", **fields)


def trace_event(trace: Any, frame: FrameType, event: str, arg: Any) -> Any:
    """既存 global/local trace の戻り値をそのまま保持して共存する。"""
    prior = None if trace.previous is None else trace.previous(frame, event, arg)
    kind = trace.registry.get(frame.f_code)
    if kind is None or (kind != "step" and not trace._inside_step(frame)):
        return prior
    state: dict[str, Any] = {}
    def dispatch(event: str, arg: Any) -> None:
        if kind == "step":
            step_event(trace, frame, event, arg, state)
        else:
            child_event(trace, frame, event, arg, state, kind)
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


def install(stack: contextlib.ExitStack, writer: Any, entry: Any, collector: Any, rec: Any, sink: Any) -> Any:
    observer = make_trace(writer, entry, collector, rec, sink)
    original = collector.RecognitionPipeline._step_side
    traced = observer.wrapper(original)
    @functools.wraps(original)
    def step(pipe: Any, side: str, frame_idx: int, time_sec: float, *args: Any, **kwargs: Any) -> Any:
        if side != "1P" or not selected(frame_idx):
            return original(pipe, side, frame_idx, time_sec, *args, **kwargs)
        return traced(pipe, side, frame_idx, time_sec, *args, **kwargs)
    entry.base.patch(stack, collector.RecognitionPipeline, "_step_side", step)
    return observer
