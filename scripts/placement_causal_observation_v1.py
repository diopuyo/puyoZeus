"""通常着地の実関数呼出を、認識結果を変えずに結ぶ診断計装。"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import inspect
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


SIDES = ("1P", "2P")
BOARD_SHAPE = (13, 6)
FORMAT = "placement-causal-observation/v1"
GUARD_PATHS: tuple[str, ...] = (
    "scripts/chain_prediction_generation_hooks_v1.py",
)


@dataclass
class _StepScope:
    """一回の実 ``_step_side`` 内だけで呼出順を保持する。"""

    side: str
    frame_idx: int
    time_sec: float
    pipe: Any
    sm: Any
    generation_before: dict[str, Any]
    calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    sequence: int = 0
    last_infer_sha: str | None = None
    last_infer_object: Any = None
    last_infer_sequence: int | None = None
    last_infer_consumed: bool = False
    observed_color_calls: int = 0


class PlacementCausalObserver:
    """実着地呼出の系譜だけを保存し、正しさやpermissionを発行しない。"""

    def __init__(self, recorder: Any) -> None:
        self.rec = recorder
        self.scope: _StepScope | None = None
        self.failures: list[dict[str, str]] = []
        self.receipt_count = 0

    def begin_step(
        self, pipe: Any, side: Any, frame_idx: Any, time_sec: Any, sm: Any,
    ) -> _StepScope | None:
        """不正時計も元処理へ投げず、診断失敗として保持する。"""
        if self.scope is not None:
            self._fail("step_reentry", "観測scopeが既にactiveです")
            return None
        try:
            _validate_clock(side, frame_idx, time_sec)
            _validate_binding(self.rec, pipe, side, sm)
            before = _generation_value(self.rec, side, frame_idx, time_sec, pipe)
            self.scope = _StepScope(side, frame_idx, float(time_sec), pipe, sm, before)
            return self.scope
        except Exception as exc:
            self._fail("step_begin", exc)
            return None

    def capture(self, kind: str, producer: Callable[[], dict[str, Any]]) -> Any:
        """serializer失敗を認識例外へ混ぜず、scope失敗にする。"""
        scope = self.scope
        if scope is None:
            return None
        try:
            value = producer()
            scope.sequence += 1
            value.update({"call_kind": kind, "call_sequence": scope.sequence})
            scope.calls.append(value)
            return value
        except Exception as exc:
            failure = _failure(kind, exc)
            scope.errors.append(failure)
            self.failures.append(failure)
            return None

    def end_step(self, token: _StepScope | None, result: Any) -> None:
        """正常return後にだけresultとgeneration afterを一つのreceiptへ結ぶ。"""
        if token is None:
            return
        try:
            after = _generation_value(
                self.rec, token.side, token.frame_idx, token.time_sec, token.pipe,
            )
            if not token.generation_before["clock_matches"] or not after["clock_matches"]:
                self._fail("generation_clock", "step時計とgeneration時計が一致しません")
                token.errors.append(self.failures[-1])
            row = _receipt(token, after, result, "returned")
            self._emit(row)
        except Exception as exc:
            self.fail_scope("step_end", exc)
            self._emit(_minimal_receipt(token, "instrumentation_failed"))
        finally:
            self.scope = None

    def end_step_error(self, token: _StepScope | None, error: BaseException) -> None:
        """原例外を置換せず、最低限の失敗receiptだけを試みる。"""
        if token is None:
            return
        try:
            self._fail("original_step", error)
            row = _minimal_receipt(token, "original_exception")
            row["original_exception"] = type(error).__name__
            self._emit(row)
        finally:
            self.scope = None

    def assert_complete(self) -> None:
        """計装失敗やreceipt皆無から正常COMPLETEを作らせない。"""
        if self.failures:
            raise RuntimeError(f"placement観測失敗: {self.failures}")
        if self.receipt_count == 0:
            raise RuntimeError("placement receiptがありません")

    def fail_scope(self, stage: str, error: object) -> None:
        """実呼出の系譜不一致を現在stepと全体の両方へ保存する。"""
        failure = _failure(stage, error)
        self.failures.append(failure)
        if self.scope is not None:
            self.scope.errors.append(failure)

    def _emit(self, row: dict[str, Any]) -> None:
        """emit失敗も認識結果へ伝播させず、finalizeで失敗させる。"""
        try:
            self.rec.emit(row)
            self.receipt_count += 1
        except Exception as exc:
            self._fail("emit", exc)

    def _fail(self, stage: str, error: object) -> None:
        """object addressを保存せず型と文面だけを保持する。"""
        self.failures.append(_failure(stage, error))


def _validate_clock(side: Any, frame_idx: Any, time_sec: Any) -> None:
    """boolや非有限値を時計へ暗黙変換しない。"""
    if side not in SIDES:
        raise ValueError("sideが不正です")
    if type(frame_idx) is not int or frame_idx < 0:
        raise ValueError("frameは0以上の厳密intが必要です")
    if isinstance(time_sec, bool) or not isinstance(time_sec, (int, float)):
        raise ValueError("timeは有限数値が必要です")
    if not math.isfinite(float(time_sec)) or float(time_sec) < 0:
        raise ValueError("timeは有限かつ0以上が必要です")


def _generation_value(
    recorder: Any, side: str, frame_idx: int, time_sec: float, pipe: Any,
) -> dict[str, Any]:
    """software generationと実active clockを推測せず保存する。"""
    owner = getattr(recorder, "generation_recorder", None)
    if owner is None:
        raise ValueError("generation_recorderがありません")
    if getattr(owner, "_pipeline", None) is not pipe:
        raise ValueError("generationのpipeline identityが一致しません")
    value = owner.generation(side)
    reset_epoch = getattr(value, "reset_epoch", None)
    action_revision = getattr(value, "action_revision", None)
    sequence = getattr(owner, "_sequence", None)
    if (getattr(value, "side", None) != side or type(reset_epoch) is not int
            or reset_epoch < 0):
        raise ValueError("generationのside/resetが不正です")
    if (action_revision is not None
            and (type(action_revision) is not int or action_revision < 1)):
        raise ValueError("generationのactionが不正です")
    if type(sequence) is not int or sequence < 0:
        raise ValueError("generation sequenceが不正です")
    return {
        "side": getattr(value, "side", None),
        "reset_epoch": reset_epoch,
        "action_revision": action_revision,
        "identity_scope": getattr(value, "identity_scope", None),
        "generation_sequence": sequence,
        "clock_active": getattr(owner, "_in_frame", None),
        "clock_matches": (
            getattr(owner, "_in_frame", None) is True
            and getattr(owner, "_frame", None) == frame_idx
            and getattr(owner, "_time", None) == float(time_sec)
        ),
    }


def _validate_binding(recorder: Any, pipe: Any, side: str, sm: Any) -> None:
    """generation recorderが束縛した実pipeline/side machineだけを受理する。"""
    owner = getattr(recorder, "generation_recorder", None)
    if owner is None or getattr(owner, "_pipeline", None) is not pipe:
        raise ValueError("generation recorderのpipeline束縛が不正です")
    machines = getattr(owner, "_machines", None)
    if not isinstance(machines, dict) or machines.get(side) is not sm:
        raise ValueError("generation recorderのside machine束縛が不正です")


def _board_value(board: Any) -> dict[str, Any] | None:
    """盤面のコピー由来digestを保存し、未知shapeを成功扱いしない。"""
    if board is None:
        return None
    grid = np.asarray(getattr(board, "_grid", board))
    if grid.shape != BOARD_SHAPE:
        raise ValueError(f"盤面shapeが不正です: {grid.shape}")
    frozen = np.asarray(grid, dtype=np.int16).copy()
    return {
        "sha256": hashlib.sha256(frozen.tobytes()).hexdigest(),
        "color": int(np.isin(frozen, (1, 2, 3, 4, 5)).sum()),
        "garbage": int((frozen == 9).sum()),
        "unknown": int((frozen == 10).sum()),
    }


def _puyo_count(board: Any) -> int | None:
    """既存Board.count_puyosと同じ空・UNKNOWN除外数を追加判定なく保存する。"""
    if board is None:
        return None
    grid = np.asarray(getattr(board, "_grid", board))
    if grid.shape != BOARD_SHAPE:
        raise ValueError(f"盤面shapeが不正です: {grid.shape}")
    return int(((grid != 0) & (grid != 10)).sum())


def _frame_value(frame: Any) -> dict[str, Any] | None:
    """実frame入力を再推論せずshape/dtype/digestへ固定する。"""
    if frame is None:
        return None
    frozen = np.ascontiguousarray(np.asarray(frame)).copy()
    return {
        "shape": list(frozen.shape),
        "dtype": str(frozen.dtype),
        "sha256": hashlib.sha256(frozen.tobytes()).hexdigest(),
    }


def _plain(value: Any) -> Any:
    """receiptに必要な小さい値だけを決定的JSON型へ変換する。"""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int)):
        return enum_value
    raise TypeError(f"未対応のreceipt値です: {type(value).__name__}")


def _result_value(result: Any) -> dict[str, Any]:
    """最終SideResultの主要fieldと全field名を保存する。"""
    names = [item.name for item in dataclasses.fields(result)] if dataclasses.is_dataclass(result) else []
    return {
        "field_names": names,
        "side": getattr(result, "side", None),
        "state": _plain(getattr(result, "state", None)),
        "cnn_board": _board_value(getattr(result, "cnn_board", None)),
        "inferred_board": _board_value(getattr(result, "inferred_board", None)),
        "confirmed_board": _board_value(getattr(result, "confirmed_board", None)),
        "prob_board_present": getattr(result, "prob_board", None) is not None,
        "score": getattr(result, "score", None),
        "score_delta": getattr(result, "score_delta", None),
        "next_pair": _plain(getattr(result, "next_pair", None)),
        "dnext_pair": _plain(getattr(result, "dnext_pair", None)),
        "landing_diag": _plain(getattr(result, "landing_diag", None)),
        "board_provenance": getattr(result, "board_provenance", None),
        "answer_check_result": _plain(getattr(result, "answer_check_result", None)),
    }


def _receipt(
    scope: _StepScope, generation_after: dict[str, Any], result: Any, status: str,
) -> dict[str, Any]:
    """一回のstepを前後generation・実call列・戻りへ結ぶ。"""
    return {
        "kind": "placement_causal_receipt",
        "format": FORMAT,
        "side": scope.side,
        "frame_idx": scope.frame_idx,
        "time_sec": scope.time_sec,
        "status": status if not scope.errors else "instrumentation_failed",
        "generation_before": scope.generation_before,
        "generation_after": generation_after,
        "generation_changed_during_step": scope.generation_before != generation_after,
        "calls": scope.calls,
        "instrumentation_errors": scope.errors,
        "result": _result_value(result),
        "causal_call_lineage_only_not_board_truth": True,
        "commit_permission_issued": False,
    }


def _minimal_receipt(scope: _StepScope, status: str) -> dict[str, Any]:
    """serializer失敗時も既知clockと失敗状態だけは残す。"""
    return {
        "kind": "placement_causal_receipt",
        "format": FORMAT,
        "side": scope.side,
        "frame_idx": scope.frame_idx,
        "time_sec": scope.time_sec,
        "status": status,
        "generation_before": scope.generation_before,
        "calls": list(scope.calls),
        "instrumentation_errors": list(scope.errors),
        "causal_call_lineage_only_not_board_truth": True,
        "commit_permission_issued": False,
    }


def _failure(stage: str, error: object) -> dict[str, str]:
    """例外reprのobject addressを避ける。"""
    return {"stage": stage, "type": type(error).__name__, "message": str(error)}


def _arg(args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str) -> Any:
    """実callの位置/keyword引数を同値のまま読む。"""
    return args[index] if len(args) > index else kwargs.get(name)


def _detector_snapshot(detector: Any, ctx: Any, signals: Any) -> dict[str, Any]:
    """Tsumo判定の実入力と内部票を複製する。"""
    baseline = getattr(ctx, "confirmed_board", None)
    current = getattr(signals, "cnn_board", None)
    baseline_count = _puyo_count(baseline)
    current_count = _puyo_count(current)
    return {
        "context_identity_matches_step": False,
        "context_state": _plain(getattr(ctx, "state", None)),
        "context_frame_idx": getattr(ctx, "frame_idx", None),
        "signal_time_sec": _plain(getattr(signals, "time_sec", None)),
        "confirmed_board": _board_value(baseline),
        "cnn_board": _board_value(current),
        "baseline_count": baseline_count,
        "current_count": current_count,
        "count_diff": (
            None if baseline_count is None or current_count is None
            else current_count - baseline_count
        ),
        "next_pair": _plain(getattr(signals, "next_pair", None)),
        "slide_motion": getattr(signals, "slide_motion", None),
        "placement_validated": getattr(signals, "placement_validated", None),
        "consec_count": getattr(detector, "_consec_count", None),
        "landed_consec_count": getattr(detector, "_landed_consec_count", None),
        "last_frame_idx": getattr(detector, "_last_frame_idx", None),
        "last_landing_board": _board_value(
            getattr(detector, "_last_cnn_board_for_landing", None),
        ),
    }


def _detector_clock_matches(scope: _StepScope | None, ctx: Any, signals: Any) -> bool:
    """ctx frameとsignal timeを実step時計へ厳密に結ぶ。"""
    if scope is None or type(getattr(ctx, "frame_idx", None)) is not int:
        return False
    signal_time = getattr(signals, "time_sec", None)
    if isinstance(signal_time, bool) or not isinstance(signal_time, (int, float)):
        return False
    return (
        math.isfinite(float(signal_time))
        and ctx.frame_idx == scope.frame_idx
        and float(signal_time) == scope.time_sec
    )


def _wrap_detector(observer: PlacementCausalObserver, original: Callable[..., Any]) -> Callable[..., Any]:
    """Tsumo原関数を一回だけ呼び、前後票を保存する。"""
    @functools.wraps(original)
    def detect(detector: Any, ctx: Any, signals: Any) -> Any:
        scope = observer.scope
        before = observer.capture("tsumo_before", lambda: _detector_snapshot(detector, ctx, signals))
        try:
            result = original(detector, ctx, signals)
        except BaseException as exc:
            observer.capture("tsumo_exception", lambda: {"exception": type(exc).__name__})
            raise
        def after() -> dict[str, Any]:
            value = _detector_snapshot(detector, ctx, signals)
            value["context_identity_matches_step"] = bool(scope and ctx is scope.sm.context)
            value["detector_clock_matches_step"] = _detector_clock_matches(scope, ctx, signals)
            value["result"] = _plain(result)
            value["before_call_sequence"] = None if before is None else before["call_sequence"]
            return value
        captured = observer.capture("tsumo_after", after)
        if captured is not None:
            if not captured["context_identity_matches_step"]:
                observer.fail_scope("tsumo_context_identity", "実stepのcontextと一致しません")
            if not captured["detector_clock_matches_step"]:
                observer.fail_scope("tsumo_clock", "ctx frame/signal timeがstep時計と一致しません")
        return result
    return detect


def _wrap_infer(observer: PlacementCausalObserver, original: Callable[..., Any]) -> Callable[..., Any]:
    """inferの実入力/返却を追加推論せず保存する。"""
    @functools.wraps(original)
    def infer(*args: Any, **kwargs: Any) -> Any:
        before = observer.capture("infer_before", lambda: {
            "previous": _board_value(_arg(args, kwargs, 0, "previous")),
            "cnn": _board_value(_arg(args, kwargs, 1, "cnn")),
            "falling_pair": _plain(_arg(args, kwargs, 2, "next_pair")),
        })
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            observer.capture("infer_exception", lambda: {"exception": type(exc).__name__})
            raise
        after = observer.capture("infer_after", lambda: {
            "result": _board_value(result),
            "before_call_sequence": None if before is None else before["call_sequence"],
        })
        if observer.scope is not None and after is not None:
            value = after.get("result")
            observer.scope.last_infer_sha = None if value is None else value.get("sha256")
            observer.scope.last_infer_object = result
            observer.scope.last_infer_sequence = after["call_sequence"]
            observer.scope.last_infer_consumed = False
            observer.scope.observed_color_calls = 0
        return result
    return infer


def _wrap_observed_color(
    observer: PlacementCausalObserver, original: Callable[..., Any],
) -> Callable[..., Any]:
    """既存の色補正一回をinferとresolveの実object鎖へ結ぶ。"""
    @functools.wraps(original)
    def correct(*args: Any, **kwargs: Any) -> Any:
        scope = observer.scope
        source = _arg(args, kwargs, 0, "inferred")
        before = observer.capture("observed_color_before", lambda: {
            "input": _board_value(source),
            "input_is_current_infer_object": bool(scope and source is scope.last_infer_object),
            "prior_observed_color_calls": 0 if scope is None else scope.observed_color_calls,
            "prior_infer_already_resolved": bool(scope and scope.last_infer_consumed),
            "prior_infer_is_immediately_previous_call": bool(
                scope and scope.last_infer_sequence == scope.sequence
            ),
            "previous": _board_value(_arg(args, kwargs, 1, "prev_confirmed")),
            "cnn": _board_value(_arg(args, kwargs, 2, "cnn_board")),
            "frame_bgr": _frame_value(_arg(args, kwargs, 4, "frame_bgr")),
        })
        valid = bool(before and before["input_is_current_infer_object"]
                     and before["prior_observed_color_calls"] == 0
                     and not before["prior_infer_already_resolved"]
                     and before["prior_infer_is_immediately_previous_call"])
        if before is not None and not valid:
            observer.fail_scope("observed_color_lineage", "infer直後の一回の色補正ではありません")
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            observer.capture("observed_color_exception", lambda: {"exception": type(exc).__name__})
            raise
        after = observer.capture("observed_color_after", lambda: {
            "result": _board_value(result), "returns_same_object": result is source,
            "before_call_sequence": None if before is None else before["call_sequence"],
        })
        if scope is not None and valid and after is not None:
            scope.last_infer_object = result
            scope.last_infer_sha = after["result"]["sha256"] if after["result"] else None
            scope.last_infer_sequence = after["call_sequence"]
            scope.last_infer_consumed = False
            scope.observed_color_calls += 1
        return result
    return correct


def _wrap_resolve(observer: PlacementCausalObserver, original: Callable[..., Any]) -> Callable[..., Any]:
    """resolveの実入力/返却と直前infer digest一致だけを保存する。"""
    @functools.wraps(original)
    def resolve(*args: Any, **kwargs: Any) -> Any:
        def build_before() -> dict[str, Any]:
            input_value = _board_value(_arg(args, kwargs, 0, "new_board"))
            scope = observer.scope
            return {
                "input": input_value,
                "previous": _board_value(kwargs.get("prev_confirmed")),
                "score_delta_observed": kwargs.get("score_delta_observed"),
                "input_is_prior_infer_object": bool(
                    scope and _arg(args, kwargs, 0, "new_board") is scope.last_infer_object
                ),
                "input_matches_prior_infer_output": bool(
                    scope and input_value
                    and input_value["sha256"] == scope.last_infer_sha
                ),
                "prior_infer_call_sequence": None if scope is None else scope.last_infer_sequence,
                "prior_infer_already_resolved": bool(scope and scope.last_infer_consumed),
            }
        before = observer.capture("resolve_before", build_before)
        if before is not None:
            if not before["input_is_prior_infer_object"]:
                observer.fail_scope("resolve_input_lineage", "直前inferの実objectと一致しません")
            elif before["prior_infer_already_resolved"]:
                observer.fail_scope("resolve_input_reuse", "同じinfer返却を二重resolveしました")
            elif observer.scope is not None:
                observer.scope.last_infer_consumed = True
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            observer.capture("resolve_exception", lambda: {"exception": type(exc).__name__})
            raise
        observer.capture("resolve_after", lambda: {
            "final_board": _board_value(result[0]),
            "chain_count": result[1],
            "before_call_sequence": None if before is None else before["call_sequence"],
        })
        return result
    return resolve


def _wrap_step(observer: PlacementCausalObserver, original: Callable[..., Any]) -> Callable[..., Any]:
    """stepの正常/例外returnを変えず観測scopeを必ず閉じる。"""
    @functools.wraps(original)
    def step(pipe: Any, *args: Any, **kwargs: Any) -> Any:
        side = _arg(args, kwargs, 0, "side")
        frame_idx = _arg(args, kwargs, 1, "frame_idx")
        time_sec = _arg(args, kwargs, 2, "time_sec")
        sm = kwargs.get("sm")
        token = observer.begin_step(pipe, side, frame_idx, time_sec, sm)
        try:
            result = original(pipe, *args, **kwargs)
        except BaseException as exc:
            observer.end_step_error(token, exc)
            raise
        observer.end_step(token, result)
        return result
    return step


def _step_globals(function: Callable[..., Any]) -> dict[str, Any]:
    """実stepが参照するglobalsをwraps鎖から一意に選ぶ。"""
    seen: set[int] = set()
    current: Any = function
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        names = set(getattr(getattr(current, "__code__", None), "co_names", ()))
        namespace = getattr(current, "__globals__", None)
        if isinstance(namespace, dict) and {"infer_placement", "resolve_after_placement"} <= names:
            return namespace
        current = getattr(current, "__wrapped__", None)
    raise RuntimeError("実_step_sideのplacement globalsを特定できません")


def _patch(stack: contextlib.ExitStack, obj: Any, name: str, value: Any) -> None:
    """descriptorを含む元属性をExitStackで復元する。"""
    original = inspect.getattr_static(obj, name)
    setattr(obj, name, value)
    stack.callback(setattr, obj, name, original)


def _patch_mapping(
    stack: contextlib.ExitStack, mapping: dict[str, Any], name: str, value: Any,
) -> None:
    """AST変換stepの専用globalsも有無を含めて復元する。"""
    existed, original = name in mapping, mapping.get(name)
    mapping[name] = value
    if existed:
        stack.callback(mapping.__setitem__, name, original)
    else:
        stack.callback(mapping.pop, name, None)


def _details_original_cell(
    function: Callable[..., Any], rec: Any,
) -> tuple[Any, Callable[..., Any]]:
    """固定details wrapperのoriginal closureだけを厳密に特定する。"""
    code, closure = getattr(function, "__code__", None), getattr(function, "__closure__", None)
    if code is None or closure is None or code.co_name != "resolve":
        raise RuntimeError("固定details resolve wrapperではありません")
    if tuple(code.co_freevars) != ("original", "rec", "signature"):
        raise RuntimeError("details resolve closure契約が一致しません")
    required = {"_getframe", "f_locals", "in_window", "bind", "emit", "caller_value"}
    if not required <= set(code.co_names):
        raise RuntimeError("details resolve caller観測契約が一致しません")
    factory = getattr(function, "__globals__", {}).get("instrument_resolve")
    nested = [item for item in getattr(getattr(factory, "__code__", None), "co_consts", ())
              if inspect.iscode(item) and item.co_name == "resolve"]
    if len(nested) != 1 or code is not nested[0]:
        raise RuntimeError("固定details instrument_resolve由来ではありません")
    cells = dict(zip(code.co_freevars, closure, strict=True))
    if cells["rec"].cell_contents is not rec:
        raise RuntimeError("details resolveのrec identityが一致しません")
    original = cells["original"].cell_contents
    if not callable(original):
        raise RuntimeError("details resolveのoriginalがcallableではありません")
    return cells["original"], original


def _restore_cell(cell: Any, original: Callable[..., Any]) -> None:
    """closure cellを元identityへ戻す。"""
    cell.cell_contents = original


def _patch_details_original(
    stack: contextlib.ExitStack, wrapper: Callable[..., Any], observer: PlacementCausalObserver,
) -> None:
    """detailsの直callerを変えず、その内側の実resolveだけを観測する。"""
    cell, original = _details_original_cell(wrapper, observer.rec)
    cell.cell_contents = _wrap_resolve(observer, original)
    stack.callback(_restore_cell, cell, original)


def install(
    stack: contextlib.ExitStack, collector: Any, rec: Any, *,
    preserve_outer_resolve_caller: bool = False,
) -> PlacementCausalObserver:
    """fresh診断runtimeへ観測だけを接続する。"""
    if not callable(getattr(rec, "emit", None)) or getattr(rec, "generation_recorder", None) is None:
        raise RuntimeError("emitとgeneration_recorderが必要です")
    cls = collector.RecognitionPipeline
    original_step = cls._step_side
    namespace = _step_globals(original_step)
    original_infer = namespace["infer_placement"]
    original_resolve = namespace["resolve_after_placement"]
    original_observed_color = namespace.get("_apply_landing_observed_color_correction")
    module = sys.modules.get(getattr(original_step, "__module__", ""))
    detector_cls = namespace.get("TsumoPhaseDetector")
    if detector_cls is None and module is not None:
        detector_cls = getattr(module, "TsumoPhaseDetector", None)
    if detector_cls is None or not callable(getattr(detector_cls, "detect", None)):
        raise RuntimeError("実TsumoPhaseDetectorが必要です")
    if not callable(original_observed_color):
        raise RuntimeError("実observed color補正が必要です")
    observer = PlacementCausalObserver(rec)
    _patch_mapping(stack, namespace, "infer_placement", _wrap_infer(observer, original_infer))
    if preserve_outer_resolve_caller:
        _patch_details_original(stack, original_resolve, observer)
    else:
        _patch_mapping(stack, namespace, "resolve_after_placement",
                       _wrap_resolve(observer, original_resolve))
    _patch_mapping(stack, namespace, "_apply_landing_observed_color_correction",
                   _wrap_observed_color(observer, original_observed_color))
    _patch(stack, detector_cls, "detect", _wrap_detector(observer, detector_cls.detect))
    _patch(stack, cls, "_step_side", _wrap_step(observer, original_step))
    return observer
