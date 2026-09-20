"""連鎖終端の実検出器入力・返却値だけを限定窓で追加記録する。"""

from __future__ import annotations

import contextlib
import functools
import inspect
import sys
from types import ModuleType
from typing import Any, Callable

from scripts import diagnose_video38_confirmed_collapse_v1 as base


FIRST_FRAME, LAST_FRAME = 35770, 35812
PREFIX = "end_boundary_"
DETECTORS = ("ChainPhaseDetector", "TsumoPhaseDetector", "GravitySettleDetector")
HELPERS = ("_is_game_event_chain_exit", "_should_suppress_game_event_exit",
           "_should_suppress_slide_exit")
COUNTERS = ("_settle_start_time", "_settle_start_frame", "_stable_consec",
            "_prev_puyo_count", "_consec_count", "_last_frame_idx", "_landed_consec_count")
SIGNALS = ("is_match_active", "next_pair", "slide_motion", "placement_validated",
           "score_delta", "own_score_delta", "effect_visible", "ojama_top_positive",
           "chain_max_hold_expired", "own_chain_hold_until_sec")
STEP_KWARGS = ("next_pair", "dnext_pair", "slide_motion", "score_d_for_self",
               "chain_max_hold_expired", "own_chain_active", "own_chain_hold_until")
PIPELINE_FIELDS = ("chain_start_next", "last_seen_next", "chain_entry_t", "chain_until")


def state_value(value: Any) -> Any:
    """Enumを固定の値へ変換する。"""
    return getattr(value, "value", value)


def event_value(value: Any) -> dict[str, Any] | None:
    """追加推論せず既存eventの公開fieldだけを読む。"""
    if value is None:
        return None
    names = ("chain_count", "mechanism", "trigger_sec", "projected_end_sec")
    return {name: getattr(value, name, None) for name in names}


def context_value(ctx: Any) -> dict[str, Any]:
    """状態とconfirmedの当該時点コピーを作る。"""
    return {"state": state_value(ctx.state), "frame_idx": ctx.frame_idx,
            "confirmed": base.board_value(ctx.confirmed_board)}


class BoundaryObserver:
    """_step_side内だけでsideを束縛し、外側helperのsideを推測しない。"""

    def __init__(self, recorder: Any) -> None:
        self.rec = recorder
        self.active_side: str | None = None

    def enabled(self) -> bool:
        return FIRST_FRAME <= self.rec.frame <= LAST_FRAME

    def emit(self, suffix: str, **fields: Any) -> None:
        self.rec.emit({"kind": PREFIX + suffix, **fields})

    def step_wrapper(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """元stepを1回だけ呼び、その動作と例外を保持する。"""
        @functools.wraps(original)
        def step(pipe: Any, side: str, frame_idx: int, time_sec: float,
                 is_active: bool, cnn_board: Any, chain_event: Any, **kwargs: Any) -> Any:
            if not self.enabled():
                return original(pipe, side, frame_idx, time_sec, is_active, cnn_board, chain_event, **kwargs)
            if frame_idx != self.rec.frame or time_sec != self.rec.time_sec:
                raise RuntimeError("終端stepの実clockが外側updateと不一致です")
            previous = self.active_side
            self.active_side = side
            try:
                ctx = kwargs["sm"].context
                fields = {key: getattr(pipe, f"_{key}_{side.lower()}") for key in PIPELINE_FIELDS}
                self.emit("step_enter", side=side, context=context_value(ctx),
                          is_match_active=is_active, input_cnn=base.board_value(cnn_board),
                          chain_event=event_value(chain_event), pipeline_fields=fields,
                          step_kwargs={key: kwargs.get(key) for key in STEP_KWARGS})
                result = original(pipe, side, frame_idx, time_sec, is_active, cnn_board, chain_event, **kwargs)
                self.emit("step_return", side=side, context=context_value(ctx),
                          result_state=state_value(result.state),
                          result_confirmed=base.board_value(result.confirmed_board))
                return result
            finally:
                self.active_side = previous
        return step

    def detector_wrapper(self, name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        """実際に呼ばれた検出器だけを記録し、未呼出を補完しない。"""
        @functools.wraps(original)
        def detect(detector: Any, ctx: Any, signals: Any) -> Any:
            if not self.enabled() or self.active_side is None:
                return original(detector, ctx, signals)
            if ctx.frame_idx != self.rec.frame or signals.time_sec != self.rec.time_sec:
                raise RuntimeError("終端detectorの実clockが外側updateと不一致です")
            before = {key: getattr(detector, key, None) for key in COUNTERS}
            context_before = context_value(ctx)
            signal_values = {key: getattr(signals, key, None) for key in SIGNALS}
            signal_values.update(cnn_board=base.board_value(signals.cnn_board),
                                 chain_event=event_value(signals.chain_event))
            result = original(detector, ctx, signals)
            self.emit("detector_return", side=self.active_side, detector=name,
                      context_before=context_before, signals=signal_values, counters_before=before,
                      counters_after={key: getattr(detector, key, None) for key in COUNTERS},
                      returned_state=state_value(result), context_after=context_value(ctx))
            return result
        return detect

    def helper_wrapper(self, name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        """分岐helperの実引数・返却値をcaller行付きで記録する。"""
        signature = inspect.signature(original)
        @functools.wraps(original)
        def helper(*args: Any, **kwargs: Any) -> Any:
            if not self.enabled():
                return original(*args, **kwargs)
            caller = sys._getframe(1)
            location = {"source_path": caller.f_code.co_filename, "line": caller.f_lineno,
                        "function": caller.f_code.co_name}
            del caller
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            result = original(*args, **kwargs)
            self.emit("exit_helper_return", side=self.active_side, helper=name,
                      arguments=dict(bound.arguments), returned=result, caller=location,
                      side_inferred_from_call_order=False)
            return result
        return helper


def install(stack: contextlib.ExitStack, collector: ModuleType, rec: Any) -> BoundaryObserver:
    """凍結collectorロード後にだけ設置し、全descriptorを元へ復元する。"""
    from src import state_detectors

    observer = BoundaryObserver(rec)
    cls = collector.RecognitionPipeline
    module = sys.modules[cls.__module__]
    base.patch(stack, cls, "_step_side", observer.step_wrapper(cls._step_side))
    for name in DETECTORS:
        detector = getattr(state_detectors, name)
        base.patch(stack, detector, "detect", observer.detector_wrapper(name, detector.detect))
    for name in HELPERS:
        original = getattr(module, name)
        base.patch(stack, module, name, observer.helper_wrapper(name, original))
    return observer
