"""完成済み短区間診断を再利用し、CPU再現に必要な境界入出力だけを追補する。"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import linecache
import sys
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any, Callable, TextIO

from scripts import diagnose_video38_confirmed_collapse_v1 as base


DETAIL_FIRST_FRAME, DETAIL_LAST_FRAME = 34700, 34720
DETAIL_LAUNCHER = base.ROOT / "scripts/launch_video38_confirmed_collapse_details_v1.sh"
DETAIL_TEST = base.ROOT / "tests/test_diagnose_video38_confirmed_collapse_details_v1.py"
REFERENCE = base.VERIFY / "video38_confirmed_collapse_2026-09-07_v2_native_identity_unproven_probe"
BOARD_LOCALS = ("prev_confirmed", "inferred_landing", "final_board", "published_confirmed")
EXIT_FUNCTIONS = ("_is_game_event_chain_exit", "_should_suppress_game_event_exit", "_should_suppress_slide_exit")
ORIGINAL_PREPARE, ORIGINAL_INSTRUMENT = base.prepare, base.instrument_pipeline


def event_value(event: Any) -> dict[str, Any] | None:
    """イベントの物理起点を別フレームのmutable参照から切り離す。"""
    if event is None:
        return None
    keys = ("trigger_sec", "end_sec", "chain_count", "total_score", "mechanism", "score_estimated")
    return {**{key: base.json_value(getattr(event, key, None)) for key in keys},
            "before_board": base.board_value(getattr(event, "before_board", None))}


def result_value(result: Any) -> dict[str, Any]:
    """物理出力を既存simulateの再実行なしで複製する。"""
    keys = ("chain_count", "total_erased", "total_ojama", "participating_cells")
    return {**{key: base.json_value(getattr(result, key, None)) for key in keys},
            "final_board": base.board_value(getattr(result, "final_board", None))}


def active_value(pipe: Any) -> dict[str, Any]:
    """対象2Pのactiveと終了条件の時点値を保存する。"""
    names = ("_chain_until_2p", "_chain_entry_t_2p", "_chain_start_next_2p",
             "_last_seen_next_2p", "_chain_event_max_until_2p")
    return {"active": event_value(getattr(pipe, "_active_chain_2p", None)),
            **{name: base.json_value(getattr(pipe, name, None)) for name in names}}


def caller_value(frame: FrameType) -> dict[str, Any]:
    """実際に呼び出した関数と行を記録する。"""
    return {"caller_function": frame.f_code.co_name, "caller_line": frame.f_lineno,
            "caller_file": frame.f_code.co_filename,
            "source": linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()}


class DetailRecorder(base.Recorder):
    """baseの鮮度管理を保ち、指定フレームのローカル盤面だけを追跡する。"""

    def __init__(self, stream: TextIO, target: dict[str, Any]) -> None:
        super().__init__(stream, target)
        self.update_code: Any = None
        self.detail_previous: dict[int, tuple[Any, int]] = {}
        self.active_previous: dict[int, tuple[Any, int]] = {}
        self.current_side: str | None = None

    def in_window(self) -> bool:
        """前後10フレームだけを詳細化し、推論区間自体は拡大しない。"""
        return DETAIL_FIRST_FRAME <= self.frame <= DETAIL_LAST_FRAME

    def begin_frame(self, frame: int, time_sec: float) -> None:
        """baseのraw鮮度と詳細traceのframe再利用防止を両立する。"""
        super().begin_frame(frame, time_sec)
        self.detail_previous.clear()
        self.active_previous.clear()

    def trace(self, frame: FrameType, event: str, arg: Any) -> Callable | None:
        """元ctx traceに加え、updateのactiveとstepの公開盤面を捕捉する。"""
        if frame.f_code is self.update_code:
            if self.in_window():
                self.trace_active(frame, event)
            return self.trace
        if frame.f_code is self.step_code and event == "call":
            self.current_side = frame.f_locals.get("side")
        result = super().trace(frame, event, arg)
        if frame.f_code is self.step_code and self.in_window():
            self.trace_boards(frame, event)
        if frame.f_code is self.step_code and event == "return":
            self.current_side = None
        return result

    def trace_active(self, frame: FrameType, event: str) -> None:
        """activeの変更を直前の実行行へ帰属させる。"""
        key = id(frame)
        if event == "call":
            self.active_previous.pop(key, None)
        if event not in ("line", "return"):
            return
        pipe = frame.f_locals.get("self")
        names = ("_chain_until_2p", "_chain_entry_t_2p", "_chain_start_next_2p", "_last_seen_next_2p")
        signature = (id(getattr(pipe, "_active_chain_2p", None)),
                     *(repr(getattr(pipe, name, None)) for name in names))
        previous = self.active_previous.get(key)
        if previous is None or previous[0] != signature:
            self.emit({"kind": "active_detail", "side": "2P", "active": active_value(pipe),
                       "after_caller_line": None if previous is None else previous[1],
                       **caller_value(frame)})
        self.active_previous[key] = (signature, frame.f_lineno)
        if event == "return":
            self.active_previous.pop(key, None)

    def trace_boards(self, frame: FrameType, event: str) -> None:
        """prev/inferred/final/publishedとpseudoの変更を代入境界ごとに記録する。"""
        key, local = id(frame), frame.f_locals
        if event == "call":
            self.detail_previous.pop(key, None)
        if event not in ("line", "return") or local.get("side") != "2P":
            return
        signature = tuple((name, None if local.get(name) is None else local[name]._grid.tobytes())
                          for name in BOARD_LOCALS)
        signature += (("pseudo", id(local.get("pseudo"))),)
        previous = self.detail_previous.get(key)
        if previous is None or previous[0] != signature:
            line = None if previous is None else previous[1]
            ctx = local.get("ctx")
            self.emit({"kind": "step_detail", "side": "2P", "after_caller_line": line,
                       "state": str(getattr(ctx, "state", None)), **caller_value(frame),
                       "boards": {name: base.board_value(local.get(name)) for name in BOARD_LOCALS},
                       "ctx_confirmed": base.board_value(getattr(ctx, "confirmed_board", None)),
                       "pseudo": event_value(local.get("pseudo")),
                       "chain_event": event_value(local.get("chain_event")),
                       "chain_count": local.get("chain_count"),
                       "falling_pair": base.json_value(local.get("falling_pair")),
                       "score_delta_self": local.get("score_d_for_self"),
                       "active": active_value(local.get("self"))})
        self.detail_previous[key] = (signature, frame.f_lineno)
        if event == "return":
            self.detail_previous.pop(key, None)


def detail_prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """完成済みbaseと新規計装の両方を同じ保存前guardに入れる。"""
    if args.start_sec != base.START_SEC or args.end_sec != base.END_SEC:
        raise ValueError("詳細診断は560～583秒の固定区間です")
    receipt, config = ORIGINAL_PREPARE(args)
    paths = (Path(__file__), DETAIL_LAUNCHER, DETAIL_TEST, REFERENCE / "COMPLETE")
    receipt["input_and_code_sha256"].update({str(path): base.sha256(path) for path in paths})
    receipt["detail_scope"] = {"frame_first": DETAIL_FIRST_FRAME, "frame_last": DETAIL_LAST_FRAME,
                              "base_module": str(Path(base.__file__)), "reference_root": str(REFERENCE),
                              "same_interval_once": True, "cpu_fixture_inputs": True}
    return receipt, config


def instrument_resolve(stack: contextlib.ExitStack, module: ModuleType, rec: DetailRecorder) -> None:
    """既存resolveを一回だけ呼び、入力と実際の返値を保存する。"""
    original, signature = module.resolve_after_placement, inspect.signature(module.resolve_after_placement)
    def resolve(*args: Any, **kwargs: Any) -> Any:
        caller = sys._getframe(1)
        capture = rec.in_window() and caller.f_locals.get("side") == "2P"
        if capture:
            bound = signature.bind(*args, **kwargs)
            before = {name: base.board_value(bound.arguments.get(name))
                      for name in ("new_confirmed", "prev_confirmed")}
        result = original(*args, **kwargs)
        if capture:
            rec.emit({"kind": "resolve_detail", "side": "2P", "inputs": before,
                      "score_delta_observed": bound.arguments.get("score_delta_observed"),
                      "result_board": base.board_value(result[0]), "chain_count": result[1],
                      **caller_value(caller)})
        return result
    base.patch(stack, module, "resolve_after_placement", resolve)


def instrument_simulate(stack: contextlib.ExitStack, cls: Any, rec: DetailRecorder) -> None:
    """候補全列挙は追加ログせず、採用resolveとchain起点のsimulateのみ記録する。"""
    original = cls.simulate
    def simulate(self: Any, board: Any) -> Any:
        caller = sys._getframe(1)
        capture = rec.in_window() and caller.f_code.co_name in ("resolve_after_placement", "_simulate_before_board")
        before = base.board_value(board) if capture else None
        result = original(self, board)
        if capture:
            rec.emit({"kind": "simulate_detail", "input_board": before, "result": result_value(result),
                      "side": rec.current_side,
                      "exclude_hidden_row_from_pop": getattr(self, "_exclude_hidden_row_from_pop", None),
                      **caller_value(caller)})
        return result
    base.patch(stack, cls, "simulate", simulate)


def instrument_stash(stack: contextlib.ExitStack, cls: Any, rec: DetailRecorder) -> None:
    """active消去の呼出行とNEXT・entry時刻を保存する。"""
    original = cls._stash_and_clear_active_chain
    def stash(self: Any, side: str) -> Any:
        capture = rec.in_window() and side == "2P"
        before = active_value(self) if capture else None
        caller = caller_value(sys._getframe(1)) if capture else {}
        result = original(self, side)
        if capture:
            rec.emit({"kind": "stash_detail", "side": side, "before": before,
                      "after": active_value(self), **caller})
        return result
    base.patch(stack, cls, "_stash_and_clear_active_chain", stash)


def instrument_decision(stack: contextlib.ExitStack, module: ModuleType, name: str, rec: DetailRecorder) -> None:
    """終了ガードの実際の引数と判定結果を再計算せず保存する。"""
    original = getattr(module, name)
    def decide(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if rec.in_window():
            rec.emit({"kind": "exit_decision_detail", "decision_function": name,
                      "args": base.json_value(args), "kwargs": base.json_value(kwargs),
                      "result": base.json_value(result), **caller_value(sys._getframe(1))})
        return result
    base.patch(stack, module, name, decide)


def instrument_emission(stack: contextlib.ExitStack, collector: ModuleType, rec: DetailRecorder) -> None:
    """回復盤面の非採用を確認するため、既存採用gateの状態と結果を残す。"""
    original = collector._should_emit
    def should_emit(state: Any, board: Any, bstate: Any, **kwargs: Any) -> Any:
        caller = sys._getframe(1)
        capture = rec.in_window() and caller.f_locals.get("side_label") == "2P"
        names = ("move_window_deadline_fi", "move_window_recorded")
        before = {name: getattr(state, name, None) for name in names} if capture else None
        result = original(state, board, bstate, **kwargs)
        if capture:
            rec.emit({"kind": "emission_detail", "side": "2P", "state": str(bstate),
                      "board": base.board_value(board), "move_window": before, "result": result})
        return result
    base.patch(stack, collector, "_should_emit", should_emit)


def instrument_erasable(stack: contextlib.ExitStack, detector: Any, simulator: Any, rec: DetailRecorder) -> None:
    """ゲート内部で実際に行った4連結検索とUNKNOWN個数・拒否結果を捕捉する。"""
    original_gate, original_find = detector._passes_erasable_gate, simulator.find_erasable_groups
    queries: list[list[dict[str, Any]]] = []
    def find(self: Any, board: Any) -> Any:
        result = original_find(self, board)
        if queries:
            queries[-1].append({"board": base.board_value(board), "group_count": len(result),
                               "groups": [{"color": getattr(group, "color", None),
                                           "cells": base.json_value(getattr(group, "cells", None))}
                                          for group in result]})
        return result
    def gate(self: Any, ctx: Any, signals: Any) -> Any:
        capture = rec.in_window() and rec.current_side == "2P"
        if not capture:
            return original_gate(self, ctx, signals)
        query: list[dict[str, Any]] = []
        before = base.board_value(ctx.confirmed_board)
        queries.append(query)
        try:
            result = original_gate(self, ctx, signals)
        finally:
            queries.pop()
        rec.emit({"kind": "erasable_gate_detail", "side": "2P", "state": str(ctx.state),
                  "confirmed": before, "chain_event": event_value(signals.chain_event),
                  "enable_formula_read_gate_bypass": self.enable_formula_read_gate_bypass,
                  "enable_chain_gate_raw_fallback": self.enable_chain_gate_raw_fallback,
                  "queries": query, "result": bool(result)})
        return result
    base.patch(stack, simulator, "find_erasable_groups", find)
    base.patch(stack, detector, "_passes_erasable_gate", gate)


def instrument_details(stack: contextlib.ExitStack, collector: ModuleType, rec: DetailRecorder) -> None:
    """baseのCUDA・生観測計装を維持したまま追加境界だけを一時ラップする。"""
    cls = collector.RecognitionPipeline
    from src.chain import ChainSimulator
    from src.state_detectors import ChainPhaseDetector
    module = sys.modules[cls.__module__]
    rec.update_code = cls.update.__code__
    ORIGINAL_INSTRUMENT(stack, collector, rec)
    instrument_resolve(stack, module, rec)
    instrument_simulate(stack, ChainSimulator, rec)
    instrument_erasable(stack, ChainPhaseDetector, ChainSimulator, rec)
    instrument_stash(stack, cls, rec)
    instrument_emission(stack, collector, rec)
    for name in EXIT_FUNCTIONS:
        instrument_decision(stack, module, name, rec)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """baseファイルを編集せず、fresh process内の参照のみ差し替えて復元する。"""
    with contextlib.ExitStack() as stack:
        base.patch(stack, base, "prepare", detail_prepare)
        base.patch(stack, base, "Recorder", DetailRecorder)
        base.patch(stack, base, "instrument_pipeline", instrument_details)
        return base.run(args)


def main() -> int:
    """原本native未証明の明示許可と新規rootを必須にする。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=base.START_SEC)
    parser.add_argument("--end-sec", type=float, default=base.END_SEC)
    parser.add_argument("--allow-native-runtime-mismatch", action="store_true")
    import json
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
