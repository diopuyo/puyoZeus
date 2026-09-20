"""追加計装をCPUの小さい入力で検証し、元認識を余分に呼ばないことを守る。"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import diagnose_video38_confirmed_collapse_details_v1 as diag


def board(count: int) -> Any:
    """13x6の独立したfixture盤面を作る。"""
    grid = np.zeros((13, 6), np.int8)
    grid.reshape(-1)[-count:] = 3
    return SimpleNamespace(_grid=grid)


def recorder() -> diag.DetailRecorder:
    """対象frameのメモリ内記録器を作る。"""
    rec = diag.DetailRecorder(io.StringIO(), diag.base.board_value(board(9)))
    rec.begin_frame(34702, 578.366667)
    return rec


def rows(rec: diag.DetailRecorder) -> list[dict[str, Any]]:
    """発行した行だけを読み取る。"""
    return [json.loads(line) for line in rec.stream.getvalue().splitlines()]


def test_step_trace_captures_published_change_and_restores_frame_ids() -> None:
    rec = recorder()
    def step(side: str) -> None:
        ctx = SimpleNamespace(state="STABLE", confirmed_board=board(11))
        prev_confirmed = board(66)
        inferred_landing = board(68)
        final_board = board(9)
        pseudo = SimpleNamespace(before_board=inferred_landing, chain_count=13)
        published_confirmed = ctx.confirmed_board
        published_confirmed = board(9)
    rec.step_code = step.__code__
    old_trace = sys.gettrace()
    try:
        sys.settrace(rec.trace)
        step("1P")
        step("2P")
        step("2P")
    finally:
        sys.settrace(old_trace)
    detail = [row for row in rows(rec) if row["kind"] == "step_detail"]
    assert all(row["side"] == "2P" for row in detail)
    assert any(row["boards"]["published_confirmed"]["color"] == 9 for row in detail
               if row["boards"]["published_confirmed"] is not None)
    assert any(row["pseudo"]["before_board"]["color"] == 68 for row in detail if row["pseudo"])
    assert rec.detail_previous == rec.previous == {}
    assert rec.current_side is None


def test_details_are_not_carried_to_next_frame() -> None:
    rec = recorder()
    rec.detail_previous[1] = ("old", 10)
    rec.active_previous[1] = ("old", 10)
    rec.begin_frame(34722, 578.7)
    assert not rec.in_window()
    assert rec.detail_previous == rec.active_previous == {}


def test_resolve_runs_once_and_input_is_copied() -> None:
    rec, calls = recorder(), []
    def original(new_confirmed: Any, chain_sim: Any, prev_confirmed: Any = None,
                 score_delta_observed: int | None = None) -> Any:
        calls.append(1)
        return board(9), 13
    module = SimpleNamespace(resolve_after_placement=original)
    def caller(side: str) -> Any:
        return module.resolve_after_placement(board(68), None, prev_confirmed=board(66), score_delta_observed=1)
    with contextlib.ExitStack() as stack:
        diag.instrument_resolve(stack, module, rec)
        caller("2P")
    assert calls == [1]
    assert module.resolve_after_placement is original
    assert rows(rec)[0]["inputs"]["new_confirmed"]["color"] == 68
    assert rows(rec)[0]["chain_count"] == 13


def test_actual_erasable_result_is_recorded_without_second_query() -> None:
    rec, calls = recorder(), []
    rec.current_side = "2P"
    class Simulator:
        def find_erasable_groups(self, value: Any) -> list[Any]:
            calls.append(1)
            return []
    class Detector:
        enable_formula_read_gate_bypass = True
        enable_chain_gate_raw_fallback = False
        def _passes_erasable_gate(self, ctx: Any, signals: Any) -> bool:
            return bool(Simulator().find_erasable_groups(ctx.confirmed_board))
    original_gate, original_find = Detector._passes_erasable_gate, Simulator.find_erasable_groups
    with contextlib.ExitStack() as stack:
        diag.instrument_erasable(stack, Detector, Simulator, rec)
        result = Detector()._passes_erasable_gate(SimpleNamespace(state="STABLE", confirmed_board=board(9)),
                                                SimpleNamespace(chain_event=None))
    assert result is False and calls == [1]
    assert rows(rec)[0]["queries"][0]["group_count"] == 0
    assert rows(rec)[0]["confirmed"]["unknown"] == 0
    assert Detector._passes_erasable_gate is original_gate
    assert Simulator.find_erasable_groups is original_find


def test_stash_captures_before_after_and_restores() -> None:
    rec = recorder()
    class Pipeline:
        _active_chain_2p: Any = SimpleNamespace(chain_count=13, before_board=board(68))
        def _stash_and_clear_active_chain(self, side: str) -> None:
            self._active_chain_2p = None
    original = Pipeline._stash_and_clear_active_chain
    with contextlib.ExitStack() as stack:
        diag.instrument_stash(stack, Pipeline, rec)
        Pipeline()._stash_and_clear_active_chain("2P")
    assert rows(rec)[0]["before"]["active"]["chain_count"] == 13
    assert rows(rec)[0]["after"]["active"] is None
    assert Pipeline._stash_and_clear_active_chain is original


def test_base_entry_points_restored_after_error(monkeypatch: pytest.MonkeyPatch) -> None:
    originals = {name: getattr(diag.base, name) for name in ("prepare", "Recorder", "instrument_pipeline")}
    def fail(args: Any) -> Any:
        assert diag.base.Recorder is diag.DetailRecorder
        raise RuntimeError("fixture")
    monkeypatch.setattr(diag.base, "run", fail)
    with pytest.raises(RuntimeError):
        diag.run(SimpleNamespace())
    assert all(getattr(diag.base, name) is value for name, value in originals.items())


def test_detail_hashes_added_before_saving(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag, "ORIGINAL_PREPARE", lambda args: ({"input_and_code_sha256": {}}, {}))
    receipt, _ = diag.detail_prepare(SimpleNamespace(start_sec=560.0, end_sec=583.0))
    hashes = receipt["input_and_code_sha256"]
    for path in (Path(diag.__file__), diag.DETAIL_LAUNCHER, diag.DETAIL_TEST, diag.REFERENCE / "COMPLETE"):
        assert hashes[str(path)] == diag.base.sha256(path)


def test_expanded_interval_rejected() -> None:
    with pytest.raises(ValueError):
        diag.detail_prepare(SimpleNamespace(start_sec=559.0, end_sec=583.0))


def test_new_functions_obey_fifty_line_limit() -> None:
    for path in (Path(diag.__file__), Path(__file__)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno - node.lineno + 1 <= 50, (path.name, node.name)
