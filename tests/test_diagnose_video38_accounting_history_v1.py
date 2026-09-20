"""履歴診断の時計・非干渉・保存境界をGPU非使用で検査する。"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import diagnose_video38_accounting_history_v1 as diag


def recorder() -> diag.HistoryRecorder:
    grid = np.zeros((13, 6), dtype=np.int8)
    return diag.HistoryRecorder(io.StringIO(), diag.base.board_value(grid))


def pipeline_fields() -> dict[str, Any]:
    result = {}
    for side in ("1p", "2p"):
        fields = dict(tsumo_count=Counter({4: 2}), pending_tsumo=deque([(5, 5)]),
                      last_seen_next=(3, 5), landing_pending=(1, (5, 5)),
                      last_consumed_color=(4, 4), first_move_sec=544.8,
                      constraint_valid=False)
        result.update({f"_{name}_{side}": value for name, value in fields.items()})
        result[f"_stable_color_memory_{side}"] = {}
    return result


def test_counter_snapshot_copies_mutable_values_without_claiming_validity() -> None:
    pipeline = SimpleNamespace(**pipeline_fields())
    before = diag.accounting_snapshot(pipeline, "1P")
    pipeline._tsumo_count_1p[4] = 99
    pipeline._pending_tsumo_1p.clear()
    assert before["tsumo_count"] == {"4": 2}
    assert before["pending_tsumo"] == [[5, 5]]
    assert before["constraint_valid"] is False


@pytest.mark.parametrize("frame,time_sec", [
    (29054, 29054 / 60), (29052, 0.0), (36300, 605.0),
])
def test_clock_rejects_wrong_start_missing_frame_or_time(frame: int, time_sec: float) -> None:
    with pytest.raises(RuntimeError, match="clock"):
        recorder().begin_frame(frame, time_sec)


def test_clock_accepts_exact_range_and_rejects_duplicate() -> None:
    rec = recorder()
    for frame in range(diag.FIRST_FRAME, diag.END_FRAME, diag.STRIDE):
        rec.begin_frame(frame, frame / diag.FPS)
    assert rec.frames == 3624
    with pytest.raises(RuntimeError):
        rec.begin_frame(rec.frame, rec.time_sec)


def test_prepare_does_not_allow_an_unregistered_range(tmp_path: Path) -> None:
    args = argparse.Namespace(start_sec=560, end_sec=605, output_root=tmp_path)
    with pytest.raises(ValueError, match="事前登録"):
        diag.prepare(args)


def test_prepare_restores_base_duration_on_failure(monkeypatch: Any, tmp_path: Path) -> None:
    original = diag.base.MAX_DURATION_SEC
    def fail(args: Any) -> Any:
        assert diag.base.MAX_DURATION_SEC == diag.END_SEC - diag.START_SEC
        raise RuntimeError("fixture")
    monkeypatch.setattr(diag.base, "prepare", fail)
    args = argparse.Namespace(start_sec=diag.START_SEC, end_sec=diag.END_SEC,
                              output_root=tmp_path)
    with pytest.raises(RuntimeError, match="fixture"):
        diag.prepare(args)
    assert diag.base.MAX_DURATION_SEC == original


def test_update_preserves_result_and_original_mutation_without_line_trace(monkeypatch: Any) -> None:
    from src import ojama_write_accounting as accounting
    rec = recorder()
    grid = np.zeros((13, 6), dtype=np.int8)
    side = SimpleNamespace(state=SimpleNamespace(value="STABLE"),
                           confirmed_board=grid, cnn_board=grid)
    expected = SimpleNamespace(p1=side, p2=side, is_match_active=True)
    class Pipeline:
        def __init__(self) -> None:
            self.__dict__.update(pipeline_fields())
        def update(self, frame_idx: int, time_sec: float, frame: np.ndarray) -> Any:
            self._tsumo_count_1p[4] += 1
            return expected
        @classmethod
        def load_default(cls, **kwargs: Any) -> Any:
            return cls()
    def reject_trace(*args: Any) -> None:
        raise AssertionError("line traceを起動してはいけません")
    monkeypatch.setattr(sys, "settrace", reject_trace)
    original = Pipeline.update
    descriptor = Pipeline.__dict__["load_default"]
    original_filter = accounting.apply_ojama_write_accounting_filter
    pipe = Pipeline()
    rec.decoded_frame = diag.FIRST_FRAME
    with contextlib.ExitStack() as stack:
        diag.instrument_pipeline(stack, SimpleNamespace(RecognitionPipeline=Pipeline), rec)
        actual = pipe.update(diag.FIRST_FRAME, diag.START_SEC, grid)
    assert actual is expected and pipe._tsumo_count_1p[4] == 3
    assert Pipeline.update is original and Pipeline.__dict__["load_default"] is descriptor
    assert accounting.apply_ojama_write_accounting_filter is original_filter
    rows = [json.loads(line) for line in rec.stream.getvalue().splitlines()]
    assert len(rows) == 4
    assert rows[1]["before"]["tsumo_count"]["4"] == 2
    assert rows[1]["after"]["tsumo_count"]["4"] == 3
    assert rows[1]["accounting_basis_verified"] is False


@pytest.mark.parametrize("frames,loads,pipeline", [(3623, True, True), (3624, False, True),
                                                    (3624, True, False)])
def test_finish_rejects_missing_frame_or_runtime_receipt(tmp_path: Path, frames: int,
                                                       loads: bool, pipeline: bool) -> None:
    rec = recorder()
    rec.frames, rec.model_loads, rec.pipeline_receipt = frames, [1] if loads else [], {"x": 1} if pipeline else {}
    with pytest.raises(RuntimeError, match="不足"):
        diag.finish(tmp_path, {}, rec, 1.0)
    assert not (tmp_path / "COMPLETE").exists()


def test_finish_saves_three_shas_without_certifying_accounting(tmp_path: Path) -> None:
    for name in ("PLAN.json", "frames.jsonl"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    rec = recorder()
    rec.frames, rec.frame = diag.EXPECTED_FRAMES, diag.END_FRAME - diag.STRIDE
    rec.model_loads, rec.pipeline_receipt = [{"fixture": True}], {"cuda": True}
    summary = diag.finish(tmp_path, {"input_and_code_sha256": {}}, rec, 2.0)
    complete = diag.base.read_json(tmp_path / "COMPLETE")
    assert len(complete["sha256"]) == 3
    assert all(diag.base.sha256(tmp_path / name) == digest
               for name, digest in complete["sha256"].items())
    assert summary["accounting_basis_verified"] is False
    assert summary["production_adoption"] is False


def test_guard_is_rechecked_after_summary_before_complete(tmp_path: Path, monkeypatch: Any) -> None:
    asset = tmp_path / "guard.txt"
    asset.write_text("before", encoding="utf-8")
    receipt = {"input_and_code_sha256": {str(asset): diag.base.sha256(asset)}}
    for name in ("PLAN.json", "frames.jsonl"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    original_write = diag.base.write_json
    def write(path: Path, value: Any) -> None:
        original_write(path, value)
        if path.name == "SUMMARY.json":
            asset.write_text("changed", encoding="utf-8")
    monkeypatch.setattr(diag.base, "write_json", write)
    rec = recorder()
    rec.frames, rec.model_loads, rec.pipeline_receipt = diag.EXPECTED_FRAMES, [1], {"x": 1}
    with pytest.raises(ValueError, match="保護asset"):
        diag.finish(tmp_path, receipt, rec, 1.0)
    assert not (tmp_path / "COMPLETE").exists()
