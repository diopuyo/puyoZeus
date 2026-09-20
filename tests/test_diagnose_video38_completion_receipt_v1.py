"""completion receiptの欠測fail-closed・anchor凍結・透過計装をCPU検査する。"""

from __future__ import annotations

import ast
import contextlib
import io
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from scripts import diagnose_video38_completion_receipt_v1 as diag


def _board(row1: int = 0, row0: int = 0) -> dict[str, Any]:
    grid = np.zeros((13, 6), dtype=np.int8)
    grid[1, :] = row1
    grid[0, :] = row0
    return {"grid": grid.tolist(), "sha256": "fixture", "color": 0,
            "garbage": 0, "unknown": 0}


def _rec() -> diag.CompletionRecorder:
    rec = diag.CompletionRecorder(io.StringIO(), _board())
    rec.begin_frame(100, 10.0)
    rec.lineages.append({"lineage_id": 1, "side": "2P", "start_frame": 100,
                         "start_time_sec": 10.0, "event": {},
                         "entry_score_anchor": None,
                         "anchor_status": "unknown_until_raw_score_jump",
                         "prediction": None})
    rec.current_lineage["2P"] = 1
    return rec


def _delta(prev: int | None, cur: int | None) -> Any:
    delta = 0 if prev is None or cur is None else cur - prev
    return SimpleNamespace(side="2P", prev_score=prev, cur_score=cur, delta=delta,
                           is_valid=prev is not None and cur is not None and delta >= 0)


def test_first_positive_raw_delta_freezes_anchor_across_later_revisions() -> None:
    rec = _rec()
    rec.record_score(_delta(None, None), None)
    assert rec.lineages[0]["entry_score_anchor"] is None
    rec.begin_frame(101, 10.1); rec.record_score(_delta(971, 2311), 2311)
    rec.begin_frame(102, 10.2); rec.record_score(_delta(2311, 6151), 6151)
    assert rec.lineages[0]["entry_score_anchor"] == 971
    assert rec.lineages[0]["anchor_source_frame"] == 101
    assert rec.lineages[0]["anchor_source"].startswith("ScoreDelta.prev_score")
    assert rec.score_rows[1]["frame_idx"] == 101
    assert rec.score_rows[1]["time_sec"] == 10.1


def test_missing_anchor_and_raw_read_remain_unknown_not_zero_or_pass() -> None:
    lineage = _rec().lineages[0]
    rows = [{"frame_idx": 1, "time_sec": 1.0, "raw_value": None}]
    evidence = diag._score_evidence(lineage, rows, 79080)
    assert evidence["anchor"] is None
    assert evidence["timeline"][0]["gain_from_anchor"] is None
    assert evidence["timeline"][0]["status"] == "unknown"
    assert evidence["first_numeric_exact_candidate"] is None


def test_early_visible_final_does_not_close_before_score_total() -> None:
    rec = _rec(); rec.verify_frames = 2
    rec.lineages[0]["entry_score_anchor"] = 971
    rec.lineages[0]["anchor_status"] = "candidate_unverified_generation"
    pred = {"chain_count": 13, "calculated_total_score": 79080,
            "final_board": _board()}
    rec.lineages[0]["prediction"] = pred
    rec.score_rows.append({"lineage_id": 1, "frame_idx": 2, "time_sec": 2.0,
                           "raw_value": 2311})
    for frame in (2, 3):
        rec.frame_rows.append({"lineage_id": 1, "side": "2P", "frame_idx": frame,
                               "time_sec": frame / 10, "state": "stable",
                               "raw_before_accounting": _board()})
    rec.prediction_rows.append({"kind": "physical_prediction", "prediction_id": 1,
                                "lineage_id": 1, "side": "2P", "caller": "fixture",
                                "frame_idx": 1, "time_sec": 0.1, "prediction": pred})
    row = diag.build_receipt(rec)["prediction_revisions"][0]
    assert row["visible_and_row0_evidence"]["max_exact_run"] == 2
    assert row["score_evidence"]["first_numeric_exact_candidate"] is None
    assert row["combined_closure_status"].startswith("not_evaluated")
    assert row["commit_permission_issued"] is False


def test_raw_row1_gravity_evidence_is_separate_from_candidate_row0() -> None:
    rows = [{"frame_idx": frame, "time_sec": frame / 30, "state": "stable",
             "raw_before_accounting": _board(row1=0)} for frame in range(5)]
    evidence = diag._consensus_evidence(rows, _board(row0=0), 5)
    assert evidence["gravity_empty_columns"] == list(range(6))
    assert evidence["row0_candidate_supported"] is True
    assert "raw STABLE row1" in evidence["row0_source"]
    assert evidence["commit_permission_issued"] is False
    assert diag._consensus_evidence(rows, _board(row0=1), 5)["row0_candidate_supported"] is False


def test_unknown_visible_cell_cannot_form_known_consensus() -> None:
    raw, final = _board(), _board()
    raw["grid"][5][2] = 10
    final["grid"][5][2] = 10
    rows = [{"frame_idx": frame, "time_sec": frame / 30, "state": "stable",
             "raw_before_accounting": raw} for frame in range(5)]
    assert diag._consensus_evidence(rows, final, 5)["max_exact_run"] == 0


def test_score_wrapper_returns_exact_original_object_and_restores() -> None:
    class Tracker:
        def _apply_read(self, cur: int | None) -> Any:
            return _delta(10, cur)
    original, tracker, rec = Tracker._apply_read, Tracker(), _rec()
    with contextlib.ExitStack() as stack:
        diag.instrument_score(stack, Tracker, rec)
        result = tracker._apply_read(20)
        assert result.cur_score == 20
    assert Tracker._apply_read is original


def test_formula_wrapper_preserves_signature_result_and_restores() -> None:
    class Accum:
        step_count = 0
        total_power = 0
        last_valid_t = None
        def update(self, value: Any) -> Any:
            return value
        def _reset_session(self) -> None:
            return None
    original, before = Accum.update, inspect.signature(Accum.update)
    step = SimpleNamespace(t_sec=1.0, left=4, right=10, product=40)
    with contextlib.ExitStack() as stack:
        diag.instrument_formula(stack, Accum, _rec())
        owner = Accum()
        assert owner.update(step) is step
        assert inspect.signature(Accum.update) == before
    assert Accum.update is original


def test_prepare_requires_fixed_interval_and_guards_owned_files(monkeypatch: Any) -> None:
    monkeypatch.setattr(diag, "ORIGINAL_COMPLETION_PREPARE",
                        lambda _args: ({"input_and_code_sha256": {},
                                        "shadow_policy": {}, "shadow_references": {}}, {}))
    monkeypatch.setattr(diag.prechain, "_reference_receipt", lambda _root: ({}, {}))
    args = SimpleNamespace(start_sec=560.0, end_sec=605.0)
    receipt, _ = diag.completion_prepare(args)
    for path in (Path(diag.__file__), diag.LAUNCHER, diag.TEST, diag.DOC):
        assert receipt["input_and_code_sha256"][str(path)] == diag.base.sha256(path)
    assert receipt["completion_receipt_scope"]["commit_or_publication_modified"] is False


def test_run_patches_window_and_chain_then_restores(monkeypatch: Any) -> None:
    originals = [(diag.base, "MAX_DURATION_SEC", diag.base.MAX_DURATION_SEC),
                 (diag.base, "END_SEC", diag.base.END_SEC),
                 (diag.atomic, "ORIGINAL_PREPARE", diag.atomic.ORIGINAL_PREPARE),
                 (diag.atomic, "AtomicRecorder", diag.atomic.AtomicRecorder),
                 (diag.atomic, "atomic_instrument", diag.atomic.atomic_instrument),
                 (diag.prechain, "shadow_finish", diag.prechain.shadow_finish)]
    def fake_run(_args: Any) -> dict[str, bool]:
        assert diag.base.MAX_DURATION_SEC == 45 and diag.base.END_SEC == 605
        assert diag.atomic.ORIGINAL_PREPARE is diag.completion_prepare
        assert diag.atomic.AtomicRecorder is diag.CompletionRecorder
        assert diag.prechain.shadow_finish is diag.completion_finish
        return {"ok": True}
    monkeypatch.setattr(diag.atomic, "run", fake_run)
    assert diag.run(SimpleNamespace()) == {"ok": True}
    for owner, name, value in originals:
        assert getattr(owner, name) is value


def test_finish_writes_receipt_and_attaches_sha(tmp_path: Path, monkeypatch: Any) -> None:
    rec = _rec(); rec.frames = 1350
    monkeypatch.setattr(diag, "_prefix_comparison", lambda _path: {"ok": True})
    def fake_finish(output: Path, _receipt: Any, _recorder: Any,
                    _elapsed: float) -> dict[str, Any]:
        diag.base.write_json(output / "COMPLETE", {"sha256": {}})
        return {"ok": True}
    monkeypatch.setattr(diag, "ORIGINAL_SHADOW_FINISH", fake_finish)
    result = diag.completion_finish(tmp_path, {"input_and_code_sha256": {}}, rec, 1.0)
    complete = diag.base.read_json(tmp_path / "COMPLETE")
    assert result["completion_receipt"]["commit_permission_issued"] is False
    assert complete["sha256"]["COMPLETION_RECEIPT.json"] == diag.base.sha256(
        tmp_path / "COMPLETION_RECEIPT.json")


def test_new_functions_obey_fifty_line_limit() -> None:
    for path in (Path(diag.__file__), Path(__file__)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno - node.lineno + 1 <= 50, (path.name, node.name)
