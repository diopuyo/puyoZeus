"""独立修正runnerの差分・排他保存・guard・一時接続を検査する。"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import diagnose_video38_boundary_repair_shadow_v1 as runner


@pytest.fixture
def short_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU契約の母数を実GPUの母数とは区別して短縮する。"""
    monkeypatch.setattr(runner, "INTERVALS", {mode: (0.0, 0.1) for mode in runner.MODULES})


def rows() -> list[dict[str, Any]]:
    """3frame左右の公開/会計を人工生成する。"""
    return [{"kind": kind, "frame_idx": frame, "time_sec": frame / 60,
             "side": side, "value": 1}
            for frame in (0, 2, 4) for side in ("1P", "2P")
            for kind in ("frame_side", "accounting_update")]


def mutation(frame: int = 2) -> dict[str, Any]:
    """実介入ではない保存契約用fixture。"""
    return {"kind": "boundary_repair_mutation", "repair": "start_epoch", "frame_idx": frame,
            "time_sec": frame / 60, "side": None, "before": {}, "after": {"x": 1}, "evidence": {}}


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    """テスト専用一時出力へ人工行を保存する。"""
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def compare_fixture(tmp_path: Path, actual: list[dict[str, Any]]) -> dict[str, Any]:
    """対照と候補を別一時pathへ保存して実比較器を呼ぶ。"""
    old, new = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    write_rows(old, rows())
    write_rows(new, actual)
    return runner.compare_rows(old, new, "start_epoch")


def test_no_intervention_is_not_quality_clear(tmp_path: Path, short_interval: None) -> None:
    result = compare_fixture(tmp_path, rows())
    assert result["all_legacy_rows_bit_exact"]
    assert result["status"] == "NO_INTERVENTION_NOT_QUALITY_CLEAR"
    assert result["quality_gate_clear"] is False


def test_post_mutation_difference_is_saved(tmp_path: Path, short_interval: None) -> None:
    actual = rows()
    actual[-1]["value"] = 2
    actual.append(mutation())
    result = compare_fixture(tmp_path, actual)
    assert result["pre_mutation_prefix_bit_exact"]
    assert len(result["changed_keys"]) == 1
    assert result["changed_field_counts"] == {"accounting_update.value": 1}
    assert result["first_mutation_frame"] == 2


@pytest.mark.parametrize("kind", ["frame_side", "accounting_update"])
def test_pre_mutation_change_rejected(tmp_path: Path, short_interval: None, kind: str) -> None:
    actual = rows()
    next(row for row in actual if row["kind"] == kind)["value"] = 4
    actual.append(mutation())
    with pytest.raises(ValueError, match="初回介入"):
        compare_fixture(tmp_path, actual)


@pytest.mark.parametrize("edit", ["missing", "duplicate", "order"])
def test_coverage_failure_rejected(tmp_path: Path, short_interval: None, edit: str) -> None:
    actual = rows()
    if edit == "missing":
        actual.pop()
    elif edit == "duplicate":
        actual.append(copy.deepcopy(actual[-1]))
    else:
        actual[-1], actual[-3] = actual[-3], actual[-1]
    actual.append(mutation())
    with pytest.raises(ValueError, match="coverage"):
        compare_fixture(tmp_path, actual)


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), -1, "0.1"])
def test_invalid_mutation_clock_rejected(value: Any, short_interval: None) -> None:
    row = mutation()
    row["time_sec"] = value
    with pytest.raises(ValueError, match="clock"):
        runner.validate_repairs([row], "start_epoch")


@pytest.mark.parametrize("value", [True, -2, 1, 6])
def test_invalid_mutation_frame_rejected(value: Any, short_interval: None) -> None:
    row = mutation()
    row["frame_idx"] = value
    with pytest.raises(ValueError, match="frame"):
        runner.validate_repairs([row], "start_epoch")


@pytest.mark.parametrize("field", ["before", "after", "evidence"])
def test_mutation_evidence_required(field: str, short_interval: None) -> None:
    row = mutation()
    del row[field]
    with pytest.raises(ValueError, match="根拠"):
        runner.validate_repairs([row], "start_epoch")


@pytest.mark.parametrize("field,value", [("repair", "chain_end"), ("kind", "boundary_repair_other")])
def test_other_repair_rejected(field: str, value: str, short_interval: None) -> None:
    row = mutation()
    row[field] = value
    with pytest.raises(ValueError, match="別mode"):
        runner.validate_repairs([row], "start_epoch")


def test_collector_missing_added_recorded(tmp_path: Path, short_interval: None) -> None:
    before, after = rows(), rows()
    old = {"kind": "collector_snapshot", "frame_idx": 2, "time_sec": 2 / 60, "side": "1P"}
    before.append(old)
    after.extend([dict(old, frame_idx=4, time_sec=4 / 60), mutation()])
    paths = (tmp_path / "old", tmp_path / "new")
    for path, values in zip(paths, (before, after), strict=True):
        write_rows(path, values)
    result = runner.compare_rows(*paths, "start_epoch")
    assert len(result["missing_keys"]) == len(result["added_keys"]) == 1
    assert len(result["missing_rows"]) == len(result["added_rows"]) == 1


def test_missing_field_and_none_are_distinct() -> None:
    assert runner.changed_fields({}, {"absent": None}) == {
        "absent": {"before_present": False, "after_present": True, "before": None, "after": None}}


def test_unknown_auxiliary_clock_preserved(tmp_path: Path, short_interval: None) -> None:
    values = rows() + [{"kind": "software_generation", "frame_idx": None, "time_sec": None}]
    paths = tmp_path / "old", tmp_path / "new"
    for path in paths:
        write_rows(path, values)
    assert runner.compare_rows(*paths, "start_epoch")["all_legacy_rows_bit_exact"]


def module_receipt(path: Path) -> dict[str, Any]:
    """仮moduleの固定SHAを保存する。"""
    return {"boundary_repair": {"module": str(path), "mode": "start_epoch",
             "module_sha256": runner.base.sha256(path)}, "input_and_code_sha256": {}}


@pytest.mark.parametrize("fail", [False, True])
def test_module_registry_restored(tmp_path: Path, fail: bool) -> None:
    path = tmp_path / "hook.py"
    path.write_text("raise RuntimeError('probe')\n" if fail else "value = 4\n", encoding="utf-8")
    name = "_video38_boundary_repair_start_epoch"
    if fail:
        with pytest.raises(RuntimeError, match="probe"), contextlib.ExitStack() as stack:
            runner.load_module(stack, module_receipt(path))
    else:
        with contextlib.ExitStack() as stack:
            assert runner.load_module(stack, module_receipt(path)).value == 4
            assert name in sys.modules
    assert name not in sys.modules


def test_module_unapproved_dependency_rejected(tmp_path: Path) -> None:
    path = tmp_path / "hook.py"
    path.write_text("REQUIRED_INPUT_SHA256={'unapproved':'bad'}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="実依存"), contextlib.ExitStack() as stack:
        runner.load_module(stack, module_receipt(path))


def test_module_hash_changed_before_load(tmp_path: Path) -> None:
    path = tmp_path / "hook.py"
    path.write_text("value=1\n", encoding="utf-8")
    receipt = module_receipt(path)
    path.write_text("value=2\n", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)), contextlib.ExitStack() as stack:
        runner.load_module(stack, receipt)


def test_finish_waits_for_all_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "result"
    output.mkdir()
    runner.base.write_json(output / "PLAN.json", {})
    (output / "frames.jsonl").write_text("", encoding="utf-8")
    comparison = {"input_sha256": {str(output / "frames.jsonl"): runner.base.sha256(output / "frames.jsonl")}}
    monkeypatch.setattr(runner, "compare_rows", lambda *args: comparison)
    def original(root: Path, *args: Any) -> dict[str, Any]:
        runner.base.write_json(root / "SUMMARY.json", {})
        names = ("PLAN.json", "frames.jsonl", "SUMMARY.json")
        runner.base.write_json(root / "COMPLETE", {"sha256": {name: runner.base.sha256(root / name) for name in names}})
        assert not (root / "COMPLETE").exists()
        return {"frame_count": 3}
    monkeypatch.setattr(runner, "ORIGINAL_HISTORY_FINISH", original)
    receipt = {"boundary_repair": {"mode": "start_epoch", "reference_root": "unused"}, "input_and_code_sha256": {}}
    runner.finish(output, receipt, None, 1.0)
    complete = runner.base.read_json(output / "COMPLETE")
    assert len(complete["sha256"]) == 4
    assert all(runner.base.sha256(output / name) == value for name, value in complete["sha256"].items())


def test_run_restores_hooks_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    original = (runner.history.prepare, runner.history.instrument_pipeline, runner.history.finish)
    def execute(args: Any) -> None:
        assert runner.history.prepare is not original[0]
        raise RuntimeError("failure")
    monkeypatch.setattr(runner.history, "run", execute)
    with pytest.raises(RuntimeError, match="failure"):
        runner.run(argparse.Namespace(mode="start_epoch"))
    assert original == (runner.history.prepare, runner.history.instrument_pipeline, runner.history.finish)


def test_function_length_limit() -> None:
    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    assert not [(node.name, node.end_lineno - node.lineno + 1) for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.end_lineno - node.lineno + 1 > 50]
