"""prechain shadowの単一軸・復元・receiptをCPU fixtureで検査する。"""

from __future__ import annotations

import ast
import contextlib
import io
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import diagnose_video38_prechain_shadow_v1 as shadow


class Board:
    """copyとgridを持つ最小盤面。"""

    def __init__(self, count: int) -> None:
        self._grid = np.zeros((13, 6), np.int8)
        if count > 0:
            self._grid.reshape(-1)[-count:] = 3

    def copy(self) -> "Board":
        value = Board(0)
        value._grid = self._grid.copy()
        return value


def recorder() -> shadow.ShadowRecorder:
    """対象frameに固定したメモリ内recorder。"""
    rec = shadow.ShadowRecorder(io.StringIO(), shadow.base.board_value(Board(9)))
    rec.begin_frame(shadow.base.TARGET_FRAME, 578.366667)
    return rec


def policy_rows(rec: shadow.ShadowRecorder) -> list[dict[str, Any]]:
    """shadow policy行だけを読む。"""
    return [json.loads(line) for line in rec.stream.getvalue().splitlines()
            if json.loads(line).get("kind") == "shadow_policy"]


def install(module: Any, rec: shadow.ShadowRecorder) -> contextlib.ExitStack:
    """候補wrapperをstackへ設置する。"""
    stack = contextlib.ExitStack()
    shadow.instrument_shadow_resolve(stack, module, rec)
    return stack


def test_normal_chain_returns_prechain_copy_and_records_both_grids() -> None:
    prechain, final, rec = Board(68), Board(9), recorder()
    def original(new_confirmed: Board, chain_sim: Any, **kwargs: Any) -> tuple[Board, int]:
        return final, 13
    module = SimpleNamespace(resolve_after_placement=original)
    with install(module, rec):
        returned, count = module.resolve_after_placement(prechain, None, score_delta_observed=1)
    row = policy_rows(rec)[0]
    assert count == 13 and returned is not prechain and returned._grid.tolist() == prechain._grid.tolist()
    assert row["original_final"]["color"] == 9
    assert row["returned_prechain"]["color"] == 68
    returned._grid[:] = 0
    assert int(np.isin(prechain._grid, (1, 2, 3, 4, 5)).sum()) == 68


def test_no_chain_returns_original_tuple_without_copy() -> None:
    prechain, rec = Board(20), recorder()
    result = (prechain, 0)
    def original(new_confirmed: Board, chain_sim: Any) -> tuple[Board, int]:
        return result
    module = SimpleNamespace(resolve_after_placement=original)
    with install(module, rec):
        actual = module.resolve_after_placement(prechain, None)
    assert actual is result
    assert policy_rows(rec)[0]["policy_applied"] is False


def test_guard_rejection_is_exact_passthrough() -> None:
    inferred, rec = Board(68), recorder()
    guarded = (inferred, 0)
    def original(new_confirmed: Board, chain_sim: Any,
                 prev_confirmed: Board | None = None) -> tuple[Board, int]:
        return guarded
    module = SimpleNamespace(resolve_after_placement=original)
    with install(module, rec):
        actual = module.resolve_after_placement(inferred, None, prev_confirmed=Board(66))
    assert actual is guarded and actual[0] is inferred
    assert policy_rows(rec)[0]["returned_prechain"]["color"] == 68


def test_original_exception_propagates_and_wrapper_restores() -> None:
    rec = recorder()
    def original(new_confirmed: Board, chain_sim: Any) -> tuple[Board, int]:
        raise LookupError("fixture")
    module = SimpleNamespace(resolve_after_placement=original)
    with pytest.raises(LookupError), install(module, rec):
        module.resolve_after_placement(Board(10), None)
    assert module.resolve_after_placement is original
    assert policy_rows(rec)[0]["status"] == "original_exception"


def test_wrapper_preserves_signature_and_input() -> None:
    rec, value = recorder(), Board(30)
    def original(new_confirmed: Board, chain_sim: Any,
                 score_delta_observed: int | None = None) -> tuple[Board, int]:
        return Board(4), 2
    module = SimpleNamespace(resolve_after_placement=original)
    before = value._grid.tobytes()
    with install(module, rec):
        assert inspect.signature(module.resolve_after_placement) == inspect.signature(original)
        module.resolve_after_placement(value, None, score_delta_observed=1)
    assert value._grid.tobytes() == before
    assert policy_rows(rec)[0]["input_unchanged"] is True


def test_same_wrapper_applies_to_main_and_secondary_callers() -> None:
    rec, calls = recorder(), []
    def original(new_confirmed: Board, chain_sim: Any) -> tuple[Board, int]:
        calls.append(len(calls))
        return Board(1), 1
    module = SimpleNamespace(resolve_after_placement=original)
    def main_path() -> Any:
        return module.resolve_after_placement(Board(10), None)
    def secondary_path() -> Any:
        return module.resolve_after_placement(Board(11), None)
    with install(module, rec):
        first, second = main_path(), secondary_path()
    assert calls == [0, 1] and first[0]._grid.sum() != second[0]._grid.sum()
    assert [row["input_prechain"]["color"] for row in policy_rows(rec)] == [10, 11]


def test_pipeline_caller_records_step_side() -> None:
    def _step_side(side: str) -> dict[str, Any]:
        return shadow._pipeline_caller(inspect.currentframe())
    value = _step_side("2P")
    assert value["caller_function"] == "_step_side"
    assert value["side"] == "2P"


def test_shadow_is_installed_before_detail_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    class Pipeline:
        def resolve_after_placement(self) -> None:
            return None
    module = SimpleNamespace(RecognitionPipeline=Pipeline)
    monkeypatch.setitem(shadow.sys.modules, Pipeline.__module__, module)
    monkeypatch.setattr(shadow, "instrument_shadow_resolve",
                        lambda *args: events.append("shadow"))
    monkeypatch.setattr(shadow, "ORIGINAL_DETAIL_INSTRUMENT",
                        lambda *args: events.append("details"))
    with contextlib.ExitStack() as stack:
        shadow.shadow_instrument(stack, module, recorder())
    assert events == ["shadow", "details"]


def test_prepare_adds_new_files_and_references(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shadow, "ORIGINAL_DETAIL_PREPARE",
                        lambda args: ({"input_and_code_sha256": {}}, {}))
    receipt, _ = shadow.shadow_prepare(SimpleNamespace(start_sec=560.0, end_sec=583.0))
    hashes = receipt["input_and_code_sha256"]
    for path in (Path(shadow.__file__), shadow.LAUNCHER, shadow.TEST, shadow.FIX_PLAN):
        assert hashes[str(path)] == shadow.base.sha256(path)
    assert set(receipt["shadow_references"]) == {"base", "details"}
    assert receipt["shadow_policy"]["pending_rollback_issue"] == "known_unfixed_separate_axis"
    assert receipt["shadow_policy"]["stale_next_exit_issue"] == "known_unfixed_separate_axis"


def test_reference_requires_all_base_artifact_hashes(tmp_path: Path) -> None:
    (tmp_path / "COMPLETE").write_text(json.dumps({"sha256": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="必須SHA receipt"):
        shadow._reference_receipt(tmp_path)


def test_shadow_summary_is_separate_and_hashed_in_complete(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    asset = tmp_path / "asset"; asset.write_text("fixed", encoding="utf-8")
    for name in ("PLAN.json", "frames.jsonl"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    receipt = {"input_and_code_sha256": {str(asset): shadow.base.sha256(asset)},
               "shadow_policy": {}, "shadow_references": {},
               "runtime": {"native_identity": {"original_extension_identity_verified": False}}}
    rec = recorder(); rec.frames = 1; rec.model_loads = [{}]; rec.pipeline_receipt = {}
    rec.pipeline_receipt["cuda"] = True
    shadow.shadow_finish(tmp_path, receipt, rec, 0.1)
    complete = shadow.base.read_json(tmp_path / "COMPLETE")
    assert complete["sha256"]["SHADOW_SUMMARY.json"] == shadow.base.sha256(
        tmp_path / "SHADOW_SUMMARY.json")
    assert shadow.base.read_json(tmp_path / "SHADOW_SUMMARY.json")["adoption"] is False


def test_expanded_interval_rejected() -> None:
    with pytest.raises(ValueError):
        shadow.shadow_prepare(SimpleNamespace(start_sec=559.0, end_sec=583.0))


def test_new_functions_obey_fifty_line_limit() -> None:
    for path in (Path(shadow.__file__), Path(__file__)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno - node.lineno + 1 <= 50, (path.name, node.name)
