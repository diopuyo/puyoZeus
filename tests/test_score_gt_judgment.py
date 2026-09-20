"""偏った標本の参考採点を、被覆不備や全体合格に読み替えない。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import score_gt_judgment as subject


def _picks(path: Path) -> Path:
    """1セルの抽出契約を作る。"""
    path.write_text(json.dumps({"picks": [
        {"side": "1P", "frame": 30, "cells": [[12, 0, 1, 2]]},
    ]}), encoding="utf-8")
    return path


def _judgment() -> dict:
    """抽出と一致する人手判定を返す。"""
    return dict(side="1P", frame=30, row=12, col=0, verdict="screen")


@pytest.mark.parametrize("case", ["missing", "duplicate", "outside"])
def test_invalid_coverage_is_rejected(tmp_path: Path, case: str) -> None:
    """欠測や重複を成功票に混ぜない。"""
    rows = {"missing": [], "duplicate": [_judgment(), _judgment()],
            "outside": [_judgment(), dict(_judgment(), frame=31)]}[case]
    with pytest.raises(ValueError, match="被覆"):
        subject.verify_coverage(rows, _picks(tmp_path / "picks.json"))


def test_matching_coverage_is_accepted(tmp_path: Path) -> None:
    """正常な人手判定は同じ母数のまま受け入れる。"""
    result = subject.verify_coverage([_judgment()], _picks(tmp_path / "picks.json"))
    assert result["抽出したセル"] == result["判定したセル"] == 1


def test_unknown_only_has_no_decided_estimate() -> None:
    """判定できたセルがゼロなら100%を捏造しない。"""
    result = subject.estimate_accuracy(dict(recorded=0, screen=0, unknown=3), 100, 10)
    assert result["判定できた分で外挿"] is None


def test_reference_score_never_claims_quality_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """旧点推定を残しながら、信頼区間と合格は明確に否定する。"""
    kit = tmp_path / "logs/diag_gt/kit_video_test"
    kit.mkdir(parents=True)
    picks_path = _picks(kit / "picks.json")
    picks = json.loads(picks_path.read_text())
    picks.update(source_detail="receipt/npz_detail.jsonl", n_boards_with_mismatch=1,
                 n_top=1, n_random=0)
    picks_path.write_text(json.dumps(picks), encoding="utf-8")
    receipt = tmp_path / "receipt"
    receipt.mkdir()
    (receipt / "receipt.json").write_text(json.dumps({"denominators_and_counts": {
        "npz_basis_cellframes_total": 1000, "npz_mismatch_cellframes_total": 1,
        "steps_recorded_to_npz": {"1P": 1},
    }}), encoding="utf-8")
    judged = tmp_path / "judged.json"
    judged.write_text(json.dumps([_judgment()]), encoding="utf-8")
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    result = subject.score_video("test", judged)
    assert result["quality_gate_clear"] is False
    assert result["confidence_interval"] is None
    assert result["母数"]["user が判定したセル"] == 1
