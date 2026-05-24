"""Phase I: HiddenRowFineTuner のテスト."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.board import COLOR_BLUE, COLOR_GREEN, COLOR_RED
from src.self_supervised.hidden_row_fine_tuner import (
    HiddenRowFineTuner,
    MIN_SAMPLES_FOR_FIT,
)
from src.self_supervised.pseudo_label import PseudoLabelSample


# ============================
# helper
# ============================


def _make_sample(
    p_observed: float,
    observed_color: int,
    predicted_color: int,
    *,
    with_match_meta: bool = True,
) -> PseudoLabelSample:
    """observed_color に対して predicted_dist[observed_color] = p_observed を設定."""
    dist = {observed_color: p_observed, COLOR_GREEN: 0.0}
    if predicted_color != observed_color:
        # predicted_color が dist に無い場合は補正のため追加
        dist[predicted_color] = max(p_observed, 0.5)
    metadata: dict[str, object] = {"side": "1P", "frame_idx": 0}
    if with_match_meta:
        metadata["match"] = (predicted_color == observed_color)
    return PseudoLabelSample(
        component="hidden_row",
        timestamp=0.0,
        input_data={
            "predicted_dist": dist,
            "side": "1P",
            "col": 2,
            "predicted_color": predicted_color,
        },
        label=observed_color,
        confidence=1.0 if predicted_color == observed_color else 0.0,
        metadata=metadata,
    )


# ============================
# tests
# ============================


def test_fine_tune_with_zero_samples(tmp_path: Path) -> None:
    """サンプル 0 件で fine_tune → skipped."""
    cal_path = tmp_path / "calib.json"
    ft = HiddenRowFineTuner(calibration_path=cal_path)
    metrics = ft.fine_tune([])
    assert metrics["n_samples"] == 0
    assert metrics["skipped_reason"] == "not_enough_samples"
    # ファイル未生成
    assert not cal_path.exists()


def test_fine_tune_below_min_samples(tmp_path: Path) -> None:
    cal_path = tmp_path / "calib.json"
    ft = HiddenRowFineTuner(calibration_path=cal_path)
    samples = [
        _make_sample(0.5, COLOR_RED, COLOR_RED)
        for _ in range(MIN_SAMPLES_FOR_FIT - 1)
    ]
    metrics = ft.fine_tune(samples)
    assert metrics["skipped_reason"] == "not_enough_samples"


def test_fine_tune_improves_brier(tmp_path: Path) -> None:
    """過信バイアス (predicted prob 1.0 だが observation hit rate 0.5) を補正
    → Platt scaling で確率を 0.5 へ寄せる → brier が改善."""
    cal_path = tmp_path / "calib.json"
    ft = HiddenRowFineTuner(calibration_path=cal_path)
    samples: list[PseudoLabelSample] = []
    # 50 件中 25 件 hit (predicted=1.0、actual outcome 50/50)
    for i in range(50):
        hit = i % 2 == 0
        if hit:
            samples.append(_make_sample(1.0, COLOR_RED, COLOR_RED))
        else:
            # observation は BLUE だが predicted_dist には RED:1.0 → BLUE:0.0
            samples.append(_make_sample(0.0, COLOR_BLUE, COLOR_RED))
    metrics = ft.fine_tune(samples)
    assert metrics["n_samples"] == 50
    assert metrics["brier_after"] <= metrics["brier_before"]
    # ファイル書き出し済
    assert cal_path.exists()
    data = json.loads(cal_path.read_text(encoding="utf-8"))
    assert "a" in data
    assert "b" in data


def test_calibration_file_written_atomically(tmp_path: Path) -> None:
    cal_path = tmp_path / "calib.json"
    ft = HiddenRowFineTuner(calibration_path=cal_path)
    samples = [_make_sample(0.5, COLOR_RED, COLOR_RED) for _ in range(20)]
    ft.fine_tune(samples)
    # tmp ファイルが残っていない
    assert not (tmp_path / "calib.json.tmp").exists()
    assert cal_path.exists()


def test_rollback_restores_backup(tmp_path: Path) -> None:
    cal_path = tmp_path / "calib.json"
    # 既存 calibration を作成 (古い値)
    cal_path.write_text(
        json.dumps({"a": 0.1, "b": 0.2}),
        encoding="utf-8",
    )
    ft = HiddenRowFineTuner(calibration_path=cal_path)
    # fine_tune → backup 取得 + 上書き
    samples = [_make_sample(0.5, COLOR_RED, COLOR_RED) for _ in range(20)]
    ft.fine_tune(samples)
    # rollback → 旧値に戻る
    ft.rollback()
    data = json.loads(cal_path.read_text(encoding="utf-8"))
    assert data["a"] == 0.1
    assert data["b"] == 0.2


def test_rollback_no_backup_deletes(tmp_path: Path) -> None:
    """backup が無ければ rollback で削除."""
    cal_path = tmp_path / "calib.json"
    ft = HiddenRowFineTuner(calibration_path=cal_path)
    samples = [_make_sample(0.5, COLOR_RED, COLOR_RED) for _ in range(20)]
    ft.fine_tune(samples)
    # backup 存在しない (初回 fine_tune)
    assert cal_path.exists()
    ft.rollback()
    assert not cal_path.exists()


def test_extract_pairs_filters_other_components(tmp_path: Path) -> None:
    cal_path = tmp_path / "calib.json"
    ft = HiddenRowFineTuner(calibration_path=cal_path)
    samples: list[PseudoLabelSample] = []
    samples.append(_make_sample(1.0, COLOR_RED, COLOR_RED))
    # 別 component の混入
    samples.append(PseudoLabelSample(
        component="score",
        timestamp=0.0,
        input_data={"patch": None},
        label=5,
        confidence=1.0,
    ))
    pairs = HiddenRowFineTuner._extract_pairs(samples)
    assert len(pairs) == 1


def test_extract_pairs_handles_str_keys(tmp_path: Path) -> None:
    """JSON 経由でキーが str 化していても color lookup できる."""
    sample = PseudoLabelSample(
        component="hidden_row",
        timestamp=0.0,
        input_data={
            "predicted_dist": {str(COLOR_RED): 0.8, str(COLOR_BLUE): 0.2},
            "side": "1P",
            "col": 2,
            "predicted_color": COLOR_RED,
        },
        label=COLOR_RED,
        confidence=1.0,
        metadata={"match": True},
    )
    pairs = HiddenRowFineTuner._extract_pairs([sample])
    assert len(pairs) == 1
    p, hit = pairs[0]
    assert pytest.approx(p) == 0.8
    assert hit == 1
