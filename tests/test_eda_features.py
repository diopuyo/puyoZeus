"""
scripts/eda_features.py のスモークテスト

合成データで CSV 読み込み・相関計算・冗長ペア検出を検証する。
matplotlib は Agg バックエンドで描画のみテストする。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.old.eda_features import (
    REDUNDANCY_THRESHOLD,
    correlations_per_phase,
    find_redundant_pairs,
    load_dataset,
    pairwise_correlation,
    pearson_with_label,
    plot_label_correlation_heatmap,
    plot_pairwise_heatmap,
    write_report,
)
from scripts.old.generate_training_dataset import (
    DEFAULT_TIME_PHASES,
    FEATURE_NAMES,
)


def _make_csv(tmp_path: Path, n: int = 30) -> Path:
    """テスト用 CSV を生成する。第 1 特徴量だけがラベルと完全相関。"""
    csv_path = tmp_path / "match_features.csv"
    rng = np.random.default_rng(42)
    header = ["video_id", "match_idx", "time_phase"]
    header.extend(FEATURE_NAMES)
    header.append("label")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n):
            label = 1 if i % 2 == 0 else -1
            row: list[object] = ["01", i, DEFAULT_TIME_PHASES[i % 5]]
            for j, _name in enumerate(FEATURE_NAMES):
                if j == 0:
                    row.append(float(label) + rng.normal(0, 0.05))
                elif j == 1:
                    # 0 と完全相関 (冗長ペア検出用)
                    row.append(float(label) + rng.normal(0, 0.05))
                else:
                    row.append(rng.normal(0, 1.0))
            row.append(label)
            w.writerow(row)
    return csv_path


def test_load_dataset_round_trip(tmp_path: Path) -> None:
    """CSV → Dataset の往復で行数が保たれる。"""
    csv_path = _make_csv(tmp_path, n=20)
    ds = load_dataset(csv_path)
    assert ds.X.shape == (20, len(FEATURE_NAMES))
    assert ds.y.shape == (20,)
    assert set(ds.y.tolist()) == {-1, 1}


def test_pearson_with_label_detects_signal(tmp_path: Path) -> None:
    """第 1 特徴量とラベルの相関が高いことを確認する.

    第 3 列以降は rng.normal の独立ノイズで、FEATURE_NAMES 長 (47) と seed=42
    に依存して corr[2] が 0.31 程度になりうる。0.4 で安全マージン (信号系の
    0.8 と十分差をつける).
    """
    csv_path = _make_csv(tmp_path, n=50)
    ds = load_dataset(csv_path)
    corr = pearson_with_label(ds.X, ds.y)
    assert corr.shape == (len(FEATURE_NAMES),)
    assert abs(corr[0]) > 0.8  # 強相関
    assert abs(corr[2]) < 0.4  # 弱相関 (seed/長さ依存ノイズに耐える閾値)


def test_pairwise_correlation_diagonal_is_one(tmp_path: Path) -> None:
    """対角成分が 1.0 であること。"""
    csv_path = _make_csv(tmp_path)
    ds = load_dataset(csv_path)
    M = pairwise_correlation(ds.X)
    assert M.shape == (len(FEATURE_NAMES), len(FEATURE_NAMES))
    np.testing.assert_allclose(np.diag(M), np.ones(len(FEATURE_NAMES)))


def test_find_redundant_pairs(tmp_path: Path) -> None:
    """強相関した 2 つの特徴量が冗長ペアとして検出される。"""
    csv_path = _make_csv(tmp_path, n=50)
    ds = load_dataset(csv_path)
    M = pairwise_correlation(ds.X)
    pairs = find_redundant_pairs(ds.feature_names, M, REDUNDANCY_THRESHOLD)
    assert any(
        {a, b} == {ds.feature_names[0], ds.feature_names[1]}
        for a, b, _ in pairs
    )


def test_correlations_per_phase_keys(tmp_path: Path) -> None:
    """time_phase 別相関の辞書キーが DEFAULT_TIME_PHASES と一致する。"""
    csv_path = _make_csv(tmp_path, n=25)
    ds = load_dataset(csv_path)
    phase_corr = correlations_per_phase(ds)
    assert set(phase_corr.keys()) == set(DEFAULT_TIME_PHASES)
    for v in phase_corr.values():
        assert v.shape == (len(FEATURE_NAMES),)


@pytest.mark.parametrize("name", ["heatmap", "pairwise"])
def test_plotting_smoke(tmp_path: Path, name: str) -> None:
    """ヒートマップ描画がエラーなく PNG を生成できる。"""
    csv_path = _make_csv(tmp_path)
    ds = load_dataset(csv_path)
    overall = pearson_with_label(ds.X, ds.y)
    phase_corr = correlations_per_phase(ds)
    out = tmp_path / f"{name}.png"
    if name == "heatmap":
        plot_label_correlation_heatmap(
            ds.feature_names, overall, phase_corr, out,
        )
    else:
        M = pairwise_correlation(ds.X)
        plot_pairwise_heatmap(ds.feature_names, M, out)
    assert out.exists() and out.stat().st_size > 0


def test_write_report(tmp_path: Path) -> None:
    """Markdown レポートが書き出される。"""
    csv_path = _make_csv(tmp_path)
    ds = load_dataset(csv_path)
    overall = pearson_with_label(ds.X, ds.y)
    phase_corr = correlations_per_phase(ds)
    M = pairwise_correlation(ds.X)
    pairs = find_redundant_pairs(ds.feature_names, M)
    out = tmp_path / "report.md"
    write_report(ds, overall, phase_corr, pairs, out)
    text = out.read_text(encoding="utf-8")
    assert "EDA レポート" in text
    assert ds.feature_names[0] in text
