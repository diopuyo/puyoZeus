"""
scripts/feature_selection.py のスモークテスト
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from scripts.old.eda_features import Dataset, load_dataset  # noqa: E402
from scripts.old.feature_selection import (  # noqa: E402
    FEATURE_COUNT_GRID,
    L1_C_GRID,
    L1SelectionResult,
    ParetoPoint,
    best_pareto_point,
    evaluate_subset,
    plot_pareto,
    rank_by_rf_importance,
    select_indices_top_n,
    select_subset,
    select_via_l1,
    sweep_pareto,
)
from scripts.old.generate_training_dataset import (  # noqa: E402
    DEFAULT_TIME_PHASES,
    FEATURE_NAMES,
)
from scripts.old.learn_weights_v2 import random_split  # noqa: E402


def _make_csv(tmp_path: Path, n: int = 150) -> Path:
    """学習可能な合成 CSV (第 1, 2 特徴量で完全分離)。"""
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
            video = ["01", "02", "03"][i % 3]
            phase = DEFAULT_TIME_PHASES[i % 5]
            row: list[object] = [video, i, phase]
            for j, _name in enumerate(FEATURE_NAMES):
                if j == 0:
                    row.append(float(label) * 1.5 + rng.normal(0, 0.1))
                elif j == 1:
                    row.append(float(label) * 0.7 + rng.normal(0, 0.2))
                else:
                    row.append(rng.normal(0, 1.0))
            row.append(label)
            w.writerow(row)
    return csv_path


def _ds(tmp_path: Path) -> Dataset:
    return load_dataset(_make_csv(tmp_path))


# ============================
# select_subset
# ============================


def test_select_subset_preserves_rows(tmp_path: Path) -> None:
    """select_subset 後も行数が保たれ、列は指定数になる。"""
    ds = _ds(tmp_path)
    sub = select_subset(ds, [0, 1, 2])
    assert sub.X.shape == (len(ds.y), 3)
    assert sub.feature_names == ds.feature_names[:3]


# ============================
# rank / select
# ============================


def test_rank_by_rf_importance_returns_all_features(tmp_path: Path) -> None:
    """RF ランキングに全特徴量が含まれる。"""
    ds = _ds(tmp_path)
    split = random_split(ds)
    ranking = rank_by_rf_importance(ds, split)
    assert len(ranking) == len(ds.feature_names)
    # 第 1 特徴量が最重要 (合成データ)
    assert ranking[0][0] in {ds.feature_names[0], ds.feature_names[1]}


def test_select_indices_top_n(tmp_path: Path) -> None:
    """select_indices_top_n は N 個のインデックスを返す。"""
    ds = _ds(tmp_path)
    split = random_split(ds)
    ranking = rank_by_rf_importance(ds, split)
    idx = select_indices_top_n(ds.feature_names, ranking, 5)
    assert len(idx) == 5
    assert all(0 <= i < len(ds.feature_names) for i in idx)


# ============================
# L1 selection
# ============================


def test_select_via_l1_returns_dataclass(tmp_path: Path) -> None:
    """select_via_l1 は L1SelectionResult を返す。"""
    ds = _ds(tmp_path)
    split = random_split(ds)
    res = select_via_l1(ds, split, C=0.1)
    assert isinstance(res, L1SelectionResult)
    assert isinstance(res.selected_features, list)


# ============================
# evaluate_subset
# ============================


def test_evaluate_subset_returns_pareto_point(tmp_path: Path) -> None:
    """evaluate_subset が ParetoPoint を返す。"""
    ds = _ds(tmp_path)
    split = random_split(ds)
    pt = evaluate_subset(ds, [0, 1], split, "manual", 2)
    assert isinstance(pt, ParetoPoint)
    assert pt.n_features == 2
    assert pt.test_acc > 0.6   # 有意な特徴量を渡しているので高め


# ============================
# sweep_pareto
# ============================


def test_sweep_pareto_returns_points(tmp_path: Path) -> None:
    """sweep_pareto が Pareto 点リストを返す。"""
    ds = _ds(tmp_path)
    split = random_split(ds)
    points = sweep_pareto(ds, split)
    assert len(points) >= len(FEATURE_COUNT_GRID) - 1


def test_best_pareto_point(tmp_path: Path) -> None:
    """best_pareto_point は test_acc 最大を返す。"""
    pts = [
        ParetoPoint("rf", 5, "split_a", 0.9, 0.5, ["a"]),
        ParetoPoint("rf", 8, "split_a", 0.9, 0.6, ["a", "b"]),
    ]
    best = best_pareto_point(pts)
    assert best.test_acc == 0.6


# ============================
# plot_pareto
# ============================


def test_plot_pareto_creates_file(tmp_path: Path) -> None:
    """plot_pareto が PNG を生成する。"""
    pts = [
        ParetoPoint("rf_topN", 5, "random_0.70", 0.9, 0.5, []),
        ParetoPoint("rf_topN", 8, "random_0.70", 0.92, 0.55, []),
    ]
    out = tmp_path / "pareto.png"
    plot_pareto(pts, out)
    assert out.exists() and out.stat().st_size > 0
