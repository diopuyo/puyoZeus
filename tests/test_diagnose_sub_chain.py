"""
scripts.diagnose_sub_chain のスモークテスト
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnose_sub_chain import (
    COL_INDEP,
    COL_QUALITY,
    HIGH_REDUNDANCY_THRESHOLD,
    col_stats,
    decide_recommendation,
    linear_fit_r2,
    load_two_columns,
    pearson_corr,
)


# ============================
# テスト用 CSV
# ============================


def _make_csv(
    tmp_path: Path, q: list[float], i: list[float],
) -> Path:
    """sub_chain_quality / sub_chain_independence を持つ CSV を作成。"""
    p = tmp_path / "test.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([COL_QUALITY, COL_INDEP, "label"])
        for a, b in zip(q, i):
            w.writerow([f"{a}", f"{b}", "1"])
    return p


# ============================
# テスト
# ============================


class TestPearsonCorr:
    def test_perfect(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        assert abs(pearson_corr(x, y) - 1.0) < 1e-9

    def test_anti(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([2.0, 1.0, 0.0])
        assert abs(pearson_corr(x, y) + 1.0) < 1e-9

    def test_uncorrelated(self):
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([0.0, 1.0, 2.0])
        assert pearson_corr(x, y) == 0.0


class TestLinearFitR2:
    def test_perfect_fit(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = 2 * x + 1
        a, b, r2 = linear_fit_r2(x, y)
        assert abs(a - 2.0) < 1e-6
        assert abs(b - 1.0) < 1e-6
        assert abs(r2 - 1.0) < 1e-6

    def test_constant_x(self):
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([0.0, 1.0, 2.0])
        a, b, r2 = linear_fit_r2(x, y)
        assert a == 0.0
        assert r2 == 0.0


class TestDecideRecommendation:
    def test_high_redundancy_drops_one(self):
        q = {"abs_mean": 0.5}
        i = {"abs_mean": 0.3}
        d = decide_recommendation(0.95, 0.9, q, i)
        assert d["decision"] == "DROP_ONE"
        assert d["drop_target"] == COL_INDEP

    def test_keep_both_when_weak(self):
        d = decide_recommendation(0.1, 0.01, {"abs_mean": 0}, {"abs_mean": 0})
        assert d["decision"] == "KEEP_BOTH"

    def test_review_when_moderate(self):
        d = decide_recommendation(0.6, 0.36, {"abs_mean": 0}, {"abs_mean": 0})
        assert d["decision"] == "REVIEW"


class TestLoadTwoColumns:
    def test_basic(self, tmp_path: Path):
        p = _make_csv(tmp_path, [0.1, 0.2, 0.3], [0.5, 0.6, 0.7])
        a, b = load_two_columns(p, COL_QUALITY, COL_INDEP)
        assert a.shape == (3,) and b.shape == (3,)
        assert pytest.approx(a[0]) == 0.1
        assert pytest.approx(b[2]) == 0.7
