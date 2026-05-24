"""
scripts.diagnose_next_acceptance のスモークテスト
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnose_next_acceptance import (
    CONSTANT_TOL,
    TARGET_COLUMN,
    compute_stats,
    is_constant,
    load_column_values,
    verify_integration,
)


# ============================
# テスト用 CSV 生成
# ============================


def _make_csv(tmp_path: Path, values: list[float], col: str = TARGET_COLUMN) -> Path:
    """指定列に値を入れた最小 CSV を作成する。"""
    p = tmp_path / "test.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", col, "label"])
        for v in values:
            w.writerow(["01", f"{v}", "1"])
    return p


# ============================
# テスト
# ============================


class TestComputeStats:
    def test_empty(self):
        s = compute_stats(np.array([], dtype=np.float64))
        assert s["n"] == 0
        assert s["std"] == 0.0

    def test_constant(self):
        s = compute_stats(np.zeros(10))
        assert s["std"] == 0.0
        assert s["unique"] == 1
        assert s["zero_ratio"] == 1.0

    def test_varying(self):
        s = compute_stats(np.array([0.1, -0.2, 0.3, 0.4]))
        assert s["std"] > 0.0
        assert s["unique"] == 4
        assert 0 <= s["zero_ratio"] <= 1


class TestIsConstant:
    def test_zero(self):
        assert is_constant(compute_stats(np.zeros(10)))

    def test_nonzero(self):
        assert not is_constant(compute_stats(np.array([0.1, 0.2, 0.3])))


class TestLoadColumnValues:
    def test_loads_values(self, tmp_path: Path):
        p = _make_csv(tmp_path, [0.0, 0.5, -0.5])
        v = load_column_values(p, TARGET_COLUMN)
        assert v.shape == (3,)

    def test_missing_column_raises(self, tmp_path: Path):
        p = _make_csv(tmp_path, [0.0])
        with pytest.raises(ValueError, match="列"):
            load_column_values(p, "nonexistent_col")


class TestVerifyIntegration:
    def test_returns_dict(self):
        out = verify_integration()
        assert "timeline_analyzer" in out
        assert "generator_v1" in out
        assert "generator_v2" in out
