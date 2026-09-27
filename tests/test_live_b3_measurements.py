"""母数・対応時刻・欠測を含むスモーク集計の試験。"""
import numpy as np
import pytest

from scripts.verify_live_degradation_20260928 import compare_cells


def test_cell_denominator_excludes_unstable_and_includes_hidden_row() -> None:
    original = np.zeros((2, 2, 13, 6), dtype=int)
    degraded = original.copy()
    degraded[0, 0, 0, 0] = 1
    degraded[1, 1, 0, 0] = 2
    stable_a = np.ones((2, 2), bool)
    stable_b = np.array([[True, True], [True, False]])
    result = compare_cells((original, stable_a), (degraded, stable_b))
    assert result['cells'] == 234 and result['different_cells'] == 1
    assert result['stable_availability_mismatches'] == 1
    assert result['first_difference'] == [0, 0, 0, 0]


def test_missing_frames_not_silently_aligned() -> None:
    with pytest.raises(ValueError):
        compare_cells((np.zeros((2, 2, 13, 6)), np.ones((2, 2), bool)),
                      (np.zeros((1, 2, 13, 6)), np.ones((1, 2), bool)))
