"""全フレーム同値比較が欠落や数値差を隠さないことを確認する。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.run_live_pipeline_20260928 import compare_arrays


def test_exact_comparison_and_nan(tmp_path: Path) -> None:
    left, right = tmp_path/'left.npz', tmp_path/'right.npz'
    for path in (left, right):
        np.savez(path, t_sec=[1.0, 2.0], p1=[np.nan, 0.5], adv=[0.0, 0.0])
    result = compare_arrays(left, right)
    assert result['compared_frames'] == 2
    assert result['mismatched_frames'] == 0


def test_mismatch_and_missing_tail_have_denominator(tmp_path: Path) -> None:
    left, right = tmp_path/'left.npz', tmp_path/'right.npz'
    np.savez(left, t_sec=[1.0, 2.0, 3.0], p1=[0.5, 0.6, 0.7])
    np.savez(right, t_sec=[1.0, 2.0], p1=[0.4, 0.6])
    result = compare_arrays(left, right)
    assert result['reference_frames'] == 3
    assert result['candidate_frames'] == 2
    assert result['mismatched_frames'] == 2
    assert result['first_mismatch']['t_sec'] == 1.0


def test_frame_alignment_is_compared(tmp_path: Path) -> None:
    left, right = tmp_path/'left.npz', tmp_path/'right.npz'
    np.savez(left, t_sec=[1.0, 2.0], p1=[0.5, 0.5])
    np.savez(right, t_sec=[1.1, 2.0], p1=[0.5, 0.5])
    result = compare_arrays(left, right)
    assert result['mismatched_frames'] == 1
    assert result['first_mismatch']['column'] == 't_sec'
