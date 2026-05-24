"""src/animation_filter.py のテスト (Phase T サイクル 2)。"""
from __future__ import annotations

import numpy as np
import pytest

from src.animation_filter import (
    AnimationFilter,
    DEFAULT_FRAME_DIFF_THRESHOLD,
    compute_frame_diff,
    compute_region_stats,
)


def _solid_frame(value: int) -> np.ndarray:
    return np.full((1080, 1920, 3), value, dtype=np.uint8)


def test_compute_region_stats_solid() -> None:
    f = _solid_frame(120)
    s = compute_region_stats(f, (282, 160, 384, 720))
    # V (HSV の V) はほぼフレーム値、std は 0
    assert s.v_mean == 120
    assert s.v_std == 0.0


def test_compute_frame_diff_zero() -> None:
    f = _solid_frame(50)
    assert compute_frame_diff(f, f) == 0.0


def test_compute_frame_diff_max() -> None:
    a = _solid_frame(0)
    b = _solid_frame(255)
    assert compute_frame_diff(a, b) == 255.0


def test_animation_filter_first_frame_not_animation() -> None:
    """最初のフレームは前フレームなしなので動かない。"""
    af = AnimationFilter()
    res = af.is_animation(_solid_frame(50), (282, 160, 384, 720))
    assert not res.is_animation


def test_animation_filter_unchanged_not_animation() -> None:
    af = AnimationFilter()
    f = _solid_frame(80)
    af.is_animation(f, (282, 160, 384, 720))
    res = af.is_animation(f, (282, 160, 384, 720))
    assert not res.is_animation


def test_animation_filter_flash_detected() -> None:
    """V 急上昇 (閃光) が検出される。"""
    af = AnimationFilter()
    region = (282, 160, 384, 720)
    af.is_animation(_solid_frame(40), region)
    res = af.is_animation(_solid_frame(200), region)
    assert res.is_animation
    assert "v_mean_delta" in res.reason or "frame_diff" in res.reason


def test_animation_filter_reset() -> None:
    af = AnimationFilter()
    region = (282, 160, 384, 720)
    af.is_animation(_solid_frame(40), region)
    af.reset()
    # reset 後に同じフレームを 1 つだけ渡す → 前フレームなしで is_animation=False
    res = af.is_animation(_solid_frame(40), region)
    assert not res.is_animation
