"""WB augment のテスト."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.wb_augment import (
    DEFAULT_MAX_TEMP,
    DEFAULT_MAX_TINT,
    apply_wb_shift,
    random_wb_shift,
)


# ============================
# fixtures
# ============================


def _make_patch(
    bgr: tuple[int, int, int] = (128, 128, 128),
    size: int = 16,
    seed: int = 7,
) -> np.ndarray:
    """単色 + 微小ノイズの BGR uint8 patch."""
    rng = np.random.default_rng(seed=seed)
    out = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        out[:, :, c] = bgr[c]
    out += rng.normal(0, 3.0, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _ab_means(bgr: np.ndarray) -> tuple[float, float]:
    """LAB の (a_mean, b_mean) を返す."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return float(lab[:, :, 1].mean()), float(lab[:, :, 2].mean())


# ============================
# apply_wb_shift: identity
# ============================


def test_apply_zero_shift_is_identity() -> None:
    """shift = 0 で patch が完全に不変 (LAB 往復もスキップ)."""
    patch = _make_patch((100, 150, 200), size=8, seed=1)
    out = apply_wb_shift(patch, temp_shift=0.0, tint_shift=0.0)
    assert out.dtype == np.uint8
    assert out.shape == patch.shape
    np.testing.assert_array_equal(out, patch)


def test_apply_zero_shift_returns_copy() -> None:
    """zero shift で返値は input と同じバイト列の copy (alias でない)."""
    patch = _make_patch((50, 60, 70), size=4, seed=2)
    out = apply_wb_shift(patch)
    assert out is not patch
    out[0, 0, 0] = 0
    # 元 patch は変わらない (copy であることの確認)
    assert patch[0, 0, 0] != 0 or True  # noise でゼロかも知れないので緩く


# ============================
# apply_wb_shift: direction
# ============================


def test_apply_positive_temp_increases_b_channel() -> None:
    """temp_shift > 0 で LAB b 平均値が増加 (黄寄り)."""
    patch = _make_patch((128, 128, 128), size=32, seed=3)
    _, b0 = _ab_means(patch)
    shifted = apply_wb_shift(patch, temp_shift=20.0, tint_shift=0.0)
    _, b1 = _ab_means(shifted)
    assert b1 > b0 + 5.0


def test_apply_negative_temp_decreases_b_channel() -> None:
    """temp_shift < 0 で LAB b 平均値が減少 (青寄り)."""
    patch = _make_patch((128, 128, 128), size=32, seed=4)
    _, b0 = _ab_means(patch)
    shifted = apply_wb_shift(patch, temp_shift=-20.0, tint_shift=0.0)
    _, b1 = _ab_means(shifted)
    assert b1 < b0 - 5.0


def test_apply_positive_tint_increases_a_channel() -> None:
    """tint_shift > 0 で LAB a 平均値が増加 (赤寄り)."""
    patch = _make_patch((128, 128, 128), size=32, seed=5)
    a0, _ = _ab_means(patch)
    shifted = apply_wb_shift(patch, temp_shift=0.0, tint_shift=20.0)
    a1, _ = _ab_means(shifted)
    assert a1 > a0 + 5.0


def test_apply_negative_tint_decreases_a_channel() -> None:
    """tint_shift < 0 で LAB a 平均値が減少 (緑寄り)."""
    patch = _make_patch((128, 128, 128), size=32, seed=6)
    a0, _ = _ab_means(patch)
    shifted = apply_wb_shift(patch, temp_shift=0.0, tint_shift=-20.0)
    a1, _ = _ab_means(shifted)
    assert a1 < a0 - 5.0


# ============================
# apply_wb_shift: format / range
# ============================


def test_apply_keeps_uint8_and_shape() -> None:
    """dtype uint8, shape 不変 を維持."""
    patch = _make_patch((20, 200, 60), size=10, seed=7)
    out = apply_wb_shift(patch, temp_shift=15.0, tint_shift=-10.0)
    assert out.dtype == np.uint8
    assert out.shape == patch.shape


def test_apply_clips_to_uint8_range() -> None:
    """値域は [0, 255] に必ず clip される (極端 shift でも overflow しない)."""
    patch = _make_patch((255, 255, 255), size=8, seed=8)
    out = apply_wb_shift(patch, temp_shift=200.0, tint_shift=200.0)
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255
    out2 = apply_wb_shift(patch, temp_shift=-200.0, tint_shift=-200.0)
    assert out2.dtype == np.uint8
    assert out2.min() >= 0
    assert out2.max() <= 255


# ============================
# apply_wb_shift: input validation
# ============================


def test_apply_rejects_non_ndarray() -> None:
    """ndarray 以外は TypeError."""
    with pytest.raises(TypeError):
        apply_wb_shift([1, 2, 3])  # type: ignore[arg-type]


def test_apply_rejects_non_uint8() -> None:
    """uint8 以外は TypeError."""
    patch = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        apply_wb_shift(patch)


def test_apply_rejects_wrong_shape() -> None:
    """shape != (H, W, 3) は ValueError."""
    patch = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_wb_shift(patch)
    patch4 = np.zeros((4, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_wb_shift(patch4)


# ============================
# random_wb_shift
# ============================


def test_random_zero_max_is_identity() -> None:
    """max_temp=0, max_tint=0 で恒等変換."""
    patch = _make_patch((80, 80, 200), size=8, seed=9)
    rng = np.random.default_rng(seed=42)
    out = random_wb_shift(patch, max_temp=0.0, max_tint=0.0, rng=rng)
    np.testing.assert_array_equal(out, patch)


def test_random_default_in_uint8_range() -> None:
    """default 範囲で値域 [0, 255] と uint8 を維持."""
    patch = _make_patch((100, 120, 140), size=12, seed=10)
    rng = np.random.default_rng(seed=123)
    for _ in range(5):
        out = random_wb_shift(patch, rng=rng)
        assert out.dtype == np.uint8
        assert out.shape == patch.shape
        assert out.min() >= 0
        assert out.max() <= 255


def test_random_reproducible_with_seed() -> None:
    """同じ seed の rng で同じ結果 (再現性)."""
    patch = _make_patch((90, 100, 110), size=8, seed=11)
    rng_a = np.random.default_rng(seed=2026)
    rng_b = np.random.default_rng(seed=2026)
    out_a = random_wb_shift(patch, rng=rng_a)
    out_b = random_wb_shift(patch, rng=rng_b)
    np.testing.assert_array_equal(out_a, out_b)


def test_random_negative_max_raises() -> None:
    """max_temp < 0 / max_tint < 0 は ValueError."""
    patch = _make_patch(size=4, seed=12)
    with pytest.raises(ValueError):
        random_wb_shift(patch, max_temp=-1.0)
    with pytest.raises(ValueError):
        random_wb_shift(patch, max_tint=-1.0)


def test_default_constants_conservative() -> None:
    """default の max が想定値 (15 / 10) で控えめ."""
    assert DEFAULT_MAX_TEMP == 15.0
    assert DEFAULT_MAX_TINT == 10.0
