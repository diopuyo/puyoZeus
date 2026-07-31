"""背景照合 NCC の直接ピアソン化が np.corrcoef と一致することを検証する。

2026-07-30 の高速化。np.corrcoef は 2xN 行列 → 2x2 共分散行列 → 対角 sqrt で
2 回除算という重い道を通るが、必要なのは 1 要素だけなので
「平均引き + 内積 2 回」で直接求める。

実測 (scripts/_diag_bg_ncc_cost_split_2026-07-30.py):
    24x24x3 で 35.97us → 5.09us = 7.1倍速、差は 4.16e-17。
    120セルまとめて einsum にする案は 2.0倍・絶対値 0.27ms しか縮まず不採用。

**bit-identical ではない** (演算順序が違う) ので、許容差内の一致と
「閾値 PATCH_NCC_EMPTY_THRESHOLD での判定が変わらないこと」を検証する。
"""

from __future__ import annotations

import numpy as np
import pytest

import src.background_fingerprint as bgfp
from src.background_fingerprint import (
    PATCH_NCC_EMPTY_THRESHOLD,
    PATCH_NCC_UNIFORM_FALLBACK,
    CellPatchFingerprint,
    _compute_ncc,
)

# corrcoef と直接ピアソンの許容差。実測は 4.16e-17 程度。
NCC_ABS_TOL: float = 1e-12
PATCH_SHAPES: tuple[tuple[int, int], ...] = ((8, 8), (16, 16), (24, 24), (32, 32))


def _make_patch(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """HSV パッチ相当の float32 配列 (H=0-180 / S,V=0-255)。"""
    patch = np.empty((h, w, 3), dtype=np.float32)
    patch[:, :, 0] = rng.integers(0, 181, size=(h, w))
    patch[:, :, 1] = rng.integers(0, 256, size=(h, w))
    patch[:, :, 2] = rng.integers(0, 256, size=(h, w))
    return patch


def _ncc_with_flag(a: np.ndarray, b: np.ndarray, direct: bool) -> float:
    """フラグを切り替えて _compute_ncc を呼ぶ (必ず元に戻す)。"""
    saved = bgfp.ENABLE_DIRECT_PEARSON_NCC
    bgfp.ENABLE_DIRECT_PEARSON_NCC = direct
    try:
        return _compute_ncc(a, b)
    finally:
        bgfp.ENABLE_DIRECT_PEARSON_NCC = saved


@pytest.mark.parametrize("h,w", PATCH_SHAPES)
def test_direct_pearson_matches_corrcoef(h: int, w: int) -> None:
    """直接ピアソンが np.corrcoef と許容差内で一致する。"""
    rng = np.random.default_rng(h * 100 + w)
    for _ in range(40):
        a = _make_patch(rng, h, w)
        b = _make_patch(rng, h, w)
        got = _ncc_with_flag(a, b, direct=True)
        expected = _ncc_with_flag(a, b, direct=False)
        assert got == pytest.approx(expected, abs=NCC_ABS_TOL)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_empty_decision_unchanged(seed: int) -> None:
    """閾値 PATCH_NCC_EMPTY_THRESHOLD による空判定が変わらない。

    NCC 値そのものより、この二値判定が変わらないことが本質。
    背景に近いパッチ (高相関) から遠いパッチまで振って確認する。
    """
    rng = np.random.default_rng(seed)
    for _ in range(60):
        bg = _make_patch(rng, 24, 24)
        # 背景にノイズを乗せて相関を段階的に落とす → 閾値をまたぐケースを作る
        for noise in (0.0, 5.0, 20.0, 50.0, 120.0):
            cur = np.clip(
                bg + rng.normal(0.0, noise, size=bg.shape), 0, 255,
            ).astype(np.float32)
            d = _ncc_with_flag(cur, bg, direct=True) >= PATCH_NCC_EMPTY_THRESHOLD
            c = _ncc_with_flag(cur, bg, direct=False) >= PATCH_NCC_EMPTY_THRESHOLD
            assert d == c


def test_uniform_current_patch_returns_fallback() -> None:
    """現フレームパッチが均一なら FALLBACK を返す (両経路で同じ)。"""
    rng = np.random.default_rng(9)
    bg = _make_patch(rng, 16, 16)
    cur = np.full((16, 16, 3), 100.0, dtype=np.float32)
    assert _ncc_with_flag(cur, bg, direct=True) == PATCH_NCC_UNIFORM_FALLBACK
    assert _ncc_with_flag(cur, bg, direct=False) == PATCH_NCC_UNIFORM_FALLBACK


def test_zero_bg_patch_still_returns_zero() -> None:
    """採取失敗ゼロパッチの多層防御が両経路で維持される。"""
    rng = np.random.default_rng(10)
    cur = _make_patch(rng, 16, 16)
    zero_bg = np.zeros((16, 16, 3), dtype=np.float32)
    assert _ncc_with_flag(cur, zero_bg, direct=True) == 0.0
    assert _ncc_with_flag(cur, zero_bg, direct=False) == 0.0


def test_identical_patch_gives_one() -> None:
    """完全一致パッチの NCC が 1.0 になる (正しさの sanity check)。"""
    rng = np.random.default_rng(11)
    p = _make_patch(rng, 24, 24)
    assert _ncc_with_flag(p, p.copy(), direct=True) == pytest.approx(1.0, abs=1e-12)


def test_ncc_to_uses_cached_bg_and_matches_direct_call() -> None:
    """ncc_to (キャッシュ経路) が _compute_ncc と一致する。

    b 側の平均引き・自己内積をキャッシュしているので、
    キャッシュ生成のロジック誤りをここで検出する。
    """
    rng = np.random.default_rng(12)
    for _ in range(30):
        bg_arr = _make_patch(rng, 24, 24)
        cur_arr = _make_patch(rng, 24, 24)
        cur = CellPatchFingerprint(patch_hsv=cur_arr)
        bg = CellPatchFingerprint(patch_hsv=bg_arr.copy())
        # 2 回呼んで (1 回目でキャッシュ生成、2 回目で再利用) 両方一致
        first = cur.ncc_to(bg)
        second = cur.ncc_to(bg)
        expected = _ncc_with_flag(cur_arr, bg_arr, direct=True)
        assert first == second
        assert first == pytest.approx(expected, abs=NCC_ABS_TOL)


def test_flag_default_is_on() -> None:
    """既定は高速経路 (実測で差 4.16e-17 と確認済みのため)。"""
    assert bgfp.ENABLE_DIRECT_PEARSON_NCC is True
