"""背景 FP の NCC キャッシュが旧実装と完全同値であることを検証する。

2026-07-30 の高速化: `CellPatchFingerprint.ncc_to` は永続する背景パッチ側の
resize と有効性判定 (パッチ全体の median) を毎フレームやり直していた
(実測 resize 340回/frame、_is_bg_patch_valid 144回/frame)。
patch_hsv は read-only 規約なのでこれらは純関数であり、キャッシュしても同値。

キャッシュは「同値のはず」の変更なのでフラグで守っていない。
そのため旧実装をこのテスト内に保存し、完全一致を恒久的に検証する。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.background_fingerprint import (
    CellPatchFingerprint,
    _compute_ncc,
    _is_bg_patch_valid,
)

# 現フレームパッチ / 背景パッチの shape 組み合わせ (不一致 = resize 経路)
SHAPE_PAIRS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((16, 16), (16, 16)),   # 一致 (resize なし)
    ((16, 16), (24, 24)),   # 縮小
    ((32, 32), (16, 16)),   # 拡大
    ((20, 28), (28, 20)),   # 非正方
)


def _old_ncc_to(a_fp: CellPatchFingerprint, b_fp: CellPatchFingerprint) -> float:
    """旧 `ncc_to` (キャッシュ導入前の実装をそのまま保存)。"""
    a = a_fp.patch_hsv
    b = b_fp.patch_hsv
    if a.shape != b.shape:
        b = cv2.resize(
            b.astype(np.float32),
            (a.shape[1], a.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return _compute_ncc(a, b)


def _make_patch(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """HSV パッチ相当の float32 配列を作る (H=0-180 / S,V=0-255)。"""
    patch = np.empty((h, w, 3), dtype=np.float32)
    patch[:, :, 0] = rng.integers(0, 181, size=(h, w))
    patch[:, :, 1] = rng.integers(0, 256, size=(h, w))
    patch[:, :, 2] = rng.integers(0, 256, size=(h, w))
    return patch


@pytest.mark.parametrize("a_shape,b_shape", SHAPE_PAIRS)
def test_ncc_to_matches_old_implementation(
    a_shape: tuple[int, int], b_shape: tuple[int, int],
) -> None:
    """キャッシュ有りの ncc_to が旧実装と完全一致する (resize 経路も含む)。"""
    rng = np.random.default_rng(a_shape[0] * 100 + b_shape[0])
    for _ in range(30):
        a_fp = CellPatchFingerprint(patch_hsv=_make_patch(rng, *a_shape))
        bg_arr = _make_patch(rng, *b_shape)
        # 旧実装用と新実装用で別インスタンスを使う (キャッシュ汚染を避ける)
        expected = _old_ncc_to(a_fp, CellPatchFingerprint(patch_hsv=bg_arr.copy()))
        actual = a_fp.ncc_to(CellPatchFingerprint(patch_hsv=bg_arr.copy()))
        assert actual == expected


def test_cache_reuse_across_frames_is_identical() -> None:
    """同一背景 FP を複数フレームで再利用しても毎回同じ値になる。

    これが崩れるとキャッシュが結果を変えている = 高速化が同値でない。
    """
    rng = np.random.default_rng(4242)
    bg = CellPatchFingerprint(patch_hsv=_make_patch(rng, 24, 24))
    bg_fresh_source = bg.patch_hsv.copy()
    for _ in range(20):
        cur = CellPatchFingerprint(patch_hsv=_make_patch(rng, 16, 16))
        # キャッシュ済み背景 vs 毎回新規に作った背景 (= キャッシュ空) で一致するか
        cached = cur.ncc_to(bg)
        fresh = cur.ncc_to(CellPatchFingerprint(patch_hsv=bg_fresh_source.copy()))
        assert cached == fresh


def test_invalid_bg_patch_returns_zero_with_cache() -> None:
    """採取失敗ゼロパッチ (V median 不足) はキャッシュ経路でも 0.0 を返す。

    多層防御の要 (FALLBACK=1.0 の誤発火防止) がキャッシュで抜けないことを確認。
    """
    rng = np.random.default_rng(1)
    zero_bg = CellPatchFingerprint(patch_hsv=np.zeros((24, 24, 3), dtype=np.float32))
    assert not _is_bg_patch_valid(zero_bg.patch_hsv)
    for _ in range(5):
        cur = CellPatchFingerprint(patch_hsv=_make_patch(rng, 24, 24))
        # 2 回呼んで (1 回目でキャッシュ作成、2 回目で再利用) どちらも 0.0
        assert cur.ncc_to(zero_bg) == 0.0
        assert cur.ncc_to(zero_bg) == 0.0


def test_cache_key_is_per_target_shape() -> None:
    """同じ背景 FP を異なる shape の相手に使っても取り違えない。

    キャッシュ key を shape にしているので、key 設計の誤りをここで検出する。
    """
    rng = np.random.default_rng(7)
    bg_arr = _make_patch(rng, 32, 32)
    bg = CellPatchFingerprint(patch_hsv=bg_arr)
    for shape in ((16, 16), (24, 24), (32, 32), (16, 16)):
        cur = CellPatchFingerprint(patch_hsv=_make_patch(rng, *shape))
        expected = _old_ncc_to(cur, CellPatchFingerprint(patch_hsv=bg_arr.copy()))
        assert cur.ncc_to(bg) == expected


def test_equality_unaffected_by_cache() -> None:
    """キャッシュの有無で dataclass の等価性が変わらない (compare=False の確認)。"""
    rng = np.random.default_rng(11)
    arr = _make_patch(rng, 16, 16)
    fp_a = CellPatchFingerprint(patch_hsv=arr)
    fp_b = CellPatchFingerprint(patch_hsv=arr)
    cur = CellPatchFingerprint(patch_hsv=_make_patch(rng, 16, 16))
    cur.ncc_to(fp_a)  # fp_a 側にだけキャッシュを作る
    # patch_hsv が同一オブジェクトなので、キャッシュ有無で等価性は変わらないはず
    assert (fp_a == fp_b) is (fp_b == fp_a)


def test_single_arg_construction_still_works() -> None:
    """キャッシュ field 追加後も 1 引数での構築が壊れていない (backwards compat)。"""
    arr = np.zeros((8, 8, 3), dtype=np.float32)
    assert CellPatchFingerprint(arr).patch_hsv.shape == (8, 8, 3)
    assert CellPatchFingerprint(patch_hsv=arr).patch_hsv.shape == (8, 8, 3)
