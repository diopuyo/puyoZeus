"""tests/test_patch_fingerprint.py

案 d: CellPatchFingerprint / PatchBackgroundFingerprint の単体テスト (10-15 件)。
NCC 計算の正確性、閾値境界、save/load roundtrip、ImageReader 互換性を検証。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.background_fingerprint import (
    PATCH_NCC_EMPTY_THRESHOLD,
    PATCH_NCC_UNIFORM_FALLBACK,
    CellFingerprint,
    CellPatchFingerprint,
    PatchBackgroundFingerprint,
    _compute_ncc,
    capture_patch_pair_robust,
    is_empty_by_patch_fp,
    load_patch_fingerprint_pair,
    save_patch_fingerprint_pair,
)
from src.board import BOARD_COLS, VISIBLE_ROWS


# ===== ユーティリティ =====

def _make_patch(h: float, s: float, v: float, h_: int = 8, w_: int = 8) -> np.ndarray:
    """均一色パッチを作成。shape: (h_, w_, 3) dtype: float32"""
    p = np.full((h_, w_, 3), [h, s, v], dtype=np.float32)
    return p


def _make_noise_patch(seed: int = 42, h_: int = 8, w_: int = 8) -> np.ndarray:
    """ランダムノイズパッチ。"""
    rng = np.random.default_rng(seed)
    return rng.random((h_, w_, 3)).astype(np.float32) * 255.0


# ===== テスト =====

def test_cell_patch_ncc_identical_returns_one() -> None:
    """同一パッチを NCC すると 1.0 になる (ただし均一パッチは fallback)。"""
    p = _make_noise_patch(seed=1)
    a = CellPatchFingerprint(patch_hsv=p)
    b = CellPatchFingerprint(patch_hsv=p.copy())
    ncc = a.ncc_to(b)
    assert abs(ncc - 1.0) < 1e-9, f"同一パッチ NCC={ncc} != 1.0"


def test_cell_patch_ncc_different_returns_low() -> None:
    """全く異なるパッチ (ぷよ色 vs 背景) は NCC < 0.9 になる。"""
    # 緑ぷよ: H=60, S=200, V=200 付近
    puyo = _make_noise_patch(seed=99)
    # 黄緑背景: H=30, S=100, V=150 付近 (完全に別のノイズ)
    bg = _make_noise_patch(seed=7)
    a = CellPatchFingerprint(patch_hsv=puyo)
    b = CellPatchFingerprint(patch_hsv=bg)
    ncc = a.ncc_to(b)
    assert ncc < 0.9, f"異なるパッチ NCC={ncc} >= 0.9 (想定: 低い)"


def test_cell_patch_is_empty_threshold() -> None:
    """threshold 境界条件のテスト。"""
    p = _make_noise_patch(seed=5)
    a = CellPatchFingerprint(patch_hsv=p)
    b = CellPatchFingerprint(patch_hsv=p.copy())
    # 同一 → NCC=1.0 → is_empty=True (threshold=0.92 < 1.0)
    assert a.is_empty_by_ncc(b, threshold=PATCH_NCC_EMPTY_THRESHOLD) is True
    # threshold=1.01 → NCC=1.0 < 1.01 → is_empty=False
    assert a.is_empty_by_ncc(b, threshold=1.01) is False


def test_cell_patch_uniform_background() -> None:
    """均一パッチ (均一背景) は NCC_UNIFORM_FALLBACK を返し、空判定になる。"""
    # 均一パッチ: std=0 → _compute_ncc が FALLBACK 返却
    p_uniform = _make_patch(30.0, 100.0, 150.0)
    ncc = _compute_ncc(p_uniform, p_uniform)
    assert ncc == PATCH_NCC_UNIFORM_FALLBACK, f"均一パッチ NCC={ncc} != fallback"
    a = CellPatchFingerprint(patch_hsv=p_uniform)
    b = CellPatchFingerprint(patch_hsv=p_uniform.copy())
    assert is_empty_by_patch_fp(a, b) is True


def test_cell_patch_puyo_on_yellow_bg() -> None:
    """黄緑背景に乗った黄ぷよは背景と区別できる (NCC < threshold → is_empty=False)。

    背景: 均一黄緑 (H=30, S=80, V=150)
    ぷよ: ランダムノイズ (= 実際の黄ぷよテクスチャを模擬)
    NCC が低くなることで is_empty=False が返る。
    """
    bg_patch = _make_patch(30.0, 80.0, 150.0)  # 均一背景
    # ぷよパッチ: 均一ではない (ノイズあり) → NCC が低下する
    puyo_patch = _make_noise_patch(seed=42)
    bg_fp = CellPatchFingerprint(patch_hsv=bg_patch)
    cur_fp = CellPatchFingerprint(patch_hsv=puyo_patch)
    # 均一背景 vs ノイズパッチ → NCC が低い → is_empty=False
    result = is_empty_by_patch_fp(cur_fp, bg_fp)
    assert result is False, "黄ぷよパッチが背景と同一判定されてはいけない"


def test_patch_bg_fp_capture_shape() -> None:
    """PatchBackgroundFingerprint.capture の出力 shape を確認。"""
    # ダミーフレーム (1920×1080 の一部、単色 BGR)
    frame = np.full((1080, 1920, 3), [100, 150, 200], dtype=np.uint8)
    region_x, region_y, region_w, region_h = 100, 50, 240, 480
    fp = PatchBackgroundFingerprint.capture(
        frame, region_x, region_y, region_w, region_h,
    )
    assert len(fp.patch_cells) == VISIBLE_ROWS
    assert len(fp.patch_cells[0]) == BOARD_COLS
    assert isinstance(fp.patch_cells[0][0], CellPatchFingerprint)
    assert fp.patch_cells[0][0].patch_hsv.ndim == 3
    assert fp.patch_cells[0][0].patch_hsv.shape[2] == 3
    assert fp.patch_cells[0][0].patch_hsv.dtype == np.float32


def test_patch_bg_fp_cell_at_compatibility() -> None:
    """PatchBackgroundFingerprint.cell_at が CellFingerprint を返す (後退互換)。"""
    frame = np.full((1080, 1920, 3), [80, 120, 200], dtype=np.uint8)
    fp = PatchBackgroundFingerprint.capture(frame, 100, 50, 240, 480)
    cell = fp.cell_at(0, 0)
    assert isinstance(cell, CellFingerprint), f"cell_at が CellFingerprint を返すべき: {type(cell)}"


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """npz 保存 → ロードで数値が一致する。"""
    frame = np.full((1080, 1920, 3), [60, 180, 200], dtype=np.uint8)
    fp1 = PatchBackgroundFingerprint.capture(frame, 100, 50, 240, 480)
    fp2 = PatchBackgroundFingerprint.capture(frame, 700, 50, 240, 480)
    save_path = tmp_path / "test_fp.npz"
    save_patch_fingerprint_pair(save_path, fp1, fp2)
    assert save_path.exists()
    loaded_fp1, loaded_fp2 = load_patch_fingerprint_pair(save_path)
    # fp1 の cell (0,0) patch が一致
    orig = fp1.patch_cells[0][0].patch_hsv
    loaded = loaded_fp1.patch_cells[0][0].patch_hsv
    np.testing.assert_allclose(orig, loaded, rtol=1e-5, atol=1e-5)
    # median cells も一致
    assert fp1.median_cells[0][0].h == loaded_fp1.median_cells[0][0].h


def test_capture_pair_patch_robust() -> None:
    """複数フレームの median 集約で shape が正しく返る。"""
    frames = [
        np.full((1080, 1920, 3), [50 + i * 5, 100, 200], dtype=np.uint8)
        for i in range(6)
    ]
    fp1, fp2 = capture_patch_pair_robust(
        frames,
        (100, 50, 240, 480),
        (700, 50, 240, 480),
    )
    assert isinstance(fp1, PatchBackgroundFingerprint)
    assert isinstance(fp2, PatchBackgroundFingerprint)
    assert len(fp1.patch_cells) == VISIBLE_ROWS
    assert len(fp1.patch_cells[0]) == BOARD_COLS


def test_image_reader_with_patch_fp() -> None:
    """ImageReader に PatchBackgroundFingerprint を渡しても TypeError にならない。"""
    from src.image_reader import ImageReader
    reader = ImageReader()
    frame = np.full((1080, 1920, 3), [50, 100, 150], dtype=np.uint8)
    fp1 = PatchBackgroundFingerprint.capture(frame, 100, 50, 240, 480)
    fp2 = PatchBackgroundFingerprint.capture(frame, 700, 50, 240, 480)
    # TypeError が起きなければ OK
    try:
        reader.set_background_fingerprints(fp1, fp2)
    except TypeError as e:
        pytest.fail(f"TypeError が発生: {e}")


def test_compute_ncc_both_uniform() -> None:
    """両方が均一パッチの場合は FALLBACK (std=0 → NaN 回避) を確認。"""
    # 同じ均一値 → ravel 後 std=0 → PATCH_NCC_UNIFORM_FALLBACK
    uniform_a = _make_patch(30.0, 100.0, 150.0)
    uniform_b = _make_patch(30.0, 100.0, 150.0)
    ncc = _compute_ncc(uniform_a, uniform_b)
    assert ncc == PATCH_NCC_UNIFORM_FALLBACK, f"両均一 NCC={ncc} != fallback"


def test_patch_cell_at_patch_out_of_range() -> None:
    """範囲外インデックスでもクラッシュせず zeros パッチが返る。"""
    frame = np.full((1080, 1920, 3), [0, 0, 0], dtype=np.uint8)
    fp = PatchBackgroundFingerprint.capture(frame, 100, 50, 240, 480)
    cell = fp.cell_at_patch(-1, -1)
    assert isinstance(cell, CellPatchFingerprint)
    assert cell.patch_hsv.shape[2] == 3


def test_patch_ncc_shape_mismatch() -> None:
    """shape が異なるパッチ同士の NCC はリサイズ後に計算される (クラッシュしない)。"""
    p_large = _make_noise_patch(seed=10, h_=16, w_=16)
    p_small = _make_noise_patch(seed=10, h_=8, w_=8)
    a = CellPatchFingerprint(patch_hsv=p_large)
    b = CellPatchFingerprint(patch_hsv=p_small)
    ncc = a.ncc_to(b)
    # クラッシュしないことを確認 (-1〜1 の範囲)
    assert -1.0 <= ncc <= 1.0 + 1e-9
