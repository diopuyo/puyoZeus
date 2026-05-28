"""StaticBoardMask 単体テスト (T4)。

合成フレームを使って cell_diff_scores / classify_background_cells /
capture_static_mask / save/load round-trip を検証する。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.background_fingerprint import (
    STATIC_BG_DIFF_THRESHOLD,
    STATIC_BG_MIN_FRAMES,
    StaticBoardMask,
    capture_static_mask,
    capture_static_mask_pair,
    load_static_mask_pair,
    save_static_mask_pair,
)
from src.board import BOARD_COLS, VISIBLE_ROWS


# テスト用定数
_REGION = (0, 0, 384, 720)  # (x, y, w, h)
_H, _W = 720, 384


def _make_frame(color: tuple[int, int, int] = (50, 50, 50)) -> np.ndarray:
    """指定 BGR 単色の合成フレームを生成する。"""
    frame = np.zeros((_H, _W, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def _make_mask(bg_color: tuple[int, int, int] = (50, 50, 50)) -> StaticBoardMask:
    """単色背景 StaticBoardMask を生成する。"""
    frames = [_make_frame(bg_color) for _ in range(STATIC_BG_MIN_FRAMES)]
    return capture_static_mask(frames, _REGION)


class TestStaticBoardMaskConstruction:
    """StaticBoardMask 生成テスト。"""

    def test_capture_returns_instance(self) -> None:
        """capture_static_mask が StaticBoardMask を返す。"""
        mask = _make_mask()
        assert isinstance(mask, StaticBoardMask)

    def test_bg_roi_shape(self) -> None:
        """bg_roi の shape が (H, W, 3) である。"""
        mask = _make_mask()
        assert mask.bg_roi.ndim == 3
        assert mask.bg_roi.shape[2] == 3

    def test_region_fields(self) -> None:
        """region フィールドが正しく格納される。"""
        mask = _make_mask()
        x, y, w, h = _REGION
        assert mask.region_x == x
        assert mask.region_y == y
        assert mask.region_w == w
        assert mask.region_h == h

    def test_insufficient_frames_raises(self) -> None:
        """フレーム不足で ValueError が発生する。"""
        frames = [_make_frame() for _ in range(STATIC_BG_MIN_FRAMES - 1)]
        with pytest.raises(ValueError):
            capture_static_mask(frames, _REGION)


class TestCellDiffScores:
    """cell_diff_scores テスト。"""

    def test_same_frame_scores_near_zero(self) -> None:
        """同一フレームなら diff はほぼ 0。"""
        mask = _make_mask(bg_color=(80, 80, 80))
        same_frame = _make_frame(color=(80, 80, 80))
        scores = mask.cell_diff_scores(same_frame)
        assert scores.shape == (VISIBLE_ROWS, BOARD_COLS)
        assert float(np.max(scores)) < 5.0  # 同色なら近い

    def test_different_frame_scores_large(self) -> None:
        """全く異なる色なら diff は大きい。"""
        mask = _make_mask(bg_color=(0, 0, 0))
        bright_frame = _make_frame(color=(200, 200, 200))
        scores = mask.cell_diff_scores(bright_frame)
        assert float(np.min(scores)) > 50.0

    def test_scores_shape(self) -> None:
        """スコアの shape は (VISIBLE_ROWS, BOARD_COLS)。"""
        mask = _make_mask()
        scores = mask.cell_diff_scores(_make_frame())
        assert scores.shape == (VISIBLE_ROWS, BOARD_COLS)


class TestClassifyBackgroundCells:
    """classify_background_cells テスト。"""

    def test_background_returns_all_true(self) -> None:
        """背景と同じフレームは全 cell True。"""
        mask = _make_mask(bg_color=(50, 50, 50))
        same_frame = _make_frame(color=(50, 50, 50))
        result = mask.classify_background_cells(same_frame)
        assert result.dtype == bool
        assert result.all()

    def test_different_color_returns_all_false(self) -> None:
        """全く異なる色は全 cell False。"""
        mask = _make_mask(bg_color=(0, 0, 0))
        bright_frame = _make_frame(color=(255, 0, 0))
        result = mask.classify_background_cells(bright_frame)
        assert not result.any()

    def test_custom_threshold(self) -> None:
        """threshold=0.0 なら誰も True にならない。"""
        mask = _make_mask(bg_color=(50, 50, 50))
        same_frame = _make_frame(color=(50, 50, 50))
        result = mask.classify_background_cells(same_frame, threshold=0.0)
        assert not result.any()


class TestCapturePair:
    """capture_static_mask_pair テスト。"""

    def test_returns_two_masks(self) -> None:
        """2 つの StaticBoardMask を返す。"""
        frames = [_make_frame() for _ in range(STATIC_BG_MIN_FRAMES)]
        m1, m2 = capture_static_mask_pair(frames, _REGION, _REGION)
        assert isinstance(m1, StaticBoardMask)
        assert isinstance(m2, StaticBoardMask)


class TestSaveLoadRoundTrip:
    """save / load round-trip テスト。"""

    def test_round_trip(self, tmp_path: Path) -> None:
        """保存 → ロードで bg_roi と region が一致する。"""
        mask1 = _make_mask(bg_color=(30, 30, 30))
        mask2 = _make_mask(bg_color=(100, 200, 100))
        npz_path = tmp_path / "test_mask.npz"
        save_static_mask_pair(npz_path, mask1, mask2)
        assert npz_path.exists()
        loaded1, loaded2 = load_static_mask_pair(npz_path)
        np.testing.assert_array_equal(loaded1.bg_roi, mask1.bg_roi)
        np.testing.assert_array_equal(loaded2.bg_roi, mask2.bg_roi)
        assert loaded1.region_x == mask1.region_x
        assert loaded2.region_w == mask2.region_w

    def test_parent_dir_created(self, tmp_path: Path) -> None:
        """親ディレクトリが存在しなくても自動作成される。"""
        mask = _make_mask()
        npz_path = tmp_path / "subdir" / "nested" / "mask.npz"
        save_static_mask_pair(npz_path, mask, mask)
        assert npz_path.exists()
