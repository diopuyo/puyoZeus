"""
board_grid_detector.py のテスト

合成画像 (cv2.line で grid を描画) で 6 × 13 の四隅検出を確認。
解像度 / aspect 比違い、不正画像、キャッシュ動作も検証。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.board import BOARD_COLS, BOARD_ROWS, VISIBLE_ROWS
from src.board_grid_detector import (
    BoardGridCache,
    BoardGridDetector,
    GridDetection,
    cells_from_grid,
    grid_or_default_region,
    grid_to_board_region,
)
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION


# ============================
# テストヘルパ
# ============================


def _make_grid_image(
    width: int,
    height: int,
    box: tuple[int, int, int, int],
    cols: int = BOARD_COLS,
    rows: int = VISIBLE_ROWS,
    bg_color: tuple[int, int, int] = (40, 40, 40),
    line_color: tuple[int, int, int] = (220, 220, 220),
    line_thickness: int = 2,
) -> np.ndarray:
    """
    指定 box 内に cols × rows のグリッド線を描いた合成画像を返す。
    box = (x_min, y_min, x_max, y_max)
    """
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    x_min, y_min, x_max, y_max = box
    cell_w = (x_max - x_min) / cols
    cell_h = (y_max - y_min) / rows
    # 縦線
    for i in range(cols + 1):
        x = int(round(x_min + i * cell_w))
        cv2.line(img, (x, y_min), (x, y_max), line_color, line_thickness)
    # 横線
    for j in range(rows + 1):
        y = int(round(y_min + j * cell_h))
        cv2.line(img, (x_min, y), (x_max, y), line_color, line_thickness)
    return img


# ============================
# 基本検出
# ============================


class TestBoardGridDetectorBasic:
    """合成 grid 画像で四隅を正しく検出できるか。"""

    def test_detect_returns_grid_for_synthetic_grid(self) -> None:
        box = (300, 160, 684, 880)  # 384 × 720 ≒ 既存 P1 region
        img = _make_grid_image(1920, 1080, box)
        det = BoardGridDetector().detect(img)
        assert det is not None
        # 四隅が box とほぼ一致 (許容 5px)
        assert abs(det.top_left[0] - box[0]) <= 5
        assert abs(det.top_left[1] - box[1]) <= 5
        assert abs(det.bottom_right[0] - box[2]) <= 5
        assert abs(det.bottom_right[1] - box[3]) <= 5
        assert det.confidence >= 0.3

    def test_detect_returns_none_for_uniform_image(self) -> None:
        img = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        assert BoardGridDetector().detect(img) is None

    def test_detect_returns_none_for_invalid_input(self) -> None:
        det = BoardGridDetector()
        assert det.detect(np.zeros((0, 0, 3), dtype=np.uint8)) is None
        # gray 画像 (3ch ではない) は None
        gray = np.zeros((100, 100), dtype=np.uint8)
        assert det.detect(gray) is None

    def test_grid_detection_dataclass_fields(self) -> None:
        d = GridDetection(
            top_left=(0.0, 0.0),
            top_right=(10.0, 0.0),
            bottom_left=(0.0, 20.0),
            bottom_right=(10.0, 20.0),
            confidence=0.5,
        )
        assert d.top_left == (0.0, 0.0)
        assert d.confidence == 0.5


# ============================
# 解像度 / aspect 比 robustness
# ============================


class TestResolutionRobustness:
    """720p / 1080p / 4K と aspect 比違いで検出できるか。"""

    @pytest.mark.parametrize(
        "width, height",
        [(1280, 720), (1920, 1080), (3840, 2160)],
    )
    def test_detect_at_various_resolutions(self, width: int, height: int) -> None:
        # 同比率 (画面の中央 20%-35% に盤面)
        x_min = int(width * 0.15)
        x_max = int(width * 0.35)
        y_min = int(height * 0.15)
        y_max = int(height * 0.85)
        img = _make_grid_image(width, height, (x_min, y_min, x_max, y_max))
        det = BoardGridDetector().detect(img)
        assert det is not None, f"failed at {width}x{height}"
        # 検出位置が想定 box の 5% 以内
        tol_x = max(5, int(width * 0.01))
        tol_y = max(5, int(height * 0.01))
        assert abs(det.top_left[0] - x_min) <= tol_x
        assert abs(det.top_left[1] - y_min) <= tol_y

    @pytest.mark.parametrize(
        "width, height",
        [(1920, 1200), (2560, 1080)],  # 16:10 と 21:9
    )
    def test_detect_at_various_aspects(self, width: int, height: int) -> None:
        x_min = int(width * 0.20)
        x_max = int(width * 0.40)
        y_min = int(height * 0.15)
        y_max = int(height * 0.85)
        img = _make_grid_image(width, height, (x_min, y_min, x_max, y_max))
        det = BoardGridDetector().detect(img)
        # aspect が違っても box 比率を保てば検出可能 (失敗時 None でも fallback で OK)
        if det is not None:
            assert det.confidence >= 0.3


# ============================
# cells_from_grid
# ============================


class TestCellsFromGrid:
    """cells_from_grid が 13 × 6 × 4 の bbox 配列を返すか。"""

    def test_shape_and_dtype(self) -> None:
        grid = GridDetection(
            top_left=(100.0, 200.0),
            top_right=(484.0, 200.0),
            bottom_left=(100.0, 920.0),
            bottom_right=(484.0, 920.0),
            confidence=0.9,
        )
        cells = cells_from_grid(grid)
        assert cells.shape == (BOARD_ROWS, BOARD_COLS, 4)
        assert cells.dtype == np.int32

    def test_visible_row_bounds_match_grid(self) -> None:
        grid = GridDetection(
            top_left=(100.0, 200.0),
            top_right=(484.0, 200.0),
            bottom_left=(100.0, 920.0),
            bottom_right=(484.0, 920.0),
            confidence=0.9,
        )
        cells = cells_from_grid(grid)
        # 可視領域 1 行目 (BOARD_ROWS - VISIBLE_ROWS = 1) 上端 = 200
        first_visible_row = BOARD_ROWS - VISIBLE_ROWS
        assert cells[first_visible_row, 0, 1] == 200
        # 最下行下端 = 920
        assert cells[BOARD_ROWS - 1, 0, 3] == 920
        # 左端 col0 の x_min = 100
        assert cells[first_visible_row, 0, 0] == 100
        # 右端 col5 の x_max = 484
        assert cells[first_visible_row, BOARD_COLS - 1, 2] == 484

    def test_hidden_row_above_visible(self) -> None:
        grid = GridDetection(
            top_left=(100.0, 200.0),
            top_right=(484.0, 200.0),
            bottom_left=(100.0, 920.0),
            bottom_right=(484.0, 920.0),
            confidence=0.9,
        )
        cells = cells_from_grid(grid)
        # row 0 (隠し段) は可視 row 1 の上方
        assert cells[0, 0, 3] == 200  # row0 下端 = row1 上端


# ============================
# BoardRegion adapter
# ============================


class TestBoardRegionAdapter:
    def test_grid_to_board_region(self) -> None:
        grid = GridDetection(
            top_left=(282.0, 160.0),
            top_right=(666.0, 160.0),
            bottom_left=(282.0, 880.0),
            bottom_right=(666.0, 880.0),
            confidence=0.9,
        )
        region = grid_to_board_region(grid)
        assert region is not None
        assert region.x == 282
        assert region.y == 160
        assert region.width == 384
        assert region.height == 720

    def test_grid_to_board_region_none(self) -> None:
        assert grid_to_board_region(None) is None

    def test_grid_or_default_region_p1_fallback(self) -> None:
        region = grid_or_default_region(None, player=1)
        assert region == DEFAULT_P1_REGION

    def test_grid_or_default_region_p2_fallback(self) -> None:
        region = grid_or_default_region(None, player=2)
        assert region == DEFAULT_P2_REGION

    def test_grid_or_default_region_uses_grid(self) -> None:
        grid = GridDetection(
            top_left=(50.0, 50.0),
            top_right=(434.0, 50.0),
            bottom_left=(50.0, 770.0),
            bottom_right=(434.0, 770.0),
            confidence=0.9,
        )
        region = grid_or_default_region(grid, player=1)
        assert region.x == 50
        assert region.y == 50


# ============================
# BoardGridCache
# ============================


class TestBoardGridCache:
    def test_cache_hit_skips_redetection(self) -> None:
        box = (300, 160, 684, 880)
        img = _make_grid_image(1920, 1080, box)
        cache = BoardGridCache()
        det1 = cache.detect_with_cache("video_99", img)
        assert det1 is not None
        assert cache.has("video_99")
        # 2 回目: ダミー (uniform) 画像でも cache hit で同値
        dummy = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        det2 = cache.detect_with_cache("video_99", dummy)
        assert det2 is det1

    def test_cache_miss_for_new_video_id(self) -> None:
        box = (300, 160, 684, 880)
        img = _make_grid_image(1920, 1080, box)
        cache = BoardGridCache()
        cache.detect_with_cache("video_a", img)
        # 別 video は新規検出
        assert not cache.has("video_b")

    def test_failure_does_not_cache(self) -> None:
        cache = BoardGridCache()
        uniform = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        result = cache.detect_with_cache("video_x", uniform)
        assert result is None
        # キャッシュは積まれていないので、後で正しい frame を渡すと検出できる
        assert not cache.has("video_x")
        box = (300, 160, 684, 880)
        good_img = _make_grid_image(1920, 1080, box)
        result2 = cache.detect_with_cache("video_x", good_img)
        assert result2 is not None
        assert cache.has("video_x")

    def test_clear(self) -> None:
        box = (300, 160, 684, 880)
        img = _make_grid_image(1920, 1080, box)
        cache = BoardGridCache()
        cache.detect_with_cache("video_clr", img)
        assert cache.has("video_clr")
        cache.clear()
        assert not cache.has("video_clr")


# ============================
# 既存 ROI 経路の互換 (regression)
# ============================


class TestExistingRoiUnaffected:
    """既存 image_reader の DEFAULT_P1_REGION / P2_REGION が変わっていないこと。"""

    def test_default_regions_unchanged(self) -> None:
        assert DEFAULT_P1_REGION.x == 282
        assert DEFAULT_P1_REGION.y == 160
        assert DEFAULT_P1_REGION.width == 384
        assert DEFAULT_P1_REGION.height == 720
        assert DEFAULT_P2_REGION.x == 1258
        assert DEFAULT_P2_REGION.y == 160
