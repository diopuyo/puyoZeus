"""
calibration.py のテスト

合成フレームから BoardRegion と HsvRange を正しく抽出できることを検証する。
実対戦フレーム不要でロジック全体をテストする。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_RED,
    COLOR_YELLOW,
    Board,
)
from src.calibration import (
    CalibratedConfig,
    CalibrationAnnotation,
    CalibrationHelper,
)
from src.image_reader import (
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
    BoardRegion,
    HsvRange,
)

from tests.fixtures import make_synthetic_frame, sample_all_colors_board


# ============================
# CalibrationHelper - region_from_corners
# ============================


class TestRegionFromCorners:
    def test_valid_corners(self):
        h = CalibrationHelper()
        r = h.region_from_corners((10, 20), (100, 120))
        assert r.x == 10
        assert r.y == 20
        assert r.width == 90
        assert r.height == 100

    def test_invalid_corners_raises(self):
        h = CalibrationHelper()
        with pytest.raises(ValueError, match="逆転"):
            h.region_from_corners((100, 100), (10, 10))


# ============================
# CalibrationHelper - sample_cell_hsv
# ============================


class TestSampleCellHsv:
    def test_sample_from_synthetic_frame(self):
        board = sample_all_colors_board()
        frame = make_synthetic_frame(board_1p=board)
        helper = CalibrationHelper()
        # row=12, col=0 に赤を配置しているはず
        h, s, v = helper.sample_cell_hsv(frame, DEFAULT_P1_REGION, 12, 0)
        # 赤の H は 0 付近 or 180 付近
        assert h <= 15 or h >= 165
        assert s > 100
        assert v > 100

    def test_sample_empty_cell(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        helper = CalibrationHelper()
        h, s, v = helper.sample_cell_hsv(frame, DEFAULT_P1_REGION, 0, 0)
        assert v == 0

    def test_sample_respects_frame_bounds(self):
        """フレーム外に近いセル座標でもクラッシュしない。"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        helper = CalibrationHelper()
        # 小さいフレームで大きな region を指定 → クリップされる
        region = BoardRegion(x=10, y=10, width=80, height=80)
        helper.sample_cell_hsv(frame, region, 12, 5)


# ============================
# CalibrationHelper - hsv_range_from_samples
# ============================


class TestHsvRangeFromSamples:
    def test_single_sample_creates_range(self):
        board = Board.from_list(
            [[COLOR_BLUE] * BOARD_COLS if r == 12 else [COLOR_EMPTY] * BOARD_COLS
             for r in range(BOARD_ROWS)]
        )
        frame = make_synthetic_frame(board_1p=board)
        helper = CalibrationHelper()
        ranges = helper.hsv_range_from_samples(
            frame, DEFAULT_P1_REGION, [(12, 0), (12, 1), (12, 2)],
        )
        assert len(ranges) == 1
        assert isinstance(ranges[0], HsvRange)
        # 青の H=115 付近
        assert ranges[0].h_min <= 115 <= ranges[0].h_max

    def test_empty_positions_raises(self):
        helper = CalibrationHelper()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            helper.hsv_range_from_samples(frame, DEFAULT_P1_REGION, [])

    def test_range_covers_all_samples(self):
        """複数色位置のサンプルから生成した Range はすべてをカバーする。"""
        board = sample_all_colors_board()
        frame = make_synthetic_frame(board_1p=board)
        helper = CalibrationHelper()
        # 青の位置のみ
        ranges = helper.hsv_range_from_samples(
            frame, DEFAULT_P1_REGION, [(12, 1)],
        )
        # 青セルの HSV を含むこと
        h, s, v = helper.sample_cell_hsv(frame, DEFAULT_P1_REGION, 12, 1)
        r = ranges[0]
        assert r.h_min <= h <= r.h_max
        assert r.s_min <= s <= r.s_max
        assert r.v_min <= v <= r.v_max


# ============================
# CalibrationHelper - calibrate_from_reference
# ============================


class TestCalibrateFromReference:
    def test_roundtrip_with_synthetic_frame(self):
        """
        合成フレーム→キャリブレーション→ImageReader 再構築
        で元の Board を読み戻せることを確認。
        """
        board = sample_all_colors_board()
        frame = make_synthetic_frame(board_1p=board)

        # 合成フレームは DEFAULT_P1_REGION に配置済み
        ann = CalibrationAnnotation(
            p1_top_left=(
                DEFAULT_P1_REGION.x,
                DEFAULT_P1_REGION.y,
            ),
            p1_bottom_right=(
                DEFAULT_P1_REGION.x + DEFAULT_P1_REGION.width,
                DEFAULT_P1_REGION.y + DEFAULT_P1_REGION.height,
            ),
            p2_top_left=(
                DEFAULT_P2_REGION.x,
                DEFAULT_P2_REGION.y,
            ),
            p2_bottom_right=(
                DEFAULT_P2_REGION.x + DEFAULT_P2_REGION.width,
                DEFAULT_P2_REGION.y + DEFAULT_P2_REGION.height,
            ),
            color_samples={
                COLOR_RED:    [(12, 0)],
                COLOR_BLUE:   [(12, 1)],
                COLOR_GREEN:  [(12, 2)],
                COLOR_YELLOW: [(12, 3)],
                COLOR_OJAMA:  [(12, 5)],
            },
        )
        helper = CalibrationHelper()
        config = helper.calibrate_from_reference(frame, ann)

        assert isinstance(config, CalibratedConfig)
        assert COLOR_RED in config.color_ranges

        # 生成した config で ImageReader を再構築 → 盤面が読める
        reader = config.build_reader()
        read = reader.read_board(frame, config.p1_region)
        # 紫を省いているので紫以外をチェック
        for col, color in [(0, COLOR_RED), (1, COLOR_BLUE),
                           (2, COLOR_GREEN), (3, COLOR_YELLOW),
                           (5, COLOR_OJAMA)]:
            assert read.get(12, col) == color


# ============================
# CalibrationAnnotation - JSON
# ============================


class TestAnnotationJson:
    def test_from_json(self, tmp_path):
        path = tmp_path / "ann.json"
        path.write_text(json.dumps({
            "p1_corners": {"top_left": [10, 20], "bottom_right": [100, 200]},
            "p2_corners": {"top_left": [200, 20], "bottom_right": [300, 200]},
            "color_samples": {"1": [[12, 0], [11, 0]]},
        }), encoding="utf-8")
        ann = CalibrationAnnotation.from_json(path)
        assert ann.p1_top_left == (10, 20)
        assert ann.p1_bottom_right == (100, 200)
        assert ann.color_samples == {1: [(12, 0), (11, 0)]}


# ============================
# CalibratedConfig - save / load
# ============================


class TestConfigPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        config = CalibratedConfig(
            p1_region=BoardRegion(x=10, y=10, width=100, height=200),
            p2_region=BoardRegion(x=200, y=10, width=100, height=200),
            color_ranges={
                COLOR_RED: [HsvRange(h_min=0, h_max=10)],
            },
        )
        path = tmp_path / "calib.json"
        config.save(path)

        loaded = CalibratedConfig.load(path)
        assert loaded.p1_region.x == 10
        assert loaded.color_ranges[COLOR_RED][0].h_min == 0

    def test_build_reader_uses_config(self, tmp_path):
        board = sample_all_colors_board()
        frame = make_synthetic_frame(board_1p=board)
        config = CalibratedConfig(
            p1_region=DEFAULT_P1_REGION,
            p2_region=DEFAULT_P2_REGION,
            color_ranges={},  # 空=デフォルトにフォールバック
        )
        reader = config.build_reader()
        read = reader.read_board(frame, config.p1_region)
        assert read.get(12, 0) == COLOR_RED
