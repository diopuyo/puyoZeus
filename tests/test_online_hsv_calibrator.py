"""OnlineHsvCalibrator のテスト (Phase Z-3I)。"""
from __future__ import annotations

import cv2
import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, Board, COLOR_EMPTY, COLOR_GREEN,
    COLOR_RED, HIDDEN_ROWS,
)
from src.image_reader import BoardRegion
from src.online_hsv_calibrator import (
    MIN_SAMPLES, OnlineHsvCalibrator, TRAINABLE_COLORS,
)

REGION = BoardRegion(x=0, y=0, width=384, height=720)


def _make_frame(saturation: int, value: int, hue: int = 60) -> np.ndarray:
    hsv = np.zeros((720, 384, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = saturation
    hsv[:, :, 2] = value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _set_board(board: Board, vrow: int, col: int, color: int) -> None:
    board.set(vrow + HIDDEN_ROWS, col, color)


def _make_full_board(color: int) -> Board:
    """全 cell を指定色で埋めた Board。"""
    board = Board()
    for vrow in range(12):
        for col in range(BOARD_COLS):
            board.set(vrow + HIDDEN_ROWS, col, color)
    return board


def test_no_samples_when_low_cnn_conf() -> None:
    """CNN 確信度が低い cell は学習対象外。"""
    calib = OnlineHsvCalibrator()
    board = _make_full_board(COLOR_GREEN)
    frame = _make_frame(saturation=180, value=200, hue=60)
    # 全 cell の CNN conf=0.5 (HIGH_CONF=0.95 未満)
    cnn_proba = np.full((12, BOARD_COLS), 0.5, dtype=np.float32)
    hsv_color = np.full((12, BOARD_COLS), COLOR_GREEN, dtype=np.int32)
    calib.update(frame, REGION, board, cnn_proba, hsv_color)
    counts = calib.get_sample_counts()
    assert all(n == 0 for n in counts.values())


def test_no_samples_when_hsv_disagrees() -> None:
    """HSV 判定が CNN と異なる cell は学習対象外。"""
    calib = OnlineHsvCalibrator()
    board = _make_full_board(COLOR_GREEN)
    frame = _make_frame(saturation=180, value=200, hue=60)
    cnn_proba = np.full((12, BOARD_COLS), 0.99, dtype=np.float32)
    hsv_color = np.full((12, BOARD_COLS), COLOR_RED, dtype=np.int32)
    calib.update(frame, REGION, board, cnn_proba, hsv_color)
    counts = calib.get_sample_counts()
    assert counts[COLOR_GREEN] == 0


def test_samples_accumulated_when_reliable() -> None:
    """CNN/HSV 一致 + 高確信度の cell は色別に蓄積される。"""
    calib = OnlineHsvCalibrator()
    board = _make_full_board(COLOR_GREEN)
    frame = _make_frame(saturation=180, value=200, hue=60)
    cnn_proba = np.full((12, BOARD_COLS), 0.99, dtype=np.float32)
    hsv_color = np.full((12, BOARD_COLS), COLOR_GREEN, dtype=np.int32)
    calib.update(frame, REGION, board, cnn_proba, hsv_color)
    counts = calib.get_sample_counts()
    assert counts[COLOR_GREEN] == 12 * BOARD_COLS  # 全 cell


def test_skip_during_chain() -> None:
    """is_chain=True の frame は学習対象外 (連鎖中の HSV は不安定)。"""
    calib = OnlineHsvCalibrator()
    board = _make_full_board(COLOR_GREEN)
    frame = _make_frame(saturation=180, value=200, hue=60)
    cnn_proba = np.full((12, BOARD_COLS), 0.99, dtype=np.float32)
    hsv_color = np.full((12, BOARD_COLS), COLOR_GREEN, dtype=np.int32)
    calib.update(frame, REGION, board, cnn_proba, hsv_color, is_chain=True)
    counts = calib.get_sample_counts()
    assert counts[COLOR_GREEN] == 0


def test_is_ready_threshold() -> None:
    """サンプル数が MIN_SAMPLES 未満なら準備未完了。"""
    calib = OnlineHsvCalibrator(min_samples=100)
    board = _make_full_board(COLOR_GREEN)
    frame = _make_frame(saturation=180, value=200, hue=60)
    cnn_proba = np.full((12, BOARD_COLS), 0.99, dtype=np.float32)
    hsv_color = np.full((12, BOARD_COLS), COLOR_GREEN, dtype=np.int32)
    # 1 frame で 72 cells、まだ MIN_SAMPLES (100) 未満
    calib.update(frame, REGION, board, cnn_proba, hsv_color)
    assert not calib.is_ready(COLOR_GREEN)
    # 2 frame 目で 144、超える
    calib.update(frame, REGION, board, cnn_proba, hsv_color)
    assert calib.is_ready(COLOR_GREEN)


def test_per_video_ranges_only_when_ready() -> None:
    """動画別 ranges はサンプル数充足色のみ含む。"""
    calib = OnlineHsvCalibrator(min_samples=100)
    board = _make_full_board(COLOR_GREEN)
    frame = _make_frame(saturation=180, value=200, hue=60)
    cnn_proba = np.full((12, BOARD_COLS), 0.99, dtype=np.float32)
    hsv_color = np.full((12, BOARD_COLS), COLOR_GREEN, dtype=np.int32)
    # GRN のみ 144 cells 超
    calib.update(frame, REGION, board, cnn_proba, hsv_color)
    calib.update(frame, REGION, board, cnn_proba, hsv_color)
    ranges = calib.get_per_video_ranges()
    assert COLOR_GREEN in ranges
    assert COLOR_RED not in ranges  # RED はサンプルなし
    h_min, h_max, s_min, s_max, v_min, v_max = ranges[COLOR_GREEN]
    # 入力 hue=60 なので範囲内に入っているはず
    assert h_min <= 60 <= h_max


def test_reset_clears_stats() -> None:
    calib = OnlineHsvCalibrator()
    board = _make_full_board(COLOR_GREEN)
    frame = _make_frame(saturation=180, value=200, hue=60)
    cnn_proba = np.full((12, BOARD_COLS), 0.99, dtype=np.float32)
    hsv_color = np.full((12, BOARD_COLS), COLOR_GREEN, dtype=np.int32)
    calib.update(frame, REGION, board, cnn_proba, hsv_color)
    assert calib.get_sample_counts()[COLOR_GREEN] > 0
    calib.reset()
    assert all(n == 0 for n in calib.get_sample_counts().values())


# ============================
# E (cycle 56) 赤色 H 循環バグ修正
# ============================


def test_circular_h_range_no_wrap_normal_case() -> None:
    """折り返しなし (= 黄色 H=14-38 等) は通常計算."""
    from src.online_hsv_calibrator import _circular_h_range

    h_samples = [20.0, 22.0, 25.0, 28.0, 30.0, 24.0, 26.0, 23.0, 27.0, 29.0]
    h_min, h_max = _circular_h_range(h_samples, mult=1.5)
    # mean ≈ 25.4, std ≈ 3.0, range ≈ (20.8, 30.0) → 整数化で (20, 29-30 程度)
    assert h_min <= 21
    assert h_max >= 28
    assert h_max <= 35  # 範囲が過剰拡張していない


def test_circular_h_range_wrap_red_case() -> None:
    """折り返しあり (= 赤色 H=0-13 + 166-180 等) で主要 cluster 選択."""
    from src.online_hsv_calibrator import _circular_h_range

    # 低 H 側 (= 0-13) 主体 + 高 H 側 (= 170-180) 少数
    h_samples = (
        [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]  # 低 H 側 7 件
        + [172.0, 178.0]  # 高 H 側 2 件
    )
    h_min, h_max = _circular_h_range(h_samples, mult=1.5)
    # 主要 cluster (= 低 H 側) のみ採用
    assert h_min == 0
    # 低 H 側の mean ~6, std ~4.3, mult=1.5 → range ~ -0.5 〜 12.5
    assert h_max <= 20  # クリップ後 0..20 程度
    # 注: 高 H 側 (170-180) は image_reader の wrap-around 用 default range が補完


def test_circular_h_range_wrap_high_majority() -> None:
    """折り返しあり、 主要 cluster が高 H 側のケース."""
    from src.online_hsv_calibrator import _circular_h_range

    h_samples = (
        [3.0, 5.0]  # 低 H 側 2 件
        + [168.0, 170.0, 172.0, 174.0, 176.0, 178.0, 180.0]  # 高 H 側 7 件
    )
    h_min, h_max = _circular_h_range(h_samples, mult=1.5)
    # 主要 cluster (= 高 H 側) のみ採用
    assert h_max == 180
    assert h_min >= 160  # 高 H 側 mean ~174, std ~4, mult=1.5 → range ~ 168-180


def test_hsv_range_red_circular_bug_fixed() -> None:
    """E 統合: 赤系 H 分布 (= 0/180 跨ぐ) で hsv_range が崩壊しないこと.

    バグ前: h_mean ≈ 90 (= 緑!) → range が壊れる
    バグ修正後: 主要 cluster 選択で正しい range
    """
    from src.online_hsv_calibrator import _ColorStats

    stats = _ColorStats()
    # 赤の典型 H 分布: 低 H 側 主体 (0-10) + 高 H 側 少数 (170-180)
    for h_val in [0, 2, 4, 6, 8, 10, 12, 5, 3, 7, 9, 4, 6]:
        stats.update(float(h_val), 200.0, 200.0)
    for h_val in [175, 178]:
        stats.update(float(h_val), 200.0, 200.0)

    h_min, h_max, _, _, _, _ = stats.hsv_range()
    # 主要 cluster (= 低 H 側 0-12) を採用、 中央値が緑 90 にならない
    assert h_min < 30, f"h_min {h_min} = 主要 cluster 採用なら 0-20 程度"
    assert h_max < 30, f"h_max {h_max} = 主要 cluster 採用なら 0-30 程度"
    # かつ「90 を中心とした緑域」 にならない
    assert not (60 <= h_min and h_max <= 120), \
        "緑域に倒れている = circular fix が機能していない"


def test_hsv_range_yellow_no_circular_unchanged() -> None:
    """E: 黄色 (= H 14-38) は折り返しなし、 既存挙動と同等."""
    from src.online_hsv_calibrator import _ColorStats

    stats = _ColorStats()
    for h_val in [20, 22, 24, 26, 28, 30, 25, 23, 27, 29]:
        stats.update(float(h_val), 200.0, 200.0)

    h_min, h_max, _, _, _, _ = stats.hsv_range()
    # 通常計算 = 14-38 域に収まる
    assert 0 <= h_min <= 30
    assert 25 <= h_max <= 50
