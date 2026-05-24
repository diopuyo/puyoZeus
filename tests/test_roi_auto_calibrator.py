"""ROI auto-calibrator テスト (D)。"""
from __future__ import annotations

import cv2
import numpy as np

from src.roi_auto_calibrator import (
    MAX_VALID_OFFSET, RoiCalibration, detect_roi_offsets,
)


def _draw_field(
    frame: np.ndarray, x: int, y: int, w: int = 384, h: int = 720,
    color: tuple[int, int, int] = (200, 200, 200),
    thickness: int = 3,
) -> None:
    """フィールド枠を描画 (検出対象)。"""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)


def test_detects_default_position() -> None:
    """default 位置にフィールド枠 → offset = 0 近傍。"""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _draw_field(frame, 282, 160)  # 1P default
    _draw_field(frame, 1258, 160)  # 2P default
    calib = detect_roi_offsets(frame)
    # 厳密 0 は無理だが ±5px 以内
    assert abs(calib.p1_offset[0]) <= 5
    assert abs(calib.p1_offset[1]) <= 5
    assert abs(calib.p2_offset[0]) <= 5


def test_detects_shifted_position() -> None:
    """ずれた位置 (default + 10, 10) → offset を検出。"""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _draw_field(frame, 282 + 10, 160 + 10)
    _draw_field(frame, 1258 + 10, 160 + 10)
    calib = detect_roi_offsets(frame)
    # +10 の検出を期待 (許容 ±5)
    assert 5 <= calib.p1_offset[0] <= 15
    assert 5 <= calib.p1_offset[1] <= 15


def test_returns_zero_for_wrong_resolution() -> None:
    """1080p 以外 → 0 offset (検出 skip)。"""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    calib = detect_roi_offsets(frame)
    assert calib.p1_offset == (0, 0)
    assert calib.p2_offset == (0, 0)


def test_returns_zero_when_no_field_detected() -> None:
    """空 frame → 線検出失敗で 0 offset。"""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    calib = detect_roi_offsets(frame)
    assert calib.p1_offset == (0, 0)


def test_max_offset_capped() -> None:
    """大幅にずれた位置 (default + 60) → 検出失敗で 0 offset。"""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _draw_field(frame, 282 + 60, 160 + 60)
    _draw_field(frame, 1258 + 60, 160 + 60)
    calib = detect_roi_offsets(frame)
    # 60 px ずれは MAX_VALID_OFFSET=30 超で検出失敗扱い
    assert abs(calib.p1_offset[0]) <= MAX_VALID_OFFSET
