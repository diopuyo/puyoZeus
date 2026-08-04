"""src/effect_glow_detector.py の単体テスト (案B, 2026-08-04)。

compute_cell_bright_ratio (V チャンネル高輝度画素比率) と is_effect_glow_active
(EFFECT_GATE_TOP_ROWS 窓内の bright_ratio_max 判定、閾値0.97) の純関数を検証する。
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from src.effect_glow_detector import (
    EFFECT_BRIGHT_RATIO_MAX_THRESHOLD,
    compute_cell_bright_ratio,
    is_effect_glow_active,
)
from src.image_reader import DEFAULT_P1_REGION

# 1920x1080 フルフレーム相当の合成フレームサイズ。
_FRAME_W = 1920
_FRAME_H = 1080


def _black_frame() -> np.ndarray:
    return np.zeros((_FRAME_H, _FRAME_W, 3), dtype=np.uint8)


def _paint_cell(
    frame: np.ndarray, row: int, col: int, bgr: tuple[int, int, int],
) -> None:
    """DEFAULT_P1_REGION の指定セル領域を単色で塗る (テスト補助)。"""
    x1, y1, x2, y2 = DEFAULT_P1_REGION.cell_sample_rect(row, col)
    frame[y1:y2, x1:x2] = bgr


# ============================
# compute_cell_bright_ratio
# ============================


def test_compute_cell_bright_ratio_all_black_is_zero() -> None:
    """全黒パッチ (V=0) の高輝度比率は 0.0。"""
    patch = np.zeros((10, 10, 3), dtype=np.uint8)
    assert compute_cell_bright_ratio(patch) == 0.0


def test_compute_cell_bright_ratio_all_white_is_one() -> None:
    """全白パッチ (BGR=255,255,255, V=255) の高輝度比率は 1.0。"""
    patch = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert compute_cell_bright_ratio(patch) == 1.0


def test_compute_cell_bright_ratio_half_white_is_about_half() -> None:
    """パッチの半分だけ白 (V>=閾値) にすると比率は約 0.5。"""
    patch = np.zeros((10, 10, 3), dtype=np.uint8)
    patch[:5, :] = 255  # 上半分のみ白
    assert compute_cell_bright_ratio(patch) == 0.5


def test_compute_cell_bright_ratio_exact_threshold_value() -> None:
    """10x10 パッチで白画素 97 個 → 比率はちょうど 0.97 (境界値検証用)。"""
    patch = np.zeros((10, 10, 3), dtype=np.uint8)
    flat = patch.reshape(-1, 3)
    flat[:97] = 255
    assert compute_cell_bright_ratio(patch) == EFFECT_BRIGHT_RATIO_MAX_THRESHOLD


# ============================
# is_effect_glow_active
# ============================


def test_is_effect_glow_active_black_frame_is_false() -> None:
    """全黒フレームでは高輝度バーストなし → False。"""
    frame = _black_frame()
    assert is_effect_glow_active(frame, DEFAULT_P1_REGION) is False


def test_is_effect_glow_active_top_row_burst_is_true() -> None:
    """EFFECT_GATE_TOP_ROWS (row1-3) 内のセルを高輝度で塗ると True。"""
    frame = _black_frame()
    _paint_cell(frame, row=1, col=0, bgr=(255, 255, 255))
    assert is_effect_glow_active(frame, DEFAULT_P1_REGION) is True


def test_is_effect_glow_active_outside_top_rows_is_false() -> None:
    """窓外 (row5) のみ高輝度でも EFFECT_GATE_TOP_ROWS 外なので False。"""
    frame = _black_frame()
    _paint_cell(frame, row=5, col=0, bgr=(255, 255, 255))
    assert is_effect_glow_active(frame, DEFAULT_P1_REGION) is False


def test_is_effect_glow_active_at_exact_threshold_is_false() -> None:
    """bright_ratio_max がちょうど閾値 (0.97) では判定は False (> の厳密不等号)。

    実セルサイズでは画素数の制約でちょうど 0.97 を作れないため、
    compute_cell_bright_ratio を monkeypatch して境界値を厳密に与える。
    """
    frame = _black_frame()
    with patch(
        "src.effect_glow_detector.compute_cell_bright_ratio",
        return_value=EFFECT_BRIGHT_RATIO_MAX_THRESHOLD,
    ):
        assert is_effect_glow_active(frame, DEFAULT_P1_REGION) is False
