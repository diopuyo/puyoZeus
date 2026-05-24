"""src/adaptive_background.py のテスト (Phase T v2 サイクル T-v2-C)。"""
from __future__ import annotations

import cv2
import numpy as np

from src.adaptive_background import (
    AdaptiveBackgroundFingerprint,
    DEFAULT_UPDATE_DISTANCE_MAX,
)
from src.background_fingerprint import (
    BackgroundFingerprint,
    CellFingerprint,
)
from src.board import BOARD_COLS, VISIBLE_ROWS


def _all_zero_fp() -> BackgroundFingerprint:
    cells = tuple(
        tuple(CellFingerprint(0, 0, 0) for _ in range(BOARD_COLS))
        for _ in range(VISIBLE_ROWS)
    )
    return BackgroundFingerprint(cells=cells)


def _frame_with_uniform_bgr(
    bgr: tuple[int, int, int],
    w: int = 384,
    h: int = 720,
) -> np.ndarray:
    """指定 BGR で塗り潰したフレーム。"""
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:] = bgr
    return f


def test_initial_fp_matches_base() -> None:
    """初期状態は base FP と同じ値を返す。"""
    base = BackgroundFingerprint(
        cells=tuple(
            tuple(CellFingerprint(50, 100, 150) for _ in range(BOARD_COLS))
            for _ in range(VISIBLE_ROWS)
        )
    )
    adaptive = AdaptiveBackgroundFingerprint(base)
    cell = adaptive.cell_at(0, 0)
    assert cell.h == 50 and cell.s == 100 and cell.v == 150


def test_to_fingerprint_returns_compatible() -> None:
    """to_fingerprint で BackgroundFingerprint 互換オブジェクト取得。"""
    base = _all_zero_fp()
    adaptive = AdaptiveBackgroundFingerprint(base)
    fp = adaptive.to_fingerprint()
    assert isinstance(fp, BackgroundFingerprint)
    assert len(fp.cells) == VISIBLE_ROWS


def test_update_skipped_when_distance_too_large() -> None:
    """背景距離 > 閾値のセル → 更新スキップ。"""
    base = _all_zero_fp()  # 全部 (0,0,0)
    adaptive = AdaptiveBackgroundFingerprint(base, learning_rate=0.5)
    # 明るい色のフレーム → BGR=(200,200,200) → HSV=(0,0,200) で V 差大 → 更新スキップ
    frame = _frame_with_uniform_bgr((200, 200, 200))
    updated = adaptive.update(frame, 0, 0, 384, 720)
    # 全セル更新スキップ
    assert updated == 0
    # 背景は (0,0,0) のまま
    assert adaptive.cell_at(0, 0).v == 0


def test_update_applies_when_distance_small() -> None:
    """背景距離 < 閾値のセル → 移動平均で更新。"""
    # base = HSV(0, 0, 50) を全セルに設定
    base = BackgroundFingerprint(
        cells=tuple(
            tuple(CellFingerprint(0, 0, 50) for _ in range(BOARD_COLS))
            for _ in range(VISIBLE_ROWS)
        )
    )
    adaptive = AdaptiveBackgroundFingerprint(base, learning_rate=0.5)
    # 暗いフレーム HSV ≈ (0, 0, 30) → 距離 |50-30|=20 < 閾値 → 更新
    frame = _frame_with_uniform_bgr((30, 30, 30))
    updated = adaptive.update(frame, 0, 0, 384, 720)
    assert updated == VISIBLE_ROWS * BOARD_COLS
    # 移動平均: 50 * 0.5 + 30 * 0.5 = 40
    cell = adaptive.cell_at(0, 0)
    assert 35 <= cell.v <= 45


def test_update_converges_with_repeated_frames() -> None:
    """同じフレームを何度も渡すと背景値が現在 HSV に収束する。"""
    base = BackgroundFingerprint(
        cells=tuple(
            tuple(CellFingerprint(0, 0, 50) for _ in range(BOARD_COLS))
            for _ in range(VISIBLE_ROWS)
        )
    )
    adaptive = AdaptiveBackgroundFingerprint(base, learning_rate=0.5)
    frame = _frame_with_uniform_bgr((30, 30, 30))
    for _ in range(20):
        adaptive.update(frame, 0, 0, 384, 720)
    cell = adaptive.cell_at(0, 0)
    # 20 回更新後 → ほぼ 30 に収束
    assert abs(cell.v - 30) < 3


def test_update_no_op_for_invalid_frame() -> None:
    """None フレーム → 0 セル更新。"""
    base = _all_zero_fp()
    adaptive = AdaptiveBackgroundFingerprint(base)
    updated = adaptive.update(None, 0, 0, 384, 720)  # type: ignore[arg-type]
    assert updated == 0


def test_threshold_customization() -> None:
    """閾値を大きくすれば、遠いセルでも更新される。"""
    base = _all_zero_fp()
    adaptive = AdaptiveBackgroundFingerprint(
        base,
        update_distance_max=DEFAULT_UPDATE_DISTANCE_MAX * 10,
        learning_rate=0.5,
    )
    frame = _frame_with_uniform_bgr((100, 100, 100))
    updated = adaptive.update(frame, 0, 0, 384, 720)
    assert updated > 0
