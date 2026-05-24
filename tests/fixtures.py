"""
テスト用フィクスチャ生成ユーティリティ

既知の Board から合成フレーム画像 (BGR) を生成し、
ImageReader の往復整合性 (Board → image → Board) を検証可能にする。

HSV 値は ImageReader.DEFAULT_COLOR_RANGES の中央値を使用し、
現状の閾値で正しく読み取れることを保証する。
"""

from __future__ import annotations

import cv2
import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
    HIDDEN_ROWS,
    VISIBLE_ROWS,
    Board,
)
from src.image_reader import (
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
    BoardRegion,
)

# ============================
# 定数定義
# ============================

# フレームサイズ (ぷよぷよeスポーツ 1080p)
DEFAULT_FRAME_WIDTH: int = 1920
DEFAULT_FRAME_HEIGHT: int = 1080

# セルパッチ塗りつぶし比率 (セルに対する割合)
# ImageReader の CELL_SAMPLE_RATIO=0.4 より大きくしないと読めないので 0.7 に設定
CELL_FILL_RATIO: float = 0.7

# HSV 代表値 (各色範囲の内側にある安定値)
# 現状の DEFAULT_COLOR_RANGES で classify() が一意に判定できる値を選択
COLOR_HSV_SAMPLES: dict[int, tuple[int, int, int]] = {
    COLOR_RED:    (5,   220, 200),
    COLOR_BLUE:   (115, 200, 200),
    COLOR_GREEN:  (65,  200, 200),
    COLOR_YELLOW: (28,  220, 220),
    COLOR_PURPLE: (148, 200, 200),
    COLOR_OJAMA:  (0,   10,  200),   # 低彩度・明るい
    COLOR_EMPTY:  (0,   0,   0),     # 暗い (V=0)
}


# ============================
# 公開関数
# ============================


def hsv_to_bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    """HSV を OpenCV BGR に変換する。"""
    hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def render_board_on_frame(
    frame: np.ndarray,
    board: Board,
    region: BoardRegion,
    fill_ratio: float = CELL_FILL_RATIO,
) -> np.ndarray:
    """
    既存のフレームに Board のセルを描画する (in-place)。

    隠し段 (row < HIDDEN_ROWS) は画面外のため描画しない。
    UNKNOWN セルも描画対象外。

    Args:
        frame: 描画対象の BGR フレーム。
        board: レンダリングする盤面。
        region: 描画位置の BoardRegion。
        fill_ratio: セル内の塗りつぶし比率。

    Returns:
        np.ndarray: 描画後のフレーム (同一インスタンス)。
    """
    cell_w = region.cell_width
    cell_h = region.cell_height
    half_w = max(1, int(cell_w * fill_ratio / 2))
    half_h = max(1, int(cell_h * fill_ratio / 2))

    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            color = board.get(row, col)
            if color == COLOR_EMPTY or color == COLOR_UNKNOWN:
                continue
            h, s, v = COLOR_HSV_SAMPLES[color]
            bgr = hsv_to_bgr(h, s, v)
            cx, cy = region.cell_center(row, col)
            cv2.rectangle(
                frame,
                (cx - half_w, cy - half_h),
                (cx + half_w, cy + half_h),
                bgr,
                thickness=-1,
            )
    return frame


def make_synthetic_frame(
    board_1p: Board | None = None,
    board_2p: Board | None = None,
    width: int = DEFAULT_FRAME_WIDTH,
    height: int = DEFAULT_FRAME_HEIGHT,
    p1_region: BoardRegion | None = None,
    p2_region: BoardRegion | None = None,
) -> np.ndarray:
    """
    1P/2P 盤面を持つ合成フレームを生成する。

    Args:
        board_1p: 1P 側の盤面 (None なら空)。
        board_2p: 2P 側の盤面 (None なら空)。
        width: フレーム幅 (px)。
        height: フレーム高さ (px)。
        p1_region: 1P 盤面位置 (None でデフォルト)。
        p2_region: 2P 盤面位置 (None でデフォルト)。

    Returns:
        np.ndarray: shape=(H, W, 3) の BGR フレーム。
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    r1 = p1_region or DEFAULT_P1_REGION
    r2 = p2_region or DEFAULT_P2_REGION
    if board_1p is not None:
        render_board_on_frame(frame, board_1p, r1)
    if board_2p is not None:
        render_board_on_frame(frame, board_2p, r2)
    return frame


# ============================
# サンプル盤面
# ============================


def sample_all_colors_board() -> Board:
    """全5色 + おじゃまを含む代表盤面を返す。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    grid[12][0] = COLOR_RED
    grid[12][1] = COLOR_BLUE
    grid[12][2] = COLOR_GREEN
    grid[12][3] = COLOR_YELLOW
    grid[12][4] = COLOR_PURPLE
    grid[12][5] = COLOR_OJAMA
    return Board.from_list(grid)


def sample_stacked_board() -> Board:
    """各列に異なる高さでぷよを積み上げた盤面。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    colors = [COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW,
              COLOR_PURPLE, COLOR_OJAMA]
    for col, color in enumerate(colors):
        for row in range(12 - col, 13):
            grid[row][col] = color
    return Board.from_list(grid)


def sample_4_chain_board() -> Board:
    """4連鎖する盤面 (test_indicators.py の設計と同じ)。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    col0_seq = [
        COLOR_YELLOW, COLOR_YELLOW,
        COLOR_BLUE, COLOR_BLUE,
        COLOR_GREEN, COLOR_GREEN,
        COLOR_RED, COLOR_RED,
    ]
    for i, color in enumerate(col0_seq):
        grid[5 + i][0] = color
    other_seq = [COLOR_YELLOW, COLOR_BLUE, COLOR_GREEN, COLOR_RED]
    for col in (1, 2, 3):
        for i, color in enumerate(other_seq):
            grid[9 + i][col] = color
    return Board.from_list(grid)
