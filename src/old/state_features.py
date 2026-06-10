"""W2.1: GameState を勝率予測モデル用の feature vector に変換。

入力: src.state_pipeline.GameState
出力: numpy float32 1D array (固定次元)

設計:
    - 盤面 P1/P2: 12 行 × 6 列 × 7 色 (空,赤,青,緑,黄,紫,お邪魔) one-hot = 504 dim each
    - ネクスト・ダブルネクスト P1/P2: 各 (top, bot) で 7 色 one-hot × 2 = 14 dim
    - 得点 P1/P2: log1p 正規化 = 1 dim each
    - pending ojama P1/P2: クリップ正規化 = 1 dim each

合計次元: 504*2 + 14*4 + 4 = 1068 dim

UNKNOWN セルは「空」と同じ扱いにする (one-hot は EMPTY 部分に 1 を立てる)。
スコア/ojama が None の場合は 0 を入れる。
"""
from __future__ import annotations

from dataclasses import dataclass

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
    COLOR_YELLOW,
    HIDDEN_ROWS,
    Board,
)

# 7 色 one-hot 順序: EMPTY, RED, BLUE, GREEN, YELLOW, PURPLE, OJAMA
NUM_COLOR_CLASSES: int = 7

_COLOR_TO_INDEX: dict[int, int] = {
    COLOR_EMPTY: 0,
    COLOR_RED: 1,
    COLOR_BLUE: 2,
    COLOR_GREEN: 3,
    COLOR_YELLOW: 4,
    COLOR_PURPLE: 5,
    COLOR_OJAMA: 6,
}

# 盤面エンコード次元 (隠し段除外、可視 12 行)
VISIBLE_ROWS: int = BOARD_ROWS - HIDDEN_ROWS
BOARD_FEATURE_DIM: int = VISIBLE_ROWS * BOARD_COLS * NUM_COLOR_CLASSES

# ペア (top, bot) one-hot 次元
PAIR_FEATURE_DIM: int = 2 * NUM_COLOR_CLASSES

# スコア log1p の最大値 (正規化用): 約 999999 → log1p ≈ 13.8
SCORE_LOG_NORM: float = 14.0
# pending ojama の上限 (実用上 200 個程度で頭打ち)
OJAMA_MAX: float = 200.0

# 全次元 (盤面 P1+P2 + next P1/P2 + dnext P1/P2 + score P1/P2 + ojama P1/P2)
TOTAL_FEATURE_DIM: int = (
    BOARD_FEATURE_DIM * 2 + PAIR_FEATURE_DIM * 4 + 4
)


def encode_board(board: Board) -> np.ndarray:
    """Board を (VISIBLE_ROWS, BOARD_COLS, 7) one-hot に変換 (flatten 前)。"""
    out = np.zeros(
        (VISIBLE_ROWS, BOARD_COLS, NUM_COLOR_CLASSES), dtype=np.float32,
    )
    for visible_row in range(VISIBLE_ROWS):
        row = visible_row + HIDDEN_ROWS
        for col in range(BOARD_COLS):
            color = int(board.get(row, col))
            idx = _COLOR_TO_INDEX.get(color, 0)  # UNKNOWN は EMPTY 扱い
            out[visible_row, col, idx] = 1.0
    return out


def encode_pair(pair: tuple[int, int] | None) -> np.ndarray:
    """ペア (top_color, bot_color) を 2*7=14 次元 one-hot に変換。"""
    out = np.zeros(2 * NUM_COLOR_CLASSES, dtype=np.float32)
    if pair is None:
        return out
    top, bot = pair
    out[_COLOR_TO_INDEX.get(int(top), 0)] = 1.0
    out[NUM_COLOR_CLASSES + _COLOR_TO_INDEX.get(int(bot), 0)] = 1.0
    return out


def encode_score(score: int | None) -> float:
    """scoreを log1p 正規化 (None → 0)。"""
    if score is None or score < 0:
        return 0.0
    return float(np.log1p(score) / SCORE_LOG_NORM)


def encode_ojama(ojama: int) -> float:
    """pending ojama を [0, 1] にクリップ正規化。"""
    if ojama is None or ojama < 0:
        return 0.0
    return float(min(ojama, OJAMA_MAX) / OJAMA_MAX)


def encode_state(state) -> np.ndarray:  # GameState (循環 import 回避で型注釈なし)
    """GameState → 固定次元 feature vector (TOTAL_FEATURE_DIM,)."""
    parts: list[np.ndarray] = []
    parts.append(encode_board(state.board_p1).flatten())
    parts.append(encode_board(state.board_p2).flatten())
    parts.append(encode_pair(state.next_p1))
    parts.append(encode_pair(state.next_p2))
    parts.append(encode_pair(state.dnext_p1))
    parts.append(encode_pair(state.dnext_p2))
    parts.append(np.array([
        encode_score(state.score_p1),
        encode_score(state.score_p2),
        encode_ojama(state.pending_ojama_p1),
        encode_ojama(state.pending_ojama_p2),
    ], dtype=np.float32))
    return np.concatenate(parts)


def swap_p1_p2(features: np.ndarray) -> np.ndarray:
    """encoded features の P1 部分と P2 部分を入れ替える。

    順序前提: [board_p1, board_p2, next_p1, next_p2, dnext_p1, dnext_p2,
               score_p1, score_p2, ojama_p1, ojama_p2]

    label も反転する必要がある (1P_won → 1 - 1P_won = 2P_won)。
    本関数は features のみ swap、label 反転は呼び出し側で行う。
    """
    if features.ndim == 1:
        out = features.copy()
        # board_p1 ↔ board_p2
        b1 = out[:BOARD_FEATURE_DIM].copy()
        b2 = out[BOARD_FEATURE_DIM:2 * BOARD_FEATURE_DIM].copy()
        out[:BOARD_FEATURE_DIM] = b2
        out[BOARD_FEATURE_DIM:2 * BOARD_FEATURE_DIM] = b1
        # next_p1 ↔ next_p2
        offset = 2 * BOARD_FEATURE_DIM
        n1 = out[offset:offset + PAIR_FEATURE_DIM].copy()
        n2 = out[offset + PAIR_FEATURE_DIM:offset + 2 * PAIR_FEATURE_DIM].copy()
        out[offset:offset + PAIR_FEATURE_DIM] = n2
        out[offset + PAIR_FEATURE_DIM:offset + 2 * PAIR_FEATURE_DIM] = n1
        # dnext_p1 ↔ dnext_p2
        offset += 2 * PAIR_FEATURE_DIM
        dn1 = out[offset:offset + PAIR_FEATURE_DIM].copy()
        dn2 = out[offset + PAIR_FEATURE_DIM:offset + 2 * PAIR_FEATURE_DIM].copy()
        out[offset:offset + PAIR_FEATURE_DIM] = dn2
        out[offset + PAIR_FEATURE_DIM:offset + 2 * PAIR_FEATURE_DIM] = dn1
        # score_p1 ↔ score_p2 (最後 4 dim の最初の 2 つ)
        out[-4], out[-3] = out[-3], out[-4]
        # ojama_p1 ↔ ojama_p2
        out[-2], out[-1] = out[-1], out[-2]
        return out
    else:
        # batch (N, D)
        return np.stack([swap_p1_p2(features[i]) for i in range(features.shape[0])])


__all__ = [
    "BOARD_FEATURE_DIM",
    "NUM_COLOR_CLASSES",
    "OJAMA_MAX",
    "PAIR_FEATURE_DIM",
    "SCORE_LOG_NORM",
    "TOTAL_FEATURE_DIM",
    "VISIBLE_ROWS",
    "encode_board",
    "encode_ojama",
    "encode_pair",
    "encode_score",
    "encode_state",
    "swap_p1_p2",
]
