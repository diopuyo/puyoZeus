"""multilanding 証明の打切り予告。探索しなくても分かる「ノード上限超過」を先に判定する。

`Search.tick()` は配置 1 個ごとに 1 ノード数える。1 段の展開が必要とするノード数は、
盤面の各列の高さだけから（連鎖シミュレーションなしで）正確に数えられる。上限を超えると
分かっている段は、途中まで探索しても最後は同じ `SearchCutoff('node_limit')`・同じ
ノード数 (上限+1) で終わるので、その無駄な探索を省く。出力は変えない。
"""
from __future__ import annotations

from math import comb

import numpy as np

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN

ROTATIONS_PER_SHAPE = 2  # 縦: 上下入替 / 横: 左右入替 (それぞれ 2 通り)
VERTICAL_ROOM = 2        # 縦置きに必要な空き段数
HORIZONTAL_ROOM = 1      # 横置きの各列に必要な空き段数


def column_heights(grid: np.ndarray) -> np.ndarray:
    """`Board.height_of` と同じ定義 (最上段の実ぷよまでの高さ、UNKNOWN 除外) を全列で返す。"""
    concrete = (grid != COLOR_EMPTY) & (grid != COLOR_UNKNOWN)
    top = np.where(concrete.any(axis=0), concrete.argmax(axis=0), BOARD_ROWS)
    return BOARD_ROWS - top


def placement_count(grid: np.ndarray) -> int:
    """`_enumerate_placements` が返す配置数 (置ける手の数)。ペアの色には依らない。"""
    room = BOARD_ROWS - column_heights(grid)
    vertical = int(np.count_nonzero(room >= VERTICAL_ROOM))
    ok = room >= HORIZONTAL_ROOM
    horizontal = int(np.count_nonzero(ok[:-1] & ok[1:]))
    return ROTATIONS_PER_SHAPE * (vertical + horizontal)


def ojama_column_combinations(dropped: int) -> int:
    """端数おじゃまの全列配置数 (`combinations(range(BOARD_COLS), dropped % BOARD_COLS)` の個数)。"""
    return comb(BOARD_COLS, dropped % BOARD_COLS)


def exceeds_limit(nodes: int, cost: int, limit: int) -> bool:
    """あと `cost` ノード数えると、上限を超えて打切りになるか。"""
    return nodes + cost > limit
