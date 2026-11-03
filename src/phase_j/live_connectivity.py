"""評価子だけで連結の有無を計算する。探索順・ノード数・証明は変更しない。"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import os
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN, HIDDEN_ROWS
from src.chain import ChainSimulator, MIN_ERASE_COUNT, NEIGHBOR_DELTAS, PuyoGroup
from src import exchange_fast_expand as expand

CELL_COUNT = BOARD_ROWS * BOARD_COLS
CACHE_SIZE = 2048
NON_GROUP_COLORS = (COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN)


def neighbors(hidden: int) -> tuple[tuple[int, ...], ...]:
    """従来の隣接順を保ち、盤端と隠し段の検査を事前計算する。"""
    return tuple(tuple((row + dr) * BOARD_COLS + col + dc
                       for dr, dc in NEIGHBOR_DELTAS
                       if hidden <= row + dr < BOARD_ROWS and 0 <= col + dc < BOARD_COLS)
                 for row in range(BOARD_ROWS) for col in range(BOARD_COLS))


NEIGHBORS = tuple(neighbors(hidden) for hidden in range(BOARD_ROWS + 1))


def local_pop(grid: np.ndarray, cells: list[tuple[int, int]], hidden: int) -> bool:
    """追加セルからの連結だけを調べる。色・訪問順・4個到達の意味は従来と同じ。"""
    adjacent = NEIGHBORS[hidden]
    seen: set[int] = set()
    for row, col in cells:
        start = row * BOARD_COLS + col
        if row < hidden or start in seen:
            continue
        color = grid.item(start)
        seen.add(start)
        stack, size = [start], 0
        while stack:
            cell = stack.pop()
            size += 1
            if size >= MIN_ERASE_COUNT:
                return True
            for other in adjacent[cell]:
                if other not in seen and grid.item(other) == color:
                    seen.add(other)
                    stack.append(other)
    return False


def is_stable(grid: np.ndarray, hidden: int) -> bool:
    """グループ詳細と隣接おじゃまを生成せず、4連結の有無だけを同じ規則で返す。"""
    values = grid.ravel().tolist()
    adjacent, seen = NEIGHBORS[hidden], [False] * CELL_COUNT
    for start in range(hidden * BOARD_COLS, CELL_COUNT):
        color = values[start]
        if seen[start] or color in NON_GROUP_COLORS:
            continue
        seen[start] = True
        stack, size = [start], 0
        while stack:
            cell = stack.pop()
            size += 1
            if size >= MIN_ERASE_COUNT:
                return False
            for other in adjacent[cell]:
                if not seen[other] and values[other] == color:
                    seen[other] = True
                    stack.append(other)
    return True


def erasable_groups(grid: np.ndarray, hidden: int) -> list[PuyoGroup]:
    """消去する連結だけ詳細を作る。BFS順と隣接おじゃまの隠し段規則も保持する。"""
    values = grid.ravel().tolist()
    adjacent, seen = NEIGHBORS[hidden], [False] * CELL_COUNT
    groups: list[PuyoGroup] = []
    for start in range(hidden * BOARD_COLS, CELL_COUNT):
        color = values[start]
        if seen[start] or color in NON_GROUP_COLORS:
            continue
        seen[start] = True
        cells = [start]
        for cell in cells:
            for other in adjacent[cell]:
                if not seen[other] and values[other] == color:
                    seen[other] = True
                    cells.append(other)
        if len(cells) < MIN_ERASE_COUNT:
            continue
        # おじゃまは幽霊連鎖規則でも隠し段の隣接分を含む。
        ojama = {divmod(other, BOARD_COLS) for cell in cells
                 for other in NEIGHBORS[0][cell] if values[other] == COLOR_OJAMA}
        groups.append(PuyoGroup(color=color, cells=frozenset(divmod(c, BOARD_COLS) for c in cells),
                                size=len(cells), ojama_adjacent=frozenset(ojama)))
    return groups


@contextmanager
def exact_connectivity() -> Iterator[dict[str, int]]:
    """既知の標準シミュレータだけで純粋な静止判定を再利用し、終了時に必ず戻す。"""
    original = expand.FastExpander.is_stable
    original_erasable = ChainSimulator._erasable_fast
    cache: OrderedDict = OrderedDict()
    counts = dict(hits=0, misses=0)

    def stable(self: Any, board: Any) -> bool:
        if type(self.sim) is not ChainSimulator:
            return original(self, board)
        grid = board._grid
        hidden = HIDDEN_ROWS if self.sim._exclude_hidden_row_from_pop else 0
        key = (grid.dtype.str, grid.shape, grid[hidden:].tobytes(), hidden)
        if key in cache:
            counts['hits'] += 1
            cache.move_to_end(key)
            return cache[key]
        counts['misses'] += 1
        value = is_stable(grid, hidden)
        cache[key] = value
        if len(cache) > CACHE_SIZE:
            cache.popitem(last=False)
        return value

    def erasable(self: Any, board: Any) -> list[PuyoGroup]:
        if type(self) is not ChainSimulator:
            return original_erasable(self, board)
        return erasable_groups(board._grid, HIDDEN_ROWS if self._exclude_hidden_row_from_pop else 0)

    with (patch.object(expand.FastExpander, 'is_stable', stable),
          patch.object(expand, 'local_pop', local_pop),
          patch.object(ChainSimulator, '_erasable_fast', erasable)):
        yield counts


@contextmanager
def configured_connectivity() -> Iterator[None]:
    """通常はOFF。呼出元が評価子の処理範囲を限定して有効化する。"""
    from src.production_config import LIVE_EXACT_CONNECTIVITY_DEFAULT
    enabled = os.environ.get('PUYO_LIVE_EXACT_CONNECTIVITY', str(int(LIVE_EXACT_CONNECTIVITY_DEFAULT))) == '1'
    if enabled:
        with exact_connectivity():
            yield
    else:
        yield
