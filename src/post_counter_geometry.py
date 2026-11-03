"""E35専用の不変な盤面幾何。探索順や火力上限には介入しない。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.board import Board, BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS, COLOR_OJAMA, DEATH_COL, DEATH_ROW
from src.chain import ChainSimulator, ChainResult, MIN_ERASE_COUNT

COLORS = (1, 2, 3, 4, 5)
GEOMETRY_CACHE_SIZE = 32768
FULL_MASK = (1 << (BOARD_ROWS * BOARD_COLS)) - 1
VISIBLE_MASK = FULL_MASK ^ ((1 << (HIDDEN_ROWS * BOARD_COLS)) - 1)
LEFT_EDGE = sum(1 << (row * BOARD_COLS) for row in range(BOARD_ROWS))
RIGHT_EDGE = LEFT_EDGE << (BOARD_COLS - 1)
PAIR_SIZE = 2
CELLS = tuple((index // BOARD_COLS, index % BOARD_COLS, 1 << index)
              for index in range(BOARD_ROWS * BOARD_COLS))
FREE_COLUMNS = tuple(tuple(sum(1 << (row * BOARD_COLS + col) for row in range(HIDDEN_ROWS, top))
                           for top in range(BOARD_ROWS + 1)) for col in range(BOARD_COLS))


def neighbors(bits: int) -> int:
    """行境界で左右が折り返さない四近傍を返す。"""
    return ((bits << BOARD_COLS) | (bits >> BOARD_COLS) |
            ((bits & ~LEFT_EDGE) >> 1) | ((bits & ~RIGHT_EDGE) << 1))


@lru_cache(maxsize=GEOMETRY_CACHE_SIZE)
def component(seed: int, mask: int) -> int:
    """集合の拡張が止まるまで整数ビット演算で連結成分を求める。"""
    group = seed
    while True:
        expanded = group | (neighbors(group) & mask)
        if expanded == group:
            return group
        group = expanded


@lru_cache(maxsize=GEOMETRY_CACHE_SIZE)
def geometry(grid: bytes) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
    """列の最上占有位置、可視色マスク、到達空きマス、全色数を共有する。"""
    tops = [BOARD_ROWS] * BOARD_COLS
    planes = [0] * (max(COLORS) + 1)
    count = 0
    for color, (row, col, bit) in zip(grid, CELLS):
        if not color:
            continue
        if tops[col] == BOARD_ROWS:
            tops[col] = row
        if color in COLORS:
            planes[color] |= bit
            count += 1
    free = sum(FREE_COLUMNS[col][top] for col, top in enumerate(tops))
    return tuple(tops), tuple(p & VISIBLE_MASK for p in planes), free, count


def usable_colors(board: Board, supply: Counter) -> int:
    """空きマスで橋渡しした同色連結の既存数と置き足し数を数える。"""
    _, planes, free, count = geometry(board._grid.tobytes())
    for color in COLORS:
        remaining = occupied = planes[color]
        available = supply[color]
        if occupied.bit_count() + available < MIN_ERASE_COUNT:
            continue
        mask = occupied | free
        while remaining:
            group = component(remaining & -remaining, mask)
            if (group & occupied).bit_count() + min((group & free).bit_count(), available) >= MIN_ERASE_COUNT:
                return count
            remaining &= ~group
    return 0


def has_clear(board: Board) -> bool:
    """消去があるときだけ詳細な連鎖シミュレータへ進む。"""
    return any(has_four(occupied) for occupied in geometry(board._grid.tobytes())[1])


@lru_cache(maxsize=GEOMETRY_CACHE_SIZE)
def has_four(bits: int) -> bool:
    """四連結以上は次数3の頂点、または隣り合う次数2以上の頂点を持つ。"""
    up, down = bits & (bits << BOARD_COLS), bits & (bits >> BOARD_COLS)
    left = bits & ((bits & ~RIGHT_EDGE) << 1)
    right = bits & ((bits & ~LEFT_EDGE) >> 1)
    vertical, horizontal = up | down, left | right
    two = (up & down) | (left & right) | (vertical & horizontal)
    three = (up & down & horizontal) | (left & right & vertical)
    return bool(three or (two & neighbors(two)))


def all_landings_dead(board: Board, count: int) -> bool:
    """端数を窒息列以外へ置いても、均等分だけで窒息するかを検査する。"""
    return geometry(board._grid.tobytes())[0][DEATH_COL] - count // BOARD_COLS <= DEATH_ROW


def place_pair(board: Board, pair: tuple[int, int], col: int, rotation: int) -> Board | None:
    """接地済みの既知色盤面に従来と同じ22配置を列高から直接置く。"""
    tops = geometry(board._grid.tobytes())[0]
    if rotation % PAIR_SIZE == 0:
        if tops[col] < PAIR_SIZE:
            return None
        upper, lower = pair if rotation == 0 else pair[::-1]
        cells = ((tops[col]-PAIR_SIZE, col, upper), (tops[col]-1, col, lower))
    else:
        if min(tops[col:col+PAIR_SIZE]) < 1:
            return None
        left, right = pair if rotation == 1 else pair[::-1]
        cells = ((tops[col]-1, col, left), (tops[col+1]-1, col+1, right))
    work = board.copy()
    for row, column, color in cells:
        work._grid[row, column] = color
    return work


def erase_mask(planes: tuple[int, ...]) -> int:
    """可視色だけから4連結以上の消去集合を返す。"""
    erased = 0
    for occupied in planes:
        if not has_four(occupied):
            continue
        remaining = occupied
        while remaining.bit_count() >= MIN_ERASE_COUNT:
            group = component(remaining & -remaining, occupied)
            if group.bit_count() >= MIN_ERASE_COUNT:
                erased |= group
            remaining &= ~group
    return erased


@lru_cache(maxsize=GEOMETRY_CACHE_SIZE)
def reply_bytes(grid: bytes) -> tuple[bytes, int, int, int]:
    """上限計算が使う完走盤面と消去数だけを求め、段別画像の生成を省く。"""
    chains = total_erased = total_ojama = 0
    while True:
        erased = erase_mask(geometry(grid)[1])
        if not erased:
            return grid, chains, total_erased, total_ojama
        ojama = sum(bit for color, (_, _, bit) in zip(grid, CELLS) if color == COLOR_OJAMA)
        adjacent = neighbors(erased) & ojama
        total_erased += erased.bit_count()
        total_ojama += adjacent.bit_count()
        chains += 1
        removed, cells = erased | adjacent, bytearray(grid)
        while removed:
            bit = removed & -removed
            cells[bit.bit_length()-1] = 0
            removed ^= bit
        for col in range(BOARD_COLS):
            values = bytes(color for color in cells[col::BOARD_COLS] if color)
            cells[col::BOARD_COLS] = bytes(BOARD_ROWS-len(values)) + values
        grid = bytes(cells)


@dataclass(frozen=True)
class BoundReply:
    """詳細連鎖ログが不要な上限計算専用の結果。"""
    final_board: Board
    chain_count: int
    total_erased: int
    total_ojama: int


class BoundSimulator(ChainSimulator):
    """非消去手と接地済み盤面への着弾を短絡するE35専用シミュレータ。"""

    def simulate(self, board: Board) -> ChainResult:
        """消去なしの詳細グループ生成を省く。消去ありは従来実装を使う。"""
        if has_clear(board):
            return super().simulate(board)
        return ChainResult(steps=[], chain_count=0, total_erased=0, total_ojama=0,
                           final_board=board.copy(), participating_cells=0)

    def simulate_reply(self, board: Board) -> BoundReply:
        """可変盤面は共有せず、キャッシュするのは不変なbytesと整数だけ。"""
        grid, chains, erased, ojama = reply_bytes(board._grid.tobytes())
        final = Board()
        final._grid = np.frombuffer(grid, dtype=np.uint8).reshape(BOARD_ROWS, BOARD_COLS).copy()
        return BoundReply(final, chains, erased, ojama)

    def drop_ojama_with_remainder_columns(self, board: Board, ojama_count: int,
                                         remainder_columns: tuple[int, ...]) -> Board:
        """上限探索の接地済み入力に、従来と同じ均等分と端数を置く。"""
        counts = self._calc_explicit_ojama_drop_counts(ojama_count, remainder_columns)
        tops = geometry(board._grid.tobytes())[0]
        work = board.copy()
        for col, (top, count) in enumerate(zip(tops, counts)):
            work._grid[max(0, top-count):top, col] = COLOR_OJAMA
        return work
