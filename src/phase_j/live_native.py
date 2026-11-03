"""評価子限定の単盤面Rust計算。UNKNOWN壁と全段のPython結果型を保持する。"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import logging
import os
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_OJAMA, COLOR_UNKNOWN, HIDDEN_ROWS, Board
from src.chain import ChainResult, ChainSimulator, ChainStep, MIN_ERASE_COUNT, PuyoGroup
from src import exchange_fast_expand as expand
from . import live_connectivity as connectivity

CELLS = BOARD_ROWS * BOARD_COLS
RECORD = CELLS * 3
MAX_STEPS = CELLS // MIN_ERASE_COUNT
BYTE_PTR = ctypes.POINTER(ctypes.c_uint8)
LIBRARY = Path(__file__).with_name('puyo_live_exact.dll')
LOGGER = logging.getLogger(__name__)
_load_warning_emitted = False
NATIVE_HIDDEN_ROWS = (0, HIDDEN_ROWS)


def valid_input(grid: np.ndarray, hidden: int = 0) -> bool:
    """符号・桁落ち変換の前にC ABIの固定盤面と整数範囲を検査する。"""
    return (isinstance(grid, np.ndarray)
            and grid.shape == (BOARD_ROWS, BOARD_COLS)
            and grid.dtype.kind in 'iu'
            and isinstance(hidden, (int, np.integer))
            and hidden in NATIVE_HIDDEN_ROWS
            and bool(np.all((grid >= 0) & (grid <= COLOR_UNKNOWN))))


def load_library() -> Any:
    """C ABIを設定する。読み込み失敗の扱いは呼出側で決める。"""
    library = ctypes.CDLL(str(LIBRARY))
    library.live_stable.argtypes = [BYTE_PTR, ctypes.c_size_t]
    library.live_stable.restype = ctypes.c_bool
    library.live_local.argtypes = [BYTE_PTR, BYTE_PTR, ctypes.c_size_t, ctypes.c_size_t]
    library.live_local.restype = ctypes.c_bool
    library.live_simulate.argtypes = [BYTE_PTR, ctypes.c_size_t, BYTE_PTR]
    library.live_simulate.restype = ctypes.c_size_t
    return library


def packed(grid: np.ndarray) -> Any:
    """C ABIへ渡す盤面を固定長で複製する。"""
    if not valid_input(grid):
        raise ValueError('native盤面の形状・整数値の範囲が不正')
    return (ctypes.c_uint8 * CELLS).from_buffer_copy(grid.astype(np.uint8, copy=False).tobytes())


def board_from(raw: Any, dtype: np.dtype) -> Board:
    """入力と同じdtypeの独立した盤面を復元する。"""
    return expand.new_board(np.array(raw, dtype=dtype).reshape(BOARD_ROWS, BOARD_COLS))


def restore_groups(before: Board, labels: Any) -> list[PuyoGroup]:
    """走査順のグループと隣接おじゃま集合を復元する。"""
    grouped: dict[int, list[int]] = {}
    for cell, label in enumerate(labels):
        if label:
            grouped.setdefault(label, []).append(cell)
    values = before._grid.ravel().tolist()
    return [PuyoGroup(values[cells[0]], frozenset(divmod(c, BOARD_COLS) for c in cells),
                     len(cells), frozenset(divmod(n, BOARD_COLS) for c in cells
                         for n in connectivity.NEIGHBORS[0][c] if values[n] == COLOR_OJAMA))
            for cells in grouped.values()]


def simulate(library: Any, simulator: ChainSimulator, board: Board) -> ChainResult:
    """連結・消去・落下をRustで実行し、全段の詳細まで既存型へ戻す。"""
    output = (ctypes.c_uint8 * (MAX_STEPS * RECORD))()
    count = library.live_simulate(packed(board._grid), int(simulator._exclude_hidden_row_from_pop), output)
    steps: list[ChainStep] = []
    for index in range(count):
        offset = index * RECORD
        before = board_from(output[offset:offset+CELLS], board._grid.dtype)
        after = board_from(output[offset+CELLS:offset+CELLS*2], board._grid.dtype)
        groups = restore_groups(before, output[offset+CELLS*2:offset+RECORD])
        ojama = len(set().union(*(g.ojama_adjacent for g in groups)))
        steps.append(ChainStep(index+1, groups, ojama, sum(g.size for g in groups), before, after))
    erased = sum(s.erased_count for s in steps)
    return ChainResult(steps, count, erased, sum(s.erased_ojama for s in steps),
                       steps[-1].board_after.copy() if steps else board.copy(), erased)


@contextmanager
def exact_native() -> Iterator[None]:
    """既存のキャッシュ・探索制御を保持して、純粋計算部分だけ差し替える。"""
    global _load_warning_emitted
    try:
        library = load_library()
    except (OSError, AttributeError) as error:
        if not _load_warning_emitted:
            LOGGER.warning('native DLLを読み込めないためPython経路を使用します: %s', error)
            _load_warning_emitted = True
        yield
        return
    original = ChainSimulator._simulate_uncached
    original_stable = connectivity.is_stable

    def uncached(self: ChainSimulator, board: Board) -> ChainResult:
        if type(self) is not ChainSimulator or not valid_input(
                board._grid, self._exclude_hidden_row_from_pop):
            return original(self, board)
        return simulate(library, self, board)

    def stable(grid: np.ndarray, hidden: int) -> bool:
        if not valid_input(grid, hidden):
            return original_stable(grid, hidden)
        return bool(library.live_stable(packed(grid), hidden))

    with (connectivity.exact_connectivity(),
          patch.object(connectivity, 'is_stable', stable),
          patch.object(ChainSimulator, '_simulate_uncached', uncached)):
        yield


@contextmanager
def configured_native() -> Iterator[None]:
    """既定OFF。評価子の処理範囲内でのみ有効化する。"""
    from src.production_config import LIVE_EXACT_NATIVE_DEFAULT
    enabled = os.environ.get('PUYO_LIVE_EXACT_NATIVE', str(int(LIVE_EXACT_NATIVE_DEFAULT))) == '1'
    if enabled:
        with exact_native():
            yield
    else:
        yield
