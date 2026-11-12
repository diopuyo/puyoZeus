"""直近に確定した対が移動したとき、その追加だけを取り消す。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.board import Board

PAIR_SIZE = 2
COLOR_MIN = 1
COLOR_MAX = 5
TOP_ROWS = 3
MAX_TRACK_SEC = 1.5
SAFE_STATES = frozenset({'stable', 'tsumo_fall'})
Cell = tuple[int, int, int]


def _additions(base: Board, observed: Board) -> tuple[Cell, ...]:
    """空升への色追加を取得する。未知升の訂正は含めない。"""
    grid = observed._grid
    mask = (base._grid == 0) & (grid >= COLOR_MIN) & (grid <= COLOR_MAX)
    return tuple((int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(mask))


@dataclass
class PendingPair:
    base: Board
    cells: tuple[Cell, ...]
    time_sec: float


class TsumoRollback:
    """対以外は保持し、NEXT・連鎖・試合境界で証拠を破棄する。"""

    def __init__(self) -> None:
        self.pending: PendingPair | None = None
        self.rollback_count = 0
        self.last_rollback: tuple[Cell, ...] = ()

    def reset(self) -> None:
        self.pending = None
        self.last_rollback = ()

    def before_update(
        self, confirmed: Board | None, observed: Board, time_sec: float,
        *, active: bool, state: str, next_shift: bool, chain: bool,
    ) -> None:
        """確定を待たせず、後続の明確な位置変化だけで取り消す。"""
        self.last_rollback = ()
        pending = self.pending
        if not active or chain or next_shift or state not in SAFE_STATES:
            self.reset()
            return
        if pending is None or confirmed is None:
            return
        elapsed = time_sec - pending.time_sec
        if elapsed < 0 or elapsed > MAX_TRACK_SEC:
            self.reset()
            return
        moved = _additions(pending.base, observed)
        if len(moved) != PAIR_SIZE or moved == pending.cells:
            return
        if sorted(v for _, _, v in moved) != sorted(v for _, _, v in pending.cells):
            return
        # 元の位置が空または移動後の対の一部であることを必要とする。
        destinations = {(r, c) for r, c, _ in moved}
        if any(observed.get(r, c) != 0 and (r, c) not in destinations
               for r, c, _ in pending.cells):
            return
        restored = []
        for r, c, color in pending.cells:
            if confirmed.get(r, c) == color:
                confirmed.set(r, c, pending.base.get(r, c))
                restored.append((r, c, color))
        self.last_rollback = tuple(restored)
        self.rollback_count += bool(restored)
        self.pending = None

    def after_update(
        self, before: Board | None, confirmed: Board | None, time_sec: float,
        *, active: bool, state: str, next_shift: bool, chain: bool,
    ) -> None:
        """今回書いた対を控える。盤面の変更や確定待機はしない。"""
        if not active or chain or next_shift or state not in SAFE_STATES:
            self.reset()
            return
        if before is None or confirmed is None:
            return
        cells = _additions(before, confirmed)
        if len(cells) != PAIR_SIZE or not any(r < TOP_ROWS for r, _, _ in cells):
            return
        # 追加以外の変化を含む更新は、操作対の確定だと断定しない。
        if np.count_nonzero(before._grid != confirmed._grid) != PAIR_SIZE:
            return
        self.pending = PendingPair(before.copy(), cells, time_sec)
