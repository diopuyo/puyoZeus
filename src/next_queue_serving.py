"""提供側の queue 整列器 (状態を持つ外部ラッパー、2026-10-01)。

ExchangeEventOverlay の履歴 (ConfirmedSide.queue) に入れる NEXT/DNEXT を、
学習の補正 queue と同じ「次に置く組 P_k, その次 P_{k+1}」の意味へ揃える。
規則は exev scripts/_diag_next_shift_align_20260930.py の R3 (Frozen(2)) と同一:
- STABLE 区間 (同一確定盤面が続く間) ごとに queue4 を1度だけ決めて凍結する。
- 区間の最初の確定 queue (CONFIRM_FRAMES フレーム連続同一) が、前区間の最後の確定 queue から
  1手繰り上がっていれば (連鎖後・おじゃま後)、手中の組 = 前区間の next を先頭へ戻す。
- 確定までの間は生の queue を返す。非 STABLE のフレームを挟むと新区間。試合が変わると初期化。
"""
from __future__ import annotations

from typing import Any

import numpy as np

CONFIRM_FRAMES = 2
COLOR_MIN, COLOR_MAX = 1, 5
QUEUE_SIZE = 4
PAIR_CELLS = 2


def queue4(next_pair: tuple | None, dnext_pair: tuple | None) -> tuple[int, ...]:
    """(next, dnext) を queue4 へ。欠測は 0。"""
    return tuple(int(v) for v in (*(next_pair or (0, 0)), *(dnext_pair or (0, 0))))


def complete(q: tuple[int, ...]) -> bool:
    """4色とも色ぷよなら True。"""
    return len(q) == QUEUE_SIZE and all(COLOR_MIN <= v <= COLOR_MAX for v in q)


class FrozenQueue:
    """1側ぶんの R3 整列器。"""

    def __init__(self, confirm: int = CONFIRM_FRAMES) -> None:
        self.confirm, self.candidate, self.count = confirm, None, 0
        self.base: tuple | None = None
        self.last_confirmed: tuple | None = None
        self.frozen: tuple | None = None
        self.grid: np.ndarray | None = None
        self.left = False

    def leave(self) -> None:
        """非 STABLE を見た。次の STABLE は新区間。"""
        self.left = True
        self.candidate, self.count = None, 0

    def update(self, grid: np.ndarray, q: tuple[int, ...]) -> tuple[int, ...]:
        """STABLE フレームの確定盤面と生 queue から整列 queue を返す。"""
        if self.grid is None or self.left or not np.array_equal(self.grid, grid):
            self.base, self.frozen = self.last_confirmed, None
        self.grid, self.left = grid.copy(), False
        if complete(q):
            self.count = self.count + 1 if q == self.candidate else 1
            self.candidate = q
            if self.count >= self.confirm:
                self.last_confirmed = q
                if self.frozen is None:
                    self.frozen = self._first(q)
        return self.frozen if self.frozen is not None else q

    def _first(self, q: tuple[int, ...]) -> tuple[int, ...]:
        """区間最初の確定 queue。繰り上がり済みなら手中の組を先頭へ戻す。"""
        base = self.base
        shifted = base is not None and q != base and q[:PAIR_CELLS] == base[PAIR_CELLS:]
        return (*base[:PAIR_CELLS], *q[:PAIR_CELLS]) if shifted else q


class QueueAlignment:
    """両側の整列器。overlay.update の先頭で毎フレーム observe する。"""

    def __init__(self) -> None:
        self.game: int | None = None
        self.sides = [FrozenQueue(), FrozenQueue()]

    def observe(self, sides: tuple[Any, Any], game: int, stable_state: Any) -> tuple:
        """両側の整列 queue (STABLE でない側は None) を返す。試合境界で作り直す。"""
        if self.game != game:
            self.game, self.sides = game, [FrozenQueue(), FrozenQueue()]
        out = []
        for aligner, side in zip(self.sides, sides):
            if side.state != stable_state or side.confirmed_board is None:
                aligner.leave()
                out.append(None)
                continue
            raw = queue4(side.next_pair, side.dnext_pair)
            out.append(aligner.update(np.asarray(side.confirmed_board._grid), raw))
        return tuple(out)
