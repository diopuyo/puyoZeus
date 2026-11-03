"""確定盤面とNEXTの手番を照合し、移行途中の組合せを公開しない。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from src.board import COLOR_UNKNOWN

COLORS = (1, 2, 3, 4, 5)
PAIR_SIZE, QUEUE_SIZE = 2, 4
MAX_CANDIDATES = 64


@dataclass(frozen=True)
class SyncedCountSide:
    """同じ手番として採用した盤面と、そこから置けるNEXT2組。"""
    grid: np.ndarray
    queue: np.ndarray
    t_sec: float


def color_counts(grid: np.ndarray) -> np.ndarray:
    """おじゃま・空白・UNKNOWNを手番の色収支へ混ぜない。"""
    return np.array([np.count_nonzero(grid == color) for color in COLORS])


class CountTurnSynchronizer:
    """色収支とNEXTの繰上がりが一致した時だけ確定入力を交換する。"""
    def __init__(self) -> None:
        self.accepted: SyncedCountSide | None = None
        self.reason = "waiting_initial_pair"
        self.saw_chain = False
        self.saw_slide = False
        self.current_pair: tuple[int, ...] | None = None
        self.candidates: list[SyncedCountSide] = []

    def observe(self, grid: np.ndarray | None, queue: np.ndarray, stamp: float,
                stable: bool, slide: bool = False, chaining: bool = False) -> bool:
        """欠測やスライド中は保持し、既発火の完走と通常の設置を分離する。"""
        self.saw_chain = self.saw_chain or chaining
        self.saw_slide = self.saw_slide or slide
        if not stable or grid is None:
            self.reason = "not_stable"
            return False
        if slide or queue.shape != (QUEUE_SIZE,) or not np.isin(queue, COLORS).all():
            self.reason = "queue_in_transition"
            return False
        if np.any(grid == COLOR_UNKNOWN):
            self.reason = "unknown_board"
            return False
        if self.accepted is None:
            return self._accept(grid, queue, stamp, "initial_stable_pair")
        old = self.accepted
        if np.array_equal(old.grid, grid) and np.array_equal(old.queue, queue):
            self.reason = "unchanged"
            return False
        delta = color_counts(grid) - color_counts(old.grid)
        if self.saw_chain and not chaining:
            return self._accept(grid, queue, stamp, "observed_chain_completion")
        if not delta.any() and np.array_equal(old.queue, queue):
            return self._accept(grid, queue, stamp, "garbage_or_position_update")
        shifted = np.array_equal(queue[:PAIR_SIZE], old.queue[PAIR_SIZE:])
        if shifted:
            self.current_pair = tuple(int(v) for v in old.queue[:PAIR_SIZE])
        expected = np.array([np.count_nonzero(old.queue[:PAIR_SIZE] == color) for color in COLORS])
        if shifted and np.array_equal(delta, expected):
            return self._accept(grid, queue, stamp, "placement_and_queue_aligned")
        if self._reacquire(grid, queue, stamp):
            return True
        self.reason = "waiting_board_for_queue" if shifted else "waiting_queue_for_board"
        return False

    def _reacquire(self, grid: np.ndarray, queue: np.ndarray, stamp: float) -> bool:
        """欠測後も観測2点の設置色と繰上がりが一致すれば、その後の手番を確定する。"""
        for prior in reversed(self.candidates):
            shifted = np.array_equal(queue[:PAIR_SIZE], prior.queue[PAIR_SIZE:])
            expected = np.array([np.count_nonzero(prior.queue[:PAIR_SIZE] == c) for c in COLORS])
            if shifted and np.array_equal(color_counts(grid)-color_counts(prior.grid), expected):
                return self._accept(grid, queue, stamp, "reacquired_matching_placement")
        if not any(np.array_equal(grid, c.grid) and np.array_equal(queue, c.queue) for c in self.candidates):
            self.candidates.append(SyncedCountSide(grid.copy(), queue.copy(), stamp))
            self.candidates = self.candidates[-MAX_CANDIDATES:]
        return False

    def _accept(self, grid: np.ndarray, queue: np.ndarray, stamp: float, reason: str) -> bool:
        """採用値を複製し、外部の盤面更新で過去入力が変わらないようにする。"""
        self.accepted = SyncedCountSide(grid.copy(), queue.copy(), stamp)
        self.reason = reason
        self.saw_chain = self.saw_slide = False
        self.current_pair = None
        self.candidates.clear()
        return True
