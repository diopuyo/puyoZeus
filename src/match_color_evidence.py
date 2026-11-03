"""既存の試合別上位4色方式を、未来参照なしの外部状態として保持する。"""
from __future__ import annotations

from collections import Counter
from typing import Any
import numpy as np
from src.board_state_machine import BoardState

GAME_COLOR_COUNT = 4
COLORS = frozenset((1, 2, 3, 4, 5))


class MatchColorEvidence:
    """collect_indicators_v2._GameColorTrackerと同じ確定セル頻度を累積する。"""

    def __init__(self) -> None:
        self.counts: Counter[int] = Counter()

    def observe(self, side: Any) -> None:
        """STABLE確定セルだけを加算し、連鎖中の読取や将来の全試合集計を使わない。"""
        if side.state != BoardState.STABLE or side.confirmed_board is None:
            return
        values, counts = np.unique(side.confirmed_board._grid, return_counts=True)
        self.counts.update({int(v): int(n) for v, n in zip(values, counts) if v in COLORS})

    def active(self) -> tuple[int, ...]:
        """4色未確認または4位同点は拒否し、恣意的な除外色選択を避ける。"""
        ranked = self.counts.most_common()
        if len(ranked) < GAME_COLOR_COUNT:
            return ()
        if len(ranked) > GAME_COLOR_COUNT and ranked[GAME_COLOR_COUNT-1][1] == ranked[GAME_COLOR_COUNT][1]:
            return ()
        return tuple(sorted(color for color, _ in ranked[:GAME_COLOR_COUNT]))
