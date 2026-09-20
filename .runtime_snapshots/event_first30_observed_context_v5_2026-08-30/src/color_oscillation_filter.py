"""B2: 直近 N フレームで色が振動しているセルを UNKNOWN に補正。

CNN/HSV 認識誤りで「色 A → 色 B → 色 A」のように同セルの色がフレーム間で
変動する場合がある (特に着地直前/直後、テクスチャ干渉など)。
本フィルタは raw 観測を直近 window_size フレーム分保持し、各セルで
通常色 (EMPTY/UNKNOWN/OJAMA 以外) が 2 種以上出現したセルを UNKNOWN に
変換する。

設計上の注意:
    - 連鎖中 (色消去 → 落下 → 別色再配置) は正常な色変化なので、
      AnimationFilter で連鎖中フレームをスキップした上で本フィルタを使う想定。
    - StatefulBoardTracker の前段で適用し、Stateful には UNKNOWN を
      含む観測を渡す。
    - OJAMA は震動対象外 (色振動しない、上書きもされない)。
"""
from __future__ import annotations

from collections import deque

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    Board,
)

# 通常色 (振動検出対象。これらが 2 種以上観測されたら振動)
NORMAL_COLORS: frozenset[int] = frozenset({1, 2, 3, 4, 5})  # RED..PURPLE

DEFAULT_WINDOW_SIZE: int = 5
DEFAULT_MIN_DISTINCT: int = 2


class ColorOscillationFilter:
    """raw 観測の直近 N フレームで色振動セルを UNKNOWN に。

    使い方:
        filt = ColorOscillationFilter()
        for raw_obs in observations:
            stable = filt.update(raw_obs)
            # stable は色振動セルが UNKNOWN 化された Board
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        min_distinct_colors: int = DEFAULT_MIN_DISTINCT,
    ) -> None:
        self._window = int(window_size)
        self._min_distinct = int(min_distinct_colors)
        self._history: deque[Board] = deque(maxlen=self._window)

    def reset(self) -> None:
        self._history.clear()

    def update(self, observation: Board) -> Board:
        """観測を履歴に追加し、振動セルを UNKNOWN 化した Board を返す。

        履歴が window_size に満たない場合はそのまま返す。
        """
        self._history.append(observation.copy())
        if len(self._history) < self._window:
            return observation.copy()
        new_board = observation.copy()
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                colors = self._distinct_normal_colors(row, col)
                if len(colors) >= self._min_distinct:
                    new_board.set(row, col, COLOR_UNKNOWN)
        return new_board

    def _distinct_normal_colors(
        self, row: int, col: int,
    ) -> set[int]:
        """履歴 N フレーム中、(row, col) で観測された通常色集合。"""
        out: set[int] = set()
        for hb in self._history:
            c = int(hb.get(row, col))
            if c in NORMAL_COLORS:
                out.add(c)
        return out


__all__ = [
    "ColorOscillationFilter",
    "DEFAULT_MIN_DISTINCT",
    "DEFAULT_WINDOW_SIZE",
    "NORMAL_COLORS",
]
