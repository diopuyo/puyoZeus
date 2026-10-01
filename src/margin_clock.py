"""両者共通の1手目設置起点。判定は純関数、観測の保持は外部wrapper。"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from src.board import Board, COLOR_RED, COLOR_PURPLE
from src.board_state_machine import BoardState

PAIR_SIZE = 2
SIDE_COUNT = 2


@dataclass(frozen=True)
class PlacementObservation:
    """直前と今回のSTABLE確定盤面。生盤面や得点は設置証拠にしない。"""

    t_sec: float
    state: BoardState
    before: Board | None
    confirmed: Board | None
    first_placement_sec: float | None = None


def first_placement_origin(observations: Iterable[PlacementObservation]) -> float | None:
    """空盤面から色ぷよ1組が確定した観測のうち、両者で最も早い時刻。

    空のSTABLE、落下中、未知色、おじゃま、途中開始の既設置盤面は除外する。
    入力順序・側の順序に依存せず、入力を変更せず、内部状態を保持しない。
    """
    candidates: list[float] = []
    for row in observations:
        if row.first_placement_sec is not None:
            stamp = row.first_placement_sec
            if math.isfinite(stamp) and math.isfinite(row.t_sec) and stamp <= row.t_sec:
                candidates.append(float(stamp))
            continue
        if (row.state != BoardState.STABLE or row.before is None
                or row.confirmed is None or not math.isfinite(row.t_sec)):
            continue
        before, after = row.before._grid, row.confirmed._grid
        if np.any(before) or np.count_nonzero(after) != PAIR_SIZE:
            continue
        colors = (after >= COLOR_RED) & (after <= COLOR_PURPLE)
        if np.count_nonzero(colors) == PAIR_SIZE:
            candidates.append(float(row.t_sec))
    return min(candidates) if candidates else None


class MarginClock:
    """試合単位で確定観測を保持する。起点は1個だけで、側別時計は持たない。"""

    def __init__(self) -> None:
        self.origin: float | None = None
        self.game: int | None = None
        self.game_start: float | None = None
        self.previous: list[Board | None] = [None] * SIDE_COUNT
        self.missed: list[bool] = [False] * SIDE_COUNT

    def observe(self, sides: tuple[Any, Any], stamp: float, game: int,
                first_placement_times: tuple[float | None, float | None] | None = None) -> None:
        """試合境界でリセットし、最初の設置後は起点を凍結する。"""
        if self.game != game:
            self.__init__()
            self.game = game
            self.game_start = stamp
        if first_placement_times is not None:
            self._observe_times(first_placement_times, stamp)
            return
        if self.origin is not None:
            return
        # 相手の1手目を見逃した状態で、遅い側の設置を両者最早と断定しない。
        earliest_known = not any(self.missed) and all(
            board is not None and not np.any(board._grid) for board in self.previous)
        observations = []
        for idx, side in enumerate(sides):
            board = side.confirmed_board
            if side.state != BoardState.STABLE or board is None:
                continue
            if not self.missed[idx]:
                observations.append(PlacementObservation(stamp, side.state, self.previous[idx], board))
            # 最初に見えた非空盤面が2手以上なら1手目は復元不能。
            if self.previous[idx] is not None and np.any(self.previous[idx]._grid):
                self.missed[idx] = True
            self.previous[idx] = board.copy()
        self.origin = first_placement_origin(observations) if earliest_known else None

    def _observe_times(self, times: tuple[float | None, float | None], stamp: float) -> None:
        """認識済み設置時刻がある経路は盤面差分で代用せず、前試合の残値を除く。"""
        assert self.game_start is not None
        rows = [PlacementObservation(stamp, BoardState.STABLE, None, None, value)
                for value in times if value is not None and value >= self.game_start]
        origin = first_placement_origin(rows)
        if origin is not None:
            self.origin = origin if self.origin is None else min(origin, self.origin)

    def elapsed(self, stamp: float, fallback: float | None) -> float:
        """欠測時は旧時計を保持し、観測できた試合だけ起点を切り替える。"""
        origin = self.origin if self.origin is not None else fallback
        return 0.0 if origin is None else max(0.0, stamp - origin)


def placement_times_from_pipeline(pipeline: Any) -> tuple[float | None, float | None]:
    """既存の設置観測を読むだけ。収集・認識側の側別時計には書き込まない。"""
    return (getattr(pipeline, "_first_move_sec_1p", None),
            getattr(pipeline, "_first_move_sec_2p", None))
