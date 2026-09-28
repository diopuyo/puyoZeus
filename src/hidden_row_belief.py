"""不可視段の因果的な周辺分布。観測による検証と予測更新を分離する。"""
from __future__ import annotations

from collections import Counter
from itertools import product
from math import prod
from typing import Any
import numpy as np

from src.board import Board, BOARD_COLS, BOARD_ROWS, COLOR_UNKNOWN
from src.board_state_machine import BoardState
from src.probabilistic_board import ProbabilisticCell
from src.chain import ChainSimulator
from src.prefire_snapshot_reader import grounded
from src import puyo_core_bridge as native

OCCUPANCY_PRIOR = 0.5
RETAINED_MASS = 0.99
MAX_COMBINATIONS = 4096
SETTLE_CONFIRMATIONS = 2
PAIR_SIZE = 2
LOG_EPSILON = 1e-12
CALIBRATION_BINS = 10


def prior(grid: np.ndarray, col: int, colors: tuple) -> ProbabilisticCell:
    """天井に達していない列は空、満杯列は占有事前と試合色一様を使う。"""
    if not colors or grid[1, col] == 0:
        return ProbabilisticCell.certain(0)
    return ProbabilisticCell({0: 1-OCCUPANCY_PRIOR,
                             **{c: OCCUPANCY_PRIOR/len(colors) for c in colors}})


def combinations(cells: list[ProbabilisticCell]) -> tuple[list[tuple], float]:
    """上位の結合確率を累積99%または定数上限まで保持する。"""
    choices = [sorted((c, p) for c, p in cell.probs.items() if p > 0) for cell in cells]
    ranked = sorted(((tuple(c for c, _ in row), prod(p for _, p in row))
                     for row in product(*choices)), key=lambda row: (-row[1], row[0]))
    selected, mass = [], 0.
    for row in ranked[:MAX_COMBINATIONS]:
        selected.append(row)
        mass += row[1]
        if mass >= RETAINED_MASS:
            break
    return [(row, weight/mass) for row, weight in selected], mass


class HiddenRowBelief:
    """確定履歴とNEXT、独立した落ち切り観測だけを時系列で取り込む。"""

    def __init__(self, simulator: ChainSimulator) -> None:
        self.simulator = simulator
        self.cells = [ProbabilisticCell.certain(0) for _ in range(BOARD_COLS)]
        self.previous: np.ndarray | None = None
        self.queue: tuple = ()
        self.pair: tuple = ()
        self.pending: bytes | None = None
        self.samples = 0
        self.audit: list[dict] = []
        self.stamp: float | None = None
        self.placements = 0
        self.palette: tuple = ()

    def observe(self, side: Any, colors: tuple, stamp: float) -> None:
        """非STABLEは二回一致した予測専用観測だけを使用し、確定盤面には書かない。"""
        queue = tuple(side.next_pair or ()) + tuple(side.dnext_pair or ())
        shifted = (len(self.queue) == 2*PAIR_SIZE and len(queue) == 2*PAIR_SIZE
                   and queue != self.queue and self.queue[PAIR_SIZE:] == queue[:PAIR_SIZE])
        board = side.confirmed_board if side.state == BoardState.STABLE else getattr(side, 'midchain_board', None)
        if board is not None and self._usable(board, side.state):
            self.advance(board._grid, colors, stamp, self.pair)
        if shifted:
            if self.pair and self.previous is not None and side.state == BoardState.STABLE:
                self._placement(self.previous, colors, self.pair)
            self.pair = self.queue[:PAIR_SIZE]
        self.queue = queue

    def _usable(self, board: Board, state: BoardState) -> bool:
        """可視UNKNOWN・浮遊と単発の途中認識を排除する。"""
        grid = board._grid.copy()
        grid[0] = 0
        if np.any(grid[1:] == COLOR_UNKNOWN) or not grounded(grid):
            self.pending, self.samples = None, 0
            return False
        raw = grid.tobytes()
        self.samples = self.samples+1 if raw == self.pending else 1
        self.pending = raw
        return state == BoardState.STABLE or self.samples >= SETTLE_CONFIRMATIONS

    def advance(self, grid: np.ndarray, colors: tuple, stamp: float, pair: tuple = ()) -> None:
        """見えた真値を旧分布で採点してから、落下・配置の分布を更新する。"""
        visible = grid.copy()
        visible[0] = 0
        if self.previous is None or (not self.palette and colors):
            self.cells = [prior(visible, c, colors) for c in range(BOARD_COLS)]
        elif not np.array_equal(self.previous, visible):
            revealed = self._reveal(visible, stamp)
            placed = self._placement(visible, colors, pair)
            for col in range(BOARD_COLS):
                if visible[1, col] == 0:
                    self.cells[col] = ProbabilisticCell.certain(0)
                elif col not in revealed and col not in placed:
                    if not np.array_equal(self.previous[:, col], visible[:, col]):
                        self.cells[col] = prior(visible, col, colors)
        self.previous, self.stamp, self.palette = visible, stamp, colors

    def _placement(self, visible: np.ndarray, colors: tuple, pair: tuple) -> set[int]:
        """読んだ組の配置を可視増分で照合し、不可視へ入る色と列を周辺化する。"""
        if len(pair) != PAIR_SIZE or not set(pair) <= set(colors):
            return set()
        before = Board.from_list(self.previous.tolist())
        matches = [b._grid for _, _, b in native.enumerate_placements(before, pair, filter_dead=False)
                   if np.array_equal(b._grid[1:], visible[1:])]
        if not matches:
            return set()
        weighted = [(b, prod(self.cells[c].get(0) for c in range(BOARD_COLS) if b[0,c])) for b in matches]
        total = sum(w for _,w in weighted)
        if not total:
            return set()
        columns = {c for c in range(BOARD_COLS) if any(b[0, c] and w for b,w in weighted)}
        for col in columns:
            counts: Counter = Counter()
            for board, weight in weighted:
                distribution = {int(board[0,col]): 1.} if board[0,col] else self.cells[col].probs
                counts.update({c: weight*p/total for c,p in distribution.items()})
            self.cells[col] = ProbabilisticCell(dict(counts))
        self.placements += bool(columns)
        self.pair = ()
        return columns

    def _reveal(self, visible: np.ndarray, stamp: float) -> set[int]:
        """一段消去の全可視セルが一致する時だけ、旧row0の着地点を採点する。"""
        board = Board.from_list(self.previous.tolist())
        result = self.simulator.simulate(board)
        if not result.steps:
            return set()
        step = result.steps[0]
        expected = step.board_after._grid.copy()
        targets = {}
        for col in range(BOARD_COLS):
            removed = int(np.count_nonzero(self.previous[:, col])-np.count_nonzero(expected[:, col]))
            if self.previous[1, col] != 0 and removed:
                targets[col] = removed
                expected[removed, col] = visible[removed, col]
        if not targets or not np.array_equal(expected[1:], visible[1:]):
            return set()
        for col, row in targets.items():
            color = int(visible[row, col])
            cell = self.cells[col]
            predicted, confidence = cell.most_likely()
            self.audit.append(dict(prior_sec=self.stamp, observed_sec=stamp, col=col, row=row,
                probs=dict(cell.probs), actual=color, predicted=predicted, confidence=confidence,
                hit=predicted == color, log_loss=-float(np.log(max(LOG_EPSILON, cell.get(color))))))
            self.cells[col] = ProbabilisticCell.certain(0)
        return set(targets)

    def snapshot(self, grid: np.ndarray, colors: tuple) -> list[ProbabilisticCell]:
        """不可視画素の推定値を真値扱いせず、取得時の高さと履歴から分布を返す。"""
        return [ProbabilisticCell(dict(self.cells[c].probs))
                if self.previous is not None and self.previous[1, c] == grid[1, c]
                else prior(grid, c, colors) for c in range(BOARD_COLS)]
