"""原候補を保持して段差横置きを補完する私有列挙器。実走への自動注入はしない。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

VERSION = 'split-landing/v1'
SPLIT_ORIENTATION = 'chigiri'
PAIR_SIZE = 2


@dataclass(frozen=True)
class SplitCandidate:
    cells: tuple[tuple[int, int], tuple[int, int]]
    colors: tuple[int, int]
    orientation: str = SPLIT_ORIENTATION


def identity(hidden: Any) -> dict:
    return dict(version=VERSION, sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_placement_sha256=hidden.P.SHA, placement_prior_calibrated=False,
        old_receipt_replay_permission=False, runtime_connected=False)


def split_cells(hidden: Any, world: tuple) -> tuple:
    board = hidden.B.Board.from_dict({'grid': world})
    top = [hidden.P.module._top_empty_row(board, col) for col in range(hidden.B.BOARD_COLS)]
    return tuple(((top[col], col), (top[col + 1], col + 1))
        for col in range(hidden.B.BOARD_COLS - 1)
        if top[col] is not None and top[col + 1] is not None and top[col] != top[col + 1])


def enumerate_candidates(hidden: Any, world: tuple, pair: tuple[int, int]) -> tuple:
    hidden.B.require(type(pair) is tuple and len(pair) == PAIR_SIZE
        and all(type(color) is int and color in hidden.B.PIECE_COLORS for color in pair), 'split_pair')
    original = hidden.enumerate_hypotheses(world, pair)
    colors = (pair,) if pair[0] == pair[1] else (pair, pair[::-1])
    added = tuple(SplitCandidate(cells, choice) for cells in split_cells(hidden, world) for choice in colors)
    result = original + added
    hidden.B.require(len({(candidate.cells, candidate.colors) for candidate in result}) == len(result),
        'split_duplicate_candidate')
    return result


def apply_candidate(hidden: Any, world: tuple, candidate: Any) -> tuple:
    if type(candidate) is hidden.Hypothesis:
        return hidden.apply_hypothesis(world, candidate)
    hidden.B.require(type(candidate) is SplitCandidate and candidate.orientation == SPLIT_ORIENTATION,
        'split_candidate_type')
    hidden.B.require(candidate.cells in split_cells(hidden, world), 'split_cells_not_supported')
    hidden.B.require(type(candidate.colors) is tuple and len(candidate.colors) == PAIR_SIZE
        and all(type(color) is int and color in hidden.B.PIECE_COLORS for color in candidate.colors), 'split_colors')
    board = hidden.B.Board.from_dict({'grid': world})
    for (row, col), color in zip(candidate.cells, candidate.colors):
        board.set(row, col, color)
    after = hidden.B.grid(board)
    hidden.B.require(hidden.B.supported(after), 'split_gravity')
    hidden.B.require(sum(c != 0 for row in after for c in row)
        == sum(c != 0 for row in world for c in row) + PAIR_SIZE, 'split_overwrite')
    return after


def weighted_candidates(hidden: Any, world: Any, pair: tuple[int, int], prior: Any) -> tuple:
    hidden.B.require(type(prior) is hidden.PlacementPrior and prior.calibrated is False
        and type(prior.assumption) is str and bool(prior.assumption), 'split_prior_type')
    candidates = enumerate_candidates(hidden, world.grid, pair)
    hidden.B.require(bool(candidates), 'split_no_placement')
    weights = tuple(prior.weight(candidate) for candidate in candidates)
    hidden.B.require(all(type(weight) is float and math.isfinite(weight) and weight > 0
        for weight in weights), 'split_prior_weight')
    total = math.fsum(weights)
    hidden.B.require(math.isfinite(total) and total > 0, 'split_prior_total')
    return tuple((candidate, world.weight * weight / total) for candidate, weight in zip(candidates, weights))
