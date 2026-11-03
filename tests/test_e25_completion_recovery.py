"""実測不一致・UNKNOWN・曖昧さを復元によって確定扱いしない。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from src.board import Board
from src.chain import ChainSimulator
from src.exchange_completion_recovery import CompletionRecovery, enumerate_completions


def chain() -> SimpleNamespace:
    return SimpleNamespace(chain_id=1, side='2P', trigger_sec=1., predicted_final_board=None,
        predicted_final_score=0, predicted_chain_count=0, formula_total=40,
        score_delta=None, drop_bonus_score=0)


def seed(recovery: CompletionRecovery, item: SimpleNamespace, board: Board, stamp: float = .9) -> None:
    recovery.seed(item, SimpleNamespace(before_board=None),
                  [SimpleNamespace(t_sec=stamp, board=board)], ChainSimulator())


def test_zenchi_rejects_third_step_inconsistency() -> None:
    path = Path('logs/e25/COMPLETION_CAUSE.json')
    board = Board.from_list(json.loads(path.read_text())['board'])
    recovery, item = CompletionRecovery(), chain()
    seed(recovery, item, board)
    for count, score in ((1, 100), (2, 420), (3, 1320)):
        item.formula_total = score
        recovery.observe(item, count, float(count))
    assert item.predicted_final_board is None
    assert recovery.entries[1]['reason'] == 'prefix_mismatch'
    assert item.predicted_final_score == 0


def test_terminal_candidate_waits_for_real_end_confirmation() -> None:
    recovery, item = CompletionRecovery(), chain()
    recovery.entries[1] = dict(options=[dict(prefix=(40,), board=Board()._grid.tolist())],
        recover=True, original=(0, 0, None), last=None)
    recovery.observe(item, 1, 1.)
    assert item.predicted_final_board is None
    assert recovery.entries[1]['reason'] == 'awaiting_end_confirmation'
    item.end_confirmed = True
    recovery.observe(item, 1, 2.)
    assert item.predicted_final_board is not None


def test_atomic_formula_ignores_ahead_of_count_cumulative_value() -> None:
    recovery, item = CompletionRecovery(), chain()
    recovery.entries[1] = dict(options=[dict(prefix=(40, 360), board=Board()._grid.tolist())],
        recover=True, original=(0, 0, None), last=None)
    item.formula_total = 360
    recovery.observe(item, 1, 1., observed_score=40)
    assert recovery.consistent(item) and item.predicted_final_score == 360


@pytest.mark.parametrize('row', [0, 6, 12])
def test_unknown_never_enumerated(row: int) -> None:
    board = Board()
    board._grid[row, 0] = 10
    assert enumerate_completions(board, ChainSimulator()) == []


def test_missing_origin_cannot_recover() -> None:
    recovery = CompletionRecovery()
    recovery.seed(chain(), None, [], ChainSimulator())
    assert not recovery.entries


def test_stale_origin_cannot_recover() -> None:
    recovery, board = CompletionRecovery(), Board()
    board._grid[-1, :3] = 1
    seed(recovery, chain(), board, stamp=0.)
    assert not recovery.entries


def test_unique_candidate_retracted_on_observation_conflict() -> None:
    recovery, item = CompletionRecovery(), chain()
    empty = Board()._grid.tolist()
    recovery.entries[1] = dict(options=[dict(prefix=(40, 360), board=empty)],
        recover=True, original=(0, 0, None), last=None)
    recovery.observe(item, 1, 1.)
    assert item.predicted_chain_count == 2 and recovery.consistent(item)
    item.formula_total = 400
    recovery.observe(item, 2, 2.)
    assert item.predicted_final_board is None and not recovery.consistent(item)


def test_existing_prediction_unchanged_by_integrity_rejection() -> None:
    board, item, recovery = Board(), chain(), CompletionRecovery()
    board._grid[-1, :4] = 1
    item.predicted_final_board = Board()._grid.tolist()
    item.predicted_final_score, item.predicted_chain_count = 40, 1
    seed(recovery, item, board)
    item.formula_total = 80
    recovery.observe(item, 1, 1.)
    assert not recovery.consistent(item)
    assert item.predicted_final_score == 40 and item.predicted_final_board is not None


def test_ambiguous_final_boards_not_adopted() -> None:
    recovery, item = CompletionRecovery(), chain()
    empty, occupied = Board()._grid.tolist(), np.ones((13, 6), dtype=int).tolist()
    recovery.entries[1] = dict(options=[dict(prefix=(40,), board=b) for b in (empty, occupied)],
        recover=True, original=(0, 0, None), last=None)
    recovery.observe(item, 1, 1.)
    assert item.predicted_final_board is None
