"""色と得点の条件を追加の複数着弾死亡に限定する。"""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from src.board import Board
from src.exchange_landing_safety import ColorScoreGuard, LandingStateSafety
from src.exchange_event_multilanding import evaluate_multilanding

QUEUE = (1, 2, 3, 4)


def test_recolor_without_score_is_rejected() -> None:
    guard, board = ColorScoreGuard(), Board()
    board._grid[-1, :3] = 3
    guard.observe(board, QUEUE, 0., 0.)
    changed = board.copy()
    changed._grid[-1, 2] = 5
    guard.observe(changed, QUEUE, 0., 1.)
    assert guard.blocker(2.) == 'color_score_mismatch'
    guard.observe(board, QUEUE, 0., 2.)
    assert guard.blocker(3.) is None


def test_erasure_explained_by_score_is_accepted() -> None:
    guard, board = ColorScoreGuard(), Board()
    board._grid[-1, :4] = 1
    guard.observe(board, QUEUE, 0., 0.)
    guard.observe(Board(), QUEUE, 40., 1.)
    assert guard.blocker(2.) is None


@pytest.mark.parametrize('score', [0., 10., 39.])
def test_drop_bonus_not_treated_as_erasure(score: float) -> None:
    guard, board = ColorScoreGuard(), Board()
    board._grid[-1, 0] = 1
    guard.observe(board, QUEUE, 0., 0.)
    guard.observe(Board(), QUEUE, score, 1.)
    assert guard.blocker(2.) == 'color_score_mismatch'


def test_next_jitter_requires_quiet_confirmation() -> None:
    guard = ColorScoreGuard()
    guard.observe(Board(), QUEUE, 0., 0.)
    assert guard.blocker(.1) == 'board_unsettled'
    assert guard.blocker(1.) is None
    guard.observe(Board(), (5, 5, 1, 2), 0., 1.)
    assert guard.blocker(1.1) == 'board_unsettled'


@pytest.mark.parametrize('score', [None, float('nan')])
def test_missing_score_does_not_certify_board(score: float | None) -> None:
    guard = ColorScoreGuard()
    guard.observe(Board(), QUEUE, score, 0.)
    assert guard.blocker(1.) == 'missing_board_score'


def test_boundary_discards_integrity_and_recovery_state() -> None:
    safety = LandingStateSafety()
    safety.guards[0].observe(Board(), QUEUE, 0., 0.)
    safety.ledger.observe(1, {(0, 1): (90, True)}, (0, 0))
    safety.reset()
    assert safety.ledger.pending == [0, 0]
    assert safety.guards[0].blocker(1.) == 'missing_stable_board'


@pytest.mark.parametrize('already_dead', [False, True])
def test_integrity_gates_only_new_multilanding_death(already_dead: bool,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    board = Board()
    board._grid[-5:, :] = 9
    projection = SimpleNamespace(safety=SimpleNamespace(blocker=lambda *args: 'color_score_mismatch'),
        _receivers=lambda *args: ((Board(), board), [0, 0]),
        _death_boards=lambda *args: ((Board(), board), (Board(), board), [True, True]),
        _known_budget=lambda *args: True, _verified_attack=lambda *args: True)
    latest = tuple(SimpleNamespace(board=b, queue=np.array(QUEUE)) for b in (Board(), board))
    monkeypatch.setattr('src.exchange_event_multilanding.cached_proof',
        lambda *args: pytest.fail('整合しない新規死亡の探索を呼んだ'))
    value = dict(dead_sides=['2P'] if already_dead else [], base_p1=.5, gfe_p1=.5)
    result = evaluate_multilanding(projection, SimpleNamespace(tracker=None),
                                  latest, [0, 171], (1, 1), 1., value)
    assert result['dead_sides'] == (['2P'] if already_dead else [])
    assert result['multi_landing'][1]['reason'] == ('already_dead' if already_dead else 'color_score_mismatch')
