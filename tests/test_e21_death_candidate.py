"""死亡候補の解除・独立消去証拠・通知の再送境界を検証する。"""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from src.board import Board, DEATH_ROW, DEATH_COL
from src.board_state_machine import BoardState
from src.chain import ChainSimulator
from src.exchange_event_death_candidate import DeathCandidateGate, occupied, observed_erasure


def board(dead: int = 1, color: int = 2, count: int = 4) -> Board:
    """窒息セルと最下段の消去群を独立に作る。"""
    value = Board()
    value._grid[DEATH_ROW, DEATH_COL] = dead
    value._grid[-1, :count] = color
    return value


def side(value: Board, state: BoardState = BoardState.STABLE) -> NS:
    """観測される側を最小構成にする。"""
    return NS(confirmed_board=value, state=state)


def event(value: Board, stamp: float = 1.) -> NS:
    """通知値自体は消去実測の代わりに使わない。"""
    return NS(trigger_sec=stamp, mechanism='baseline', chain_count=9, total_score=99999, before_board=value)


def gate() -> DeathCandidateGate:
    """両側の死亡候補を初期化する。"""
    value = DeathCandidateGate(ChainSimulator(exclude_hidden_row_from_pop=True))
    value.observe((side(board()), side(board())), (), 0., 1)
    return value


@pytest.mark.parametrize('cell,expected', [(0, False), (1, True), (2, True), (3, True), (4, True), (5, True), (9, True), (10, False)])
def test_occupied(cell: int, expected: bool) -> None:
    assert occupied(board(cell)) is expected


def test_hidden_row_not_dead() -> None:
    value = board(0)
    value._grid[0, DEATH_COL] = 1
    assert not occupied(value)


@pytest.mark.parametrize('color', [1, 2, 3, 4, 5])
def test_erasure_requires_four_same_color(color: int) -> None:
    before, after = board(color=color), board(count=0)
    assert observed_erasure(before, after)
    assert not observed_erasure(board(color=color, count=3), after)


def test_garbage_loss_is_not_erasure() -> None:
    assert not observed_erasure(board(color=9), board(count=0))


def test_unknown_cannot_prove_erasure() -> None:
    value = board(count=0)
    value._grid[-2, 0] = 10
    assert not observed_erasure(board(), value)


def test_prediction_alone_is_held() -> None:
    value = gate()
    assert value.notifications(0, event(board()), 2.) == []
    assert len(value.pending[0]) == 1


def test_old_board_cannot_prove_new_erasure() -> None:
    value = gate()
    value.boards[0] = board(count=0)
    assert value.notifications(0, event(board()), 2.) == []


def test_independent_stable_erasure_accepts() -> None:
    value = gate()
    value.notifications(0, event(board()), 2.)
    value.observe((side(board(count=0)), side(board())), (), 3., 1)
    assert len(value.notifications(0, None, 3.)) == 1
    assert value.candidates[0] is not None


def test_nonstable_prediction_does_not_count_as_observed() -> None:
    value = gate()
    value.notifications(0, event(board()), 2.)
    value.observe((side(board(count=0), BoardState.CHAIN), side(board())), (), 3., 1)
    assert value.notifications(0, None, 3.) == []


def test_candidate_clear_releases_original_notification() -> None:
    value = gate()
    original = event(board())
    value.notifications(0, original, 2.)
    value.observe((side(board(0)), side(board())), (), 3., 1)
    released = value.notifications(0, None, 3.)
    assert len(released) == 1 and released[0].trigger_sec == original.trigger_sec
    assert value.audit[0]['outcome'] == 'cleared'


def test_duplicate_notification_released_once() -> None:
    value = gate()
    for stamp in (2., 3., 4.):
        value.notifications(0, event(board()), stamp)
    assert len(value.pending[0]) == len(value.audit[0]['held']) == 1


@pytest.mark.parametrize('dead', [('1P',), ('2P',), ('1P', '2P')])
def test_confirmed_death_drops_pending(dead: tuple) -> None:
    value = gate()
    for idx in (0, 1):
        value.notifications(idx, event(board()), 2.)
    value.observe((side(board()), side(board())), dead, 3., 1)
    for label in dead:
        idx = int(label == '2P')
        assert not value.pending[idx] and not value.notifications(idx, event(board()), 4.)


def test_reset_does_not_replay_previous_game() -> None:
    value = gate()
    value.notifications(0, event(board()), 2.)
    value.reset()
    assert not value.notifications(0, None, 3.)


def test_opponent_notification_unaffected() -> None:
    value = gate()
    value.observe((side(board()), side(board(0))), (), 2., 1)
    original = event(board(0))
    assert value.notifications(1, original, 2.) == [original]


def test_pending_copy_is_immutable_from_caller() -> None:
    value, original = gate(), event(board())
    value.notifications(0, original, 2.)
    original.before_board._grid[:] = 0
    assert occupied(next(iter(value.pending[0].values())).before_board)


def test_zero_prediction_not_accepted_by_reported_chain_count() -> None:
    value = gate()
    value.notifications(0, event(board(count=0)), 2.)
    value.observe((side(board(count=0)), side(board())), (), 3., 1)
    assert not value.notifications(0, None, 3.)


@pytest.mark.parametrize('function', ['overlay', 'replay', 'generate'])
def test_public_flag_defaults_off(function: str) -> None:
    import inspect
    from src.exchange_event_overlay import ExchangeEventOverlay
    from scripts.replay_exchange_event_20260926 import replay
    from scripts.visualize_advantage_overlay import generate
    target = dict(overlay=ExchangeEventOverlay, replay=replay, generate=generate)[function]
    assert inspect.signature(target).parameters['death_candidate_guard'].default is False


def test_accepted_chain_continuation_is_not_a_new_fire() -> None:
    value = gate()
    value.accepted[0].add(1.)
    continuation = event(board())
    assert value.notifications(0, continuation, 2.) == [continuation]
    assert not value.pending[0]
