"""初段の取り直しと再発火期限を現在までの観測だけで確認する。"""
from types import SimpleNamespace

import numpy as np
import pytest

from src.phase_j._live_ex_frozen.src import dl1k_retr_recheck as recheck
from src.phase_j._live_ex_frozen.src import dl1k_retr_deadline as deadline
from src.phase_j._live_ex_frozen.src.fire_board import ChainForecast, Forecast
from src.phase_j._live_ex_frozen.src import dl1k_retr_continue as continuation
from tests.test_retr_survivor_followup import small_case

FIRST = dict(valid=True, product=40)


def tracker() -> SimpleNamespace:
    board = np.zeros((13, 6), dtype=np.int8)
    fc = Forecast(1., (40,), (40,), 'none', None, 1.)
    return SimpleNamespace(fc=fc, trace=lambda b: ([[(1, 4)], [(2, 4)]], board), _pred=None)


def test_recheck_off_has_no_state_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(recheck.ENV, raising=False)
    value = tracker()
    before = vars(value).copy()
    recheck.observe(value, 1.1, FIRST, np.zeros((13, 6)))
    assert vars(value) == before


def test_recheck_requires_two_current_boards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(recheck.ENV, '1')
    value, board = tracker(), np.zeros((13, 6))
    recheck.observe(value, 1.1, FIRST, board)
    assert value.fc.steps == (40,)
    recheck.observe(value, 1.2, FIRST, board)
    assert value.fc.steps == (40, 320)
    assert value.fc.observed == (40,)
    assert value.fc.chain_id == 1.


@pytest.mark.parametrize('stamp,product', [(1.1, 40), (1.5, 40), (1.2, 50)])
def test_recheck_rejects_duplicate_stale_or_wrong_formula(
        monkeypatch: pytest.MonkeyPatch, stamp: float, product: int) -> None:
    monkeypatch.setenv(recheck.ENV, '1')
    value, board = tracker(), np.zeros((13, 6))
    recheck.observe(value, 1.1, FIRST, board)
    recheck.observe(value, stamp, dict(valid=True, product=product), board)
    assert value.fc.steps == (40,)


def test_new_chain_cannot_reuse_previous_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(recheck.ENV, '1')
    value, board = tracker(), np.zeros((13, 6))
    recheck.observe(value, 1.1, FIRST, board)
    value.fc.chain_id = 1.15
    recheck.observe(value, 1.2, FIRST, board)
    assert value.fc.steps == (40,)


def test_observation_hook_finds_board_after_first_formula(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(recheck.ENV, '1')
    board = np.zeros((13, 6), dtype=np.int8)
    value = ChainForecast(trace=lambda b: ([[(1, 4)], [(2, 4)]], board))
    value.observe(1., FIRST, None)
    value.observe(1.1, FIRST, board)
    assert value.fc.source == 'none'
    value.observe(1.2, FIRST, board)
    assert value.fc.source == 'observed_first_step_recheck'
    assert value.fc.steps == (40, 320)


@pytest.mark.parametrize('end,attacker,expected', [(10., 5., 1), (10., 10., 1), (8., 10., 3)])
def test_last_placement_may_fire(end: float, attacker: float, expected: int) -> None:
    assert deadline.hands(end, attacker, 1., 16) == expected


def test_small_chain_keeps_last_firing_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    run, _, _ = small_case(monkeypatch)
    old_end, old_h = continuation.timing(run, 0, 1, 40, 14)
    monkeypatch.setenv(deadline.ENV, '1')
    new_end, new_h = continuation.timing(run, 0, 1, 40, 14)
    assert new_end == old_end
    assert new_h == old_h+1


def test_falling_unknown_board_remains_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    run, board, calls = small_case(monkeypatch)
    monkeypatch.setenv(deadline.ENV, '1')
    run.engine.state.sides[0].state = 'OJAMA_FALL'
    assert continuation.apply(run, 0, board, 100, 14)
    assert run.extra['retR_h_p1'] == 0
    assert calls == []
