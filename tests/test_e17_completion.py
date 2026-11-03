"""終了前・別側・次着手・おじゃまを混同しないことを検証する。"""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from src.board import Board, COLOR_OJAMA, COLOR_UNKNOWN
from src.board_state_machine import BoardState
from src.exchange_event_completion_check import CompletionVerifier
from src.exchange_event_overlay import ConfirmedSide
from src.exchange_event_tracker import ExchangeChainRecord


def context() -> tuple:
    board = Board()
    board._grid[-1, 0] = 1
    saved = ConfirmedSide(2., board, np.ones(4, int))
    overlay = NS(_history=[[saved], []], _completion_stamp=2.)
    result = NS(p1=NS(state=BoardState.STABLE), p2=NS(state=BoardState.CHAIN))
    chain = ExchangeChainRecord("1P", 1, 0., 0., end_signal_sec=1., end_confirmed=True,
                                predicted_final_board=board._grid.tolist())
    return CompletionVerifier(), overlay, result, chain


@pytest.mark.parametrize("state", [BoardState.CHAIN, BoardState.GRAVITY_SETTLE, BoardState.OJAMA_FALL])
def test_nonstable_is_not_compared(state: BoardState) -> None:
    verifier, overlay, result, chain = context()
    result.p1.state = state
    overlay._history[0][0].board._grid[-1, 0] = 2
    assert not verifier.check(overlay, result, chain)
    assert verifier.samples[1].board is None


def test_quiet_confirmation_uses_first_not_latest_board() -> None:
    verifier, overlay, result, chain = context()
    chain.end_confirmed = False
    assert not verifier.check(overlay, result, chain)
    later = Board()
    overlay._history[0].append(ConfirmedSide(3., later, np.ones(4, int)))
    chain.end_confirmed = True
    assert not verifier.check(overlay, result, chain)
    assert verifier.samples[1].stamp == 2.


@pytest.mark.parametrize("cutoff", [1.5, 2.])
def test_placement_before_first_board_skips_comparison(cutoff: float) -> None:
    verifier, overlay, result, chain = context()
    chain.post_end_drop_sec = cutoff
    assert not verifier.check(overlay, result, chain)
    assert verifier.samples[1].board is None


def test_next_tsumo_closes_window() -> None:
    verifier, overlay, result, chain = context()
    result.p1.state = BoardState.TSUMO_FALL
    overlay._completion_stamp = 1.5
    assert not verifier.check(overlay, result, chain)
    result.p1.state = BoardState.STABLE
    assert not verifier.check(overlay, result, chain)
    assert verifier.samples[1].board is None


def test_garbage_is_ignored_without_moving_colors() -> None:
    verifier, overlay, result, chain = context()
    overlay._history[0][0].board._grid[-1, 1] = COLOR_OJAMA
    assert not verifier.check(overlay, result, chain)
    assert verifier.samples[1].checked


def test_color_position_difference_is_detected() -> None:
    verifier, overlay, result, chain = context()
    overlay._history[0][0].board._grid[-1, 0] = 0
    overlay._history[0][0].board._grid[-2, 0] = 1
    assert verifier.check(overlay, result, chain)


def test_other_side_does_not_supply_board() -> None:
    verifier, overlay, result, chain = context()
    overlay._history.reverse()
    assert not verifier.check(overlay, result, chain)
    assert verifier.samples[1].board is None


def test_revoked_end_removes_mismatch() -> None:
    verifier, overlay, result, chain = context()
    overlay._history[0][0].board._grid[-1, 0] = 2
    assert verifier.check(overlay, result, chain)
    chain.end_signal_sec = None
    assert not verifier.check(overlay, result, chain)
    assert not verifier.samples


def test_unknown_first_board_is_not_replaced() -> None:
    verifier, overlay, result, chain = context()
    overlay._history[0][0].board._grid[-1, 0] = COLOR_UNKNOWN
    assert not verifier.check(overlay, result, chain)
    overlay._history[0].append(ConfirmedSide(3., Board(), np.ones(4, int)))
    assert not verifier.check(overlay, result, chain)
    assert verifier.samples[1].stamp == 2.


def test_mismatch_waits_for_end_confirmation() -> None:
    verifier, overlay, result, chain = context()
    overlay._history[0][0].board._grid[-1, 0] = 2
    chain.end_confirmed = False
    assert not verifier.check(overlay, result, chain)
    chain.end_confirmed = True
    assert verifier.check(overlay, result, chain)


def test_new_end_candidate_resets_old_comparison() -> None:
    verifier, overlay, result, chain = context()
    overlay._history[0][0].board._grid[-1, 0] = 2
    assert verifier.check(overlay, result, chain)
    chain.end_signal_sec = 3.
    assert not verifier.check(overlay, result, chain)
    assert verifier.samples[1].board is None


def test_tsumo_at_end_signal_is_already_next_move() -> None:
    verifier, overlay, result, chain = context()
    result.p1.state = BoardState.TSUMO_FALL
    overlay._completion_stamp = chain.end_signal_sec
    assert not verifier.check(overlay, result, chain)
    result.p1.state = BoardState.STABLE
    assert not verifier.check(overlay, result, chain)
    assert verifier.samples[1].board is None


def test_board_at_end_signal_is_not_after_completion() -> None:
    verifier, overlay, result, chain = context()
    chain.end_signal_sec = overlay._history[0][0].t_sec
    assert not verifier.check(overlay, result, chain)
    assert verifier.samples[1].board is None


def test_later_drop_does_not_reopen_comparison_window() -> None:
    verifier, overlay, result, chain = context()
    chain.post_end_drop_sec = 1.5
    assert not verifier.check(overlay, result, chain)
    chain.post_end_drop_sec = 3.
    assert not verifier.check(overlay, result, chain)
    assert verifier.samples[1].cutoff == 1.5
    assert verifier.samples[1].board is None
