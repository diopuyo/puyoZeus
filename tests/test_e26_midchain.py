"""途中盤面は次段式一致後だけ予測へ入り、現在層へ混ざらない。"""
from __future__ import annotations

from types import SimpleNamespace as NS
import numpy as np
import pytest

from src.board import Board
from src.board_state_machine import BoardState as S
from src.chain import ChainSimulator
from src.exchange_event_tracker import ExchangeChainRecord
from src.exchange_midchain_completion import MidchainCompletion, compact, remaining


def setup() -> tuple:
    """初段40点の観測後、次段が赤4個の盤面を準備する。"""
    chain = ExchangeChainRecord('1P', 1, 1., 1., predicted_final_score=0., predicted_chain_count=0)
    board = Board()
    board._grid[-1, :4] = 1
    engine = MidchainCompletion(ChainSimulator())
    overlay = NS(_game=1, tracker=NS(latest_chain=lambda side: chain if side == '1P' else None))
    result = NS(p1=NS(state=S.CHAIN, chain_event=NS(trigger_sec=1., mechanism='formula_read',
        chain_count=1, total_score=40.), midchain_board=None),
        p2=NS(state=S.STABLE, chain_event=None, midchain_board=None))
    engine.observe(overlay, result, 1.)
    return engine, overlay, result, chain, board


def sample(engine: MidchainCompletion, overlay: NS, result: NS, board: Board) -> None:
    """落ち切りの連続二観測を入力する。"""
    result.p1.state, result.p1.midchain_board = S.GRAVITY_SETTLE, board
    engine.observe(overlay, result, 1.1)
    engine.observe(overlay, result, 1.1+1/30)


def formula(engine: MidchainCompletion, overlay: NS, result: NS,
            count: int = 2, score: float = 360.) -> None:
    """独立した次段の掛け算読取を入力する。"""
    result.p1.chain_event = NS(trigger_sec=1., mechanism='formula_read', chain_count=count, total_score=score)
    result.p1.state = S.CHAIN
    engine.observe(overlay, result, 1.3)


def test_next_formula_adopts_without_mutating_observations() -> None:
    engine, overlay, result, chain, board = setup()
    saved = board._grid.copy()
    sample(engine, overlay, result, board)
    assert chain.predicted_final_board is None
    formula(engine, overlay, result)
    assert chain.predicted_final_score == 360 and chain.predicted_chain_count == 2
    assert np.count_nonzero(chain.predicted_final_board) == 0
    np.testing.assert_array_equal(board._grid, saved)
    assert engine.provenance(1)[0]['layer'] == 'prediction'


@pytest.mark.parametrize('count,score', [(2, 320.), (2, 361.), (3, 360.), (2, 400.)])
def test_mismatched_next_formula_discards(count: int, score: float) -> None:
    engine, overlay, result, chain, board = setup()
    sample(engine, overlay, result, board)
    formula(engine, overlay, result, count, score)
    assert chain.predicted_final_board is None
    assert engine.summary()['next_mismatch'] == 1


@pytest.mark.parametrize('row,col', [(0, 0), (1, 2), (12, 5)])
def test_unknown_never_generates_candidate(row: int, col: int) -> None:
    engine, overlay, result, chain, board = setup()
    board._grid[row, col] = 10
    sample(engine, overlay, result, board)
    formula(engine, overlay, result)
    assert not engine.audit and chain.predicted_final_board is None


def test_floating_cells_are_not_collapsed_to_fake_settlement() -> None:
    engine, overlay, result, chain, board = setup()
    board._grid[-3, 5] = 2
    sample(engine, overlay, result, board)
    formula(engine, overlay, result)
    assert not engine.audit and not compact(board)


@pytest.mark.parametrize('state', [S.STABLE, S.CHAIN, S.TSUMO_FALL, S.OJAMA_FALL, S.MENU])
def test_non_settle_boards_are_never_used(state: S) -> None:
    engine, overlay, result, chain, board = setup()
    result.p1.state, result.p1.midchain_board = state, board
    engine.observe(overlay, result, 1.1)
    engine.observe(overlay, result, 1.2)
    formula(engine, overlay, result)
    assert not engine.audit and chain.predicted_final_board is None


def test_one_frame_is_not_settlement() -> None:
    engine, overlay, result, chain, board = setup()
    result.p1.state, result.p1.midchain_board = S.GRAVITY_SETTLE, board
    engine.observe(overlay, result, 1.1)
    formula(engine, overlay, result)
    assert not engine.audit


def test_repeated_timestamp_does_not_count_as_two_frames() -> None:
    engine, overlay, result, chain, board = setup()
    result.p1.state, result.p1.midchain_board = S.GRAVITY_SETTLE, board
    engine.observe(overlay, result, 1.1)
    engine.observe(overlay, result, 1.1)
    formula(engine, overlay, result)
    assert not engine.audit


def test_board_change_resets_settlement_evidence() -> None:
    engine, overlay, result, chain, board = setup()
    result.p1.state, result.p1.midchain_board = S.GRAVITY_SETTLE, board
    engine.observe(overlay, result, 1.1)
    result.p1.midchain_board = board.copy()
    result.p1.midchain_board._grid[-1, 4] = 2
    engine.observe(overlay, result, 1.2)
    formula(engine, overlay, result)
    assert not engine.audit


def test_missing_raw_never_uses_confirmed_board() -> None:
    engine, overlay, result, chain, board = setup()
    result.p1.state, result.p1.confirmed_board = S.GRAVITY_SETTLE, board
    engine.observe(overlay, result, 1.1)
    engine.observe(overlay, result, 1.2)
    formula(engine, overlay, result)
    assert not engine.audit and engine.skipped['missing_board'] == 2


def test_boundary_discards_pending() -> None:
    engine, overlay, result, chain, board = setup()
    sample(engine, overlay, result, board)
    engine.reset()
    formula(engine, overlay, result)
    assert not engine.audit


def test_later_formula_conflict_restores_original_prediction() -> None:
    engine, overlay, result, chain, board = setup()
    sample(engine, overlay, result, board)
    formula(engine, overlay, result)
    formula(engine, overlay, result, 3, 1000.)
    assert chain.predicted_final_score == 0 and chain.predicted_final_board is None
    assert engine.audit[0]['revoke_reason'] == 'later_formula_mismatch'


def test_same_notification_not_double_counted() -> None:
    engine, overlay, result, chain, board = setup()
    sample(engine, overlay, result, board)
    formula(engine, overlay, result)
    engine.observe(overlay, result, 1.4)
    assert engine.summary()['accepted'] == 1


@pytest.mark.parametrize('score,expected', [(360., False), (400., True), (320., True)])
def test_final_score_is_audited(score: float, expected: bool) -> None:
    engine, overlay, result, chain, board = setup()
    sample(engine, overlay, result, board)
    formula(engine, overlay, result)
    chain.score_delta, chain.score_ready_sec, chain.end_signal_sec = score, 3., 2.
    assert engine.summary()['rows'][0]['final_mismatch'] is expected


def test_unresolved_final_is_not_counted_as_correct() -> None:
    engine, overlay, result, chain, board = setup()
    sample(engine, overlay, result, board)
    formula(engine, overlay, result)
    assert engine.summary()['final_unresolved'] == 1


def test_chain_offset_uses_third_step_bonus() -> None:
    board = setup()[-1]
    value = remaining(board, 2, 360., ChainSimulator())
    assert value['prefix'] == {3: 1000.}


def test_empty_remaining_chain_does_not_create_prediction() -> None:
    assert remaining(Board(), 1, 40., ChainSimulator()) is None


@pytest.mark.parametrize('mechanism', ['formula', 'physics', 'baseline'])
def test_only_formula_read_can_verify(mechanism: str) -> None:
    engine, overlay, result, chain, board = setup()
    sample(engine, overlay, result, board)
    result.p1.chain_event = NS(trigger_sec=1., mechanism=mechanism, chain_count=2, total_score=360.)
    engine.observe(overlay, result, 1.3)
    assert chain.predicted_final_board is None


@pytest.mark.parametrize('state', [S.MENU, S.TSUMO_FALL, S.OJAMA_FALL])
def test_operation_or_landing_discards_candidate(state: S) -> None:
    engine, overlay, result, chain, board = setup()
    sample(engine, overlay, result, board)
    result.p1.state = state
    engine.observe(overlay, result, 1.25)
    formula(engine, overlay, result)
    assert not engine.audit


def test_raw_prediction_board_never_enters_confirmed_history() -> None:
    from src.exchange_event_overlay import ExchangeEventOverlay
    overlay = object.__new__(ExchangeEventOverlay)
    overlay._history, overlay._snapshots, overlay._start = [[], []], [], None
    raw, confirmed = setup()[-1], Board()
    side = NS(state=S.GRAVITY_SETTLE, confirmed_board=confirmed, midchain_board=raw,
              next_pair=(1, 1), dnext_pair=(2, 2))
    overlay._remember((side, side), NS(), 1.)
    assert overlay._history == [[], []]
    side.state = S.STABLE
    overlay._remember((side, side), NS(), 2.)
    assert not np.any(overlay._history[0][-1].board._grid)


def test_unknown_hidden_row_is_preserved_by_reader() -> None:
    from src.midchain_board_reader import MidchainBoardReader
    reader = object.__new__(MidchainBoardReader)
    board = Board()
    board._grid[1:, 2] = 9
    reader.reader = NS(_p1_region=None, _p2_region=None, read_board=lambda *a, **k: board.copy())
    sides = (NS(state=S.GRAVITY_SETTLE), NS(state=S.STABLE))
    first, second = reader.read(np.zeros((1, 1, 3), np.uint8), sides)
    assert first._grid[0, 2] == 10 and second is None


@pytest.mark.parametrize('score', [None, float('nan'), float('inf'), -40.])
def test_invalid_score_does_not_verify(score: float | None) -> None:
    engine, overlay, result, chain, board = setup()
    sample(engine, overlay, result, board)
    formula(engine, overlay, result, 2, score)
    assert chain.predicted_final_board is None


def test_observation_gap_cannot_establish_settlement() -> None:
    engine, overlay, result, chain, board = setup()
    result.p1.state, result.p1.midchain_board = S.GRAVITY_SETTLE, board
    engine.observe(overlay, result, 1.1)
    engine.observe(overlay, result, 1.9)
    formula(engine, overlay, result)
    assert not engine.audit


def test_completed_prediction_not_attached_to_new_exchange() -> None:
    engine, overlay, result, chain, board = setup()
    sample(engine, overlay, result, board)
    formula(engine, overlay, result)
    assert not engine.provenance(1, [])
    assert engine.provenance(1, [chain])
    chain.end_signal_sec, chain.score_ready_sec = 2., 3.
    assert not engine.provenance(1, [chain])
