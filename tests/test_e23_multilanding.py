"""複数着弾の生存経路・不確実性・既定OFFを検証する。"""
from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from src.board import Board, COLOR_UNKNOWN
from src.exchange_event_landing import ExchangeLandingProjection
from src.exchange_event_multilanding import prove_multilanding, evaluate_multilanding, cached_proof, Search


def no_response(board: Board, queue: np.ndarray, hands: int, elapsed: float) -> float:
    """最大火力の補助上界を与えず、全配置による検査を残す。"""
    return 0.


def buried_board() -> Board:
    """1回は耐え、2回の上限着弾で窒息する受け盤面。"""
    board = Board()
    board._grid[-5:, :] = 9
    return board


def test_two_landings_kill_when_one_does_not() -> None:
    board = buried_board()
    before = board._grid.copy()
    result = prove_multilanding(board, (1, 2, 3, 4), 171, 1, 0., no_response)
    assert result['dead'] and len(result['rounds']) == 2
    assert result['rounds'][0][0]['dropped'] == 30
    assert result['rounds'][0][0]['remaining'] == 141
    np.testing.assert_array_equal(board._grid, before)


def test_one_landing_can_survive() -> None:
    result = prove_multilanding(buried_board(), (1, 2, 3, 4), 30, 1, 0., no_response)
    assert not result['dead'] and result['reason'] == 'surviving_response'


def test_response_recomputed_after_landing() -> None:
    heights = []
    def response(board: Board, queue: np.ndarray, hands: int, elapsed: float) -> float:
        heights.append(board.height_of(2))
        return 200. if board.height_of(2) >= 10 else 0.
    result = prove_multilanding(buried_board(), (1, 2, 3, 4), 171, 1, 0., response)
    assert not result['dead']
    assert 5 in heights and max(heights) >= 10


@pytest.mark.parametrize('limit', [0, 1, 22])
def test_cutoff_is_not_death(limit: int) -> None:
    result = prove_multilanding(buried_board(), (1, 2, 3, 4), 171, 1, 0., no_response,
                               node_limit=limit)
    assert not result['dead'] and result['reason'] == 'node_limit'


@pytest.mark.parametrize('row', [0, 1, 12])
def test_unknown_is_not_death(row: int) -> None:
    board = buried_board()
    board._grid[row, 0] = COLOR_UNKNOWN
    result = prove_multilanding(board, (1, 2, 3, 4), 171, 1, 0., no_response)
    assert not result['dead'] and result['reason'] == 'unknown_board'


def test_actual_fire_can_cancel_even_when_beam_reports_zero() -> None:
    board = Board()
    board._grid[-1, :3] = 1
    result = prove_multilanding(board, (1, 1, 1, 1), 1, 1, 0., no_response)
    assert not result['dead']
    assert result['rounds'][0][0]['maximum_send'] >= 1


def test_nonfinite_response_is_uncertain() -> None:
    result = prove_multilanding(buried_board(), (1, 2, 3, 4), 171, 1, 0.,
                               lambda *args: float('nan'))
    assert not result['dead'] and result['reason'] == 'nonfinite_response'


def test_existing_projection_defaults_off() -> None:
    assert not ExchangeLandingProjection().multi_landing_death


@pytest.mark.parametrize('reason', ['uncertain_completion', 'unknown_budget', 'unverified_attack',
                                  'unknown_board', 'single_landing'])
def test_uncertain_inputs_do_not_start_proof(reason: str, monkeypatch: pytest.MonkeyPatch) -> None:
    boards = (Board(), buried_board())
    if reason == 'unknown_board':
        boards[1]._grid[0, 0] = COLOR_UNKNOWN
    projection = SimpleNamespace(
        _receivers=lambda *args: (boards, [0, 0]),
        _death_boards=lambda *args: (boards, boards, [True, reason != 'uncertain_completion']),
        _known_budget=lambda *args: reason != 'unknown_budget',
        _verified_attack=lambda *args: reason != 'unverified_attack')
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail('不確実な入力で証明を呼んだ')
    monkeypatch.setattr('src.exchange_event_multilanding.prove_multilanding', unexpected)
    latest = tuple(SimpleNamespace(board=b, queue=np.ones(4, dtype=int)) for b in boards)
    result = evaluate_multilanding(projection, SimpleNamespace(tracker=None), latest,
        [0, 30 if reason == 'single_landing' else 171], (1, 1), 0., dict(dead_sides=[]))
    assert result['dead_sides'] == []
    assert result['multi_landing'][1]['reason'] == reason


def test_exact_all_pairs_when_next_missing() -> None:
    board = Board()
    board._grid[-1, :3] = 5
    result = prove_multilanding(board, (), 1, 1, 0., no_response)
    assert not result['dead'] and result['rounds'][0][0]['maximum_send'] >= 1


def test_shorter_reply_not_lost_with_multiple_hand_budget() -> None:
    board = Board()
    board._grid[-1, :3] = 1
    result = prove_multilanding(board, (1, 1, 2, 2), 1, 2, 0., no_response)
    assert not result['dead']
    assert result['rounds'][0][0]['maximum_send'] >= 1


def test_remainder_landing_has_a_surviving_column() -> None:
    board = Board()
    board._grid[-11:, :] = 9
    result = prove_multilanding(board, (1, 2, 3, 4), 1, 1, 0., no_response)
    assert not result['dead'] and result['reason'] == 'surviving_response'


def test_fractional_score_is_credited_optimistically() -> None:
    board = Board()
    board._grid[-1, :3] = 1
    replies = Search(no_response, 0.).responses(board, (1, 2), 1)
    assert max(sent for _, sent in replies) == 1  # 4個消し40点も1個として上方丸め。


def test_cache_tracks_board_queue_amount_hands_rate_credit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    def proof(*args: object) -> dict:
        calls.append(args)
        return dict(dead=False)
    monkeypatch.setattr('src.exchange_event_multilanding.prove_multilanding', proof)
    projection = ExchangeLandingProjection(multi_landing_death=True)
    board, queue = buried_board(), (1, 2, 3, 4)
    cached_proof(projection, board, queue, 171, 1, 0., 0)
    cached_proof(projection, board, queue, 171, 1, 40., 0)
    assert len(calls) == 1
    for incoming, hands, elapsed, credit in [(172, 1, 0., 0), (171, 2, 0., 0),
                                           (171, 1, 100., 0), (171, 1, 0., 1)]:
        cached_proof(projection, board, queue, incoming, hands, elapsed, credit)
    cached_proof(projection, board, (2, 1, 3, 4), 171, 1, 0., 0)
    board._grid[-6, 0] = 1
    cached_proof(projection, board, queue, 171, 1, 0., 0)
    assert len(calls) == 7
