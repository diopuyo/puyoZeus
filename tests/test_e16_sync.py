"""手番の前後が混ざったNEXTと盤面をcountへ公開しない。"""
from __future__ import annotations

import numpy as np
from src.exchange_event_sync import CountTurnSynchronizer
from src.board import BOARD_ROWS, BOARD_COLS

OLD_QUEUE = np.array([5, 4, 4, 3], np.int8)
NEW_QUEUE = np.array([4, 3, 5, 5], np.int8)


def test_queue_first_waits_for_matching_placement() -> None:
    sync = CountTurnSynchronizer()
    before = np.zeros((BOARD_ROWS, BOARD_COLS), np.int8)
    assert sync.observe(before, OLD_QUEUE, 0., True)
    assert not sync.observe(before, np.zeros_like(OLD_QUEUE), 1., True, slide=True)
    assert not sync.observe(before, NEW_QUEUE, 2., True)
    assert sync.current_pair == (5, 4)
    np.testing.assert_array_equal(sync.accepted.queue, OLD_QUEUE)
    after = before.copy()
    after[-1, :2] = OLD_QUEUE[:2]
    assert sync.observe(after, NEW_QUEUE, 3., True)
    np.testing.assert_array_equal(sync.accepted.grid, after)
    assert sync.accepted.t_sec == 3.


def test_board_first_waits_for_queue_and_rejects_partial_piece() -> None:
    sync = CountTurnSynchronizer()
    board = np.zeros((BOARD_ROWS, BOARD_COLS), np.int8)
    sync.observe(board, OLD_QUEUE, 0., True)
    board[-1, 0] = OLD_QUEUE[0]
    assert not sync.observe(board, NEW_QUEUE, 1., True)
    board[-1, 1] = OLD_QUEUE[1]
    assert not sync.observe(board, OLD_QUEUE, 1.5, True, slide=True)
    assert not sync.observe(board, OLD_QUEUE, 2., True)
    assert sync.observe(board, NEW_QUEUE, 3., True)


def test_garbage_and_chain_completion_do_not_need_next_advance() -> None:
    sync = CountTurnSynchronizer()
    board = np.zeros((BOARD_ROWS, BOARD_COLS), np.int8)
    board[-1, :4] = 1
    sync.observe(board, OLD_QUEUE, 0., True)
    board[-1, -1] = 9
    assert sync.observe(board, OLD_QUEUE, 1., True)
    assert not sync.observe(board, OLD_QUEUE, 2., False, chaining=True)
    board[-1, :4] = 0
    assert sync.observe(board, OLD_QUEUE, 3., True)
    assert sync.reason == "observed_chain_completion"


def test_reacquire_after_missing_turn_requires_new_matching_pair() -> None:
    sync = CountTurnSynchronizer()
    board = np.zeros((BOARD_ROWS, BOARD_COLS), np.int8)
    sync.observe(board, OLD_QUEUE, 0., True)
    board[-1, :4] = (5, 4, 4, 3)
    missed_queue = np.array([2, 3, 1, 5], np.int8)
    assert not sync.observe(board, missed_queue, 2., True)
    assert sync.accepted.t_sec == 0.
    # 次の手番を観測できれば、古い採用盤面からの総差分で凍結し続けない。
    advanced = np.array([1, 5, 4, 2], np.int8)
    assert not sync.observe(board, advanced, 3., True)
    board[-2, :2] = missed_queue[:2]
    assert sync.observe(board, advanced, 4., True)
    assert sync.reason == "reacquired_matching_placement"
    assert sync.accepted.t_sec == 4.
