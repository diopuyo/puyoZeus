"""盤面境界・組別欠測・未来参照を防ぐ5B読みの回帰。"""
import numpy as np

from src.prefire_v5b_queue import PairReading, SideQueueV5B, StableQueuesV5B
from types import SimpleNamespace


def feed(queue: SideQueueV5B, board: bytes, colors: tuple, start: int = 0) -> None:
    """同じ画面読みを3回与える。"""
    for frame in range(start, start + 3):
        queue.observe(frame / 30, board, np.array(colors))


def test_pair_requires_three_frames() -> None:
    reading = PairReading()
    for _ in range(2):
        reading.observe((1, 2))
        assert reading.accepted is None
    reading.observe((1, 2))
    assert reading.accepted == (1, 2)


def test_flicker_retains_observed_pair_until_confirmation() -> None:
    reading = PairReading()
    for _ in range(3):
        reading.observe((1, 2))
    reading.observe((2, 3))
    assert reading.accepted == (1, 2)
    reading.observe((2, 3))
    reading.observe((2, 3))
    assert reading.accepted == (2, 3)


def test_unknown_second_does_not_discard_first() -> None:
    queue = SideQueueV5B()
    feed(queue, b'a', (1, 2, 0, 0))
    queue.observe(.1, b'b', np.array((1, 2, 0, 0)))
    assert queue.hand == (1, 2)
    assert queue.pairs[0].accepted == (1, 2)
    assert queue.known() == (1, 2, 0, 0, 0, 0)


def test_continuous_reading_survives_board_boundary() -> None:
    queue = SideQueueV5B()
    feed(queue, b'a', (1, 2, 3, 4))
    queue.observe(.1, b'b', np.array((1, 2, 3, 4)))
    assert queue.known() == (1, 2, 3, 4, 0, 0)


def test_observed_promotion_exposes_three_known_pairs() -> None:
    queue = SideQueueV5B()
    feed(queue, b'a', (1, 2, 3, 4))
    feed(queue, b'b', (1, 2, 3, 4), 3)
    feed(queue, b'b', (3, 4, 2, 5), 6)
    assert queue.known() == (1, 2, 3, 4, 2, 5)


def test_elapsed_time_does_not_invent_promotion() -> None:
    queue = SideQueueV5B()
    feed(queue, b'a', (1, 2, 1, 2))
    feed(queue, b'b', (1, 2, 1, 2), 300)
    assert queue.known() == (1, 2, 1, 2, 0, 0)


def test_game_reset_discards_hand() -> None:
    queues = StableQueuesV5B()
    queues.sides[0].hand = (1, 2)
    queues.update(SimpleNamespace(_game=2, _history=[[], []]))
    assert queues.sides[0].hand is None


def test_no_observation_does_not_invent_hand() -> None:
    queue = SideQueueV5B()
    feed(queue, b'a', (1, 2, 3, 4))
    assert queue.known() == (0, 0, 0, 0, 0, 0)


def test_next_hand_uses_last_observed_next() -> None:
    queue = SideQueueV5B()
    feed(queue, b'a', (1, 2, 3, 4))
    feed(queue, b'b', (1, 2, 3, 4), 3)
    queue.observe(.2, b'c', np.array((3, 4, 2, 5)))
    assert queue.hand == (1, 2)


def test_unread_new_interval_does_not_inherit_old_next() -> None:
    queue = SideQueueV5B()
    feed(queue, b'a', (1, 2, 3, 4))
    feed(queue, b'b', (0, 0, 0, 0), 3)
    assert queue.known() == (1, 2, 0, 0, 0, 0)
    feed(queue, b'c', (0, 0, 0, 0), 6)
    assert queue.known() == (0, 0, 0, 0, 0, 0)
