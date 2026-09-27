"""非同期探索の世代・期限・直前値保持を検証する。"""
from concurrent.futures import Future
import numpy as np
import pytest
from src.board import Board
from src.phase_j.live_counter import AsyncCounter


class Executor:
    def __init__(self) -> None:
        self.jobs: list[tuple] = []

    def submit(self, function: object, arguments: dict | None = None) -> Future:
        future = Future()
        future.set_running_or_notify_cancel()
        self.jobs.append((future, arguments))
        return future


def board(color: int = 0) -> Board:
    grid = np.zeros((13, 6), dtype=np.int8)
    grid[-1, 0] = color
    return Board.from_list(grid.tolist())


def complete(executor: Executor) -> None:
    future, arguments = executor.jobs[0]
    future.set_result((arguments['generation'], (20, 0.8, 0.2), 2))


def test_result_arrives_on_next_update() -> None:
    executor = Executor()
    tracker = AsyncCounter(executor)
    tracker.update(board(), board(), 3, t_sec=1)
    assert tracker.pending and len(executor.jobs) == 1
    complete(executor)
    assert tracker.update(board(), board(), 2, t_sec=2) == (20, 0.8, 0.2)
    assert not tracker.pending and tracker.accepted == 1


def test_changed_board_discards_completed_result() -> None:
    executor = Executor()
    tracker = AsyncCounter(executor)
    tracker.update(board(), board(), 3, t_sec=1)
    complete(executor)
    tracker.update(board(1), board(), 2, t_sec=2)
    assert tracker.discarded == 1 and tracker.accepted == 0
    assert tracker.pending and tracker._last_result is None
    assert len(executor.jobs) == 2


def test_pending_keeps_previous_accepted_value_and_generation() -> None:
    executor = Executor()
    tracker = AsyncCounter(executor)
    tracker.update(board(), board(), 3, t_sec=1)
    complete(executor)
    tracker.update(board(), board(), 2, t_sec=2)
    assert tracker.update(board(1), board(), 2, t_sec=2) == (20, 0.8, 0.2)
    assert tracker.pending and tracker.result_generation != tracker.generation


def test_nonstable_change_and_aba_invalidate_job() -> None:
    executor = Executor()
    tracker = AsyncCounter(executor)
    tracker.invalidate((1, b'a'))
    tracker.update(board(), board(), 3, t_sec=1)
    tracker.invalidate((1, b'b'))
    tracker.invalidate((1, b'a'))
    complete(executor)
    tracker.update(board(), board(), 2, t_sec=2)
    assert tracker.discarded == 1 and tracker.accepted == 0


def test_expired_result_is_discarded() -> None:
    executor = Executor()
    tracker = AsyncCounter(executor)
    tracker.update(board(), board(), 1, t_sec=1)
    complete(executor)
    tracker.update(board(), board(), 1, t_sec=3)
    assert tracker.discarded == 1 and tracker.pending


def test_only_one_inflight_and_input_is_copied() -> None:
    executor = Executor()
    tracker = AsyncCounter(executor)
    original = board()
    tracker.update(original, board(), 3, t_sec=1)
    original._grid[-1, 0] = 1
    for _ in range(10):
        tracker.update(original, board(), 3, t_sec=2)
    assert len(executor.jobs) == 1
    assert executor.jobs[0][1]['b1']._grid[-1, 0] == 0


def test_zero_budget_invalidates_and_clears_previous() -> None:
    executor = Executor()
    tracker = AsyncCounter(executor)
    tracker.update(board(), board(), 3, t_sec=1)
    complete(executor)
    result = tracker.update(board(), board(), 0, t_sec=2)
    assert np.isnan(result[1]) and not tracker.pending
    assert tracker.discarded == 1


def test_worker_failure_keeps_pending() -> None:
    executor = Executor()
    tracker = AsyncCounter(executor)
    tracker.update(board(), board(), 3, t_sec=1)
    executor.jobs[0][0].set_exception(RuntimeError('失敗'))
    tracker.update(board(), board(), 2, t_sec=2)
    assert tracker.pending and tracker.error == '失敗'


def test_result_generation_is_checked_independently() -> None:
    executor = Executor()
    tracker = AsyncCounter(executor)
    tracker.update(board(), board(), 3, t_sec=1)
    executor.jobs[0][0].set_result((tracker.generation-1, (20, 0.8, 0.2), 2))
    tracker.update(board(), board(), 2, t_sec=2)
    assert tracker.accepted == 0 and tracker.discarded == 1 and tracker.pending


@pytest.mark.parametrize('changed', [dict(next1=(1, 2)), dict(defender_side='2P'),
                                   dict(threshold_ojama=20)])
def test_scope_change_discards_result(changed: dict) -> None:
    executor = Executor()
    tracker = AsyncCounter(executor)
    tracker.update(board(), board(), 3, t_sec=1)
    complete(executor)
    tracker.update(board(), board(), 2, t_sec=2, **changed)
    assert tracker.discarded == 1 and tracker.accepted == 0


def test_same_board_new_game_discards_result() -> None:
    executor = Executor()
    tracker = AsyncCounter(executor)
    tracker.invalidate((1, True, b'board'))
    tracker.update(board(), board(), 3, t_sec=1)
    tracker.invalidate((2, True, b'board'))
    complete(executor)
    tracker.update(board(), board(), 2, t_sec=2)
    assert tracker.discarded == 1


def test_refresh_preserves_original_result_timestamp() -> None:
    executor = Executor()
    tracker = AsyncCounter(executor)
    tracker.update(board(), board(), 3, t_sec=1)
    complete(executor)
    tracker.update(board(), board(), 2, t_sec=1.1)
    tracker.update(board(), board(), 2, t_sec=1.6)
    assert tracker.pending and tracker.result_time == 1 and tracker.request_time == 1.6


def test_all_trackers_share_one_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from src.phase_j.live_counter import factory
    executor = Executor()
    monkeypatch.setattr('src.phase_j.live_counter.ProcessPoolExecutor', lambda **kwargs: executor)
    bridge = SimpleNamespace(counters=[])
    tracker_type = factory(bridge)
    first, second = tracker_type(), tracker_type()
    assert first.executor is second.executor
    assert first.owns_executor and not second.owns_executor


def test_publication_uses_current_tracker_not_last_created() -> None:
    from types import SimpleNamespace
    from scripts.run_live_pipeline_20260928 import ResultSink
    from src.phase_j.live_bridge import RecognitionBridge
    tracker = AsyncCounter(Executor())
    tracker.pending = True
    publisher = SimpleNamespace(offer=lambda row: None)
    sink = ResultSink(publisher)
    bridge = RecognitionBridge(False, sink.observe)
    sink.bridge = bridge
    bridge.counters = [tracker, AsyncCounter(Executor())]
    notice = SimpleNamespace(frame=1, t_sec=1.0, captured_at=1.0, recognized_at=1.1,
                             dropped_before=0, result_bytes=b'test', pipeline=None)
    side = SimpleNamespace(state=SimpleNamespace(name='STABLE'), score=0)
    overlay = SimpleNamespace(tracker=SimpleNamespace(source='G_fe', probability=0.5,
                                                      _static_probability=0.5))
    bridge.observe(notice, 0.5, 0.0, overlay, SimpleNamespace(p1=side, p2=side),
                   1, 0, (tracker,))
    assert sink.rows[0]['counter_search']['pending']
