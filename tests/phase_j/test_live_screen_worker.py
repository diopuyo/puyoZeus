"""検証の遅延・入力断・例外で認識所有者と競合しないことを検証する。"""
from concurrent.futures import Future
from threading import Event, get_ident
from typing import Any

import numpy as np
import pytest

from src.phase_j.live_screen_worker import AsyncScreenState, ScreenResult, ScreenWorker
from tests.phase_j.test_live_fast_first import session

IMAGE = np.zeros((2, 2, 3), dtype=np.uint8)
SIZE = (2, 2)


def judge(passed: bool, verified: bool, streak: int) -> tuple[bool, int]:
    return (True, 0) if passed else (verified and streak+1 < 2, streak+1)


def completed(state: AsyncScreenState, passed: bool, feedback: bool = False) -> None:
    state.worker.future = Future()
    state.worker.pending_generation = state.worker.generation
    state.worker.future.set_result(ScreenResult(state.worker.generation, 0., passed, feedback, .08))


def test_slow_worker_never_blocks_and_owns_snapshot() -> None:
    entered, release = Event(), Event()
    calls: list[Any] = []
    def verifier(image: Any) -> bool:
        entered.set()
        assert release.wait(2)
        calls.append((get_ident(), image.copy()))
        return True
    worker = ScreenWorker(verifier)
    image = IMAGE.copy()
    try:
        assert worker.submit(image, 0.)
        assert entered.wait(2)
        image[:] = 255
        assert worker.poll() is None
        assert not worker.submit(image, 1.)
        release.set()
        worker.future.result(timeout=2)
        assert worker.poll().passed
        assert calls[0][0] != get_ident()
        assert np.array_equal(calls[0][1], IMAGE)
    finally:
        release.set()
        worker.close()


def test_initial_hold_continuation_and_failure_debounce() -> None:
    state = AsyncScreenState(lambda _: True, judge, 1.)
    try:
        state.last_verify = 0.
        assert state.update(IMAGE, SIZE, .1, True) == (False, 0)
        completed(state, True)
        assert state.update(IMAGE, SIZE, .2, True) == (True, 0)
        state.worker.future = Future()
        assert state.update(IMAGE, SIZE, .3, True) == (True, 0)
        completed(state, False)
        assert state.update(IMAGE, SIZE, .4, True) == (True, 1)
        completed(state, False, True)
        assert state.update(IMAGE, SIZE, .5, True) == (False, 2)
        assert state.feedback
    finally:
        state.close()


def test_first_image_is_not_dropped_while_verifying() -> None:
    state = AsyncScreenState(lambda image: np.array_equal(image, IMAGE), judge, 1.)
    try:
        assert state.update(IMAGE, SIZE, 0., True) == (True, 0)
        assert state.latest_result is not None and state.latest_result.captured_at == 0.
        assert state.worker.future is None
    finally:
        state.close()


@pytest.mark.parametrize('transport,size', [(False, SIZE), (True, (4, 4))])
def test_stale_completion_cannot_restore_new_input(transport: bool, size: tuple) -> None:
    state = AsyncScreenState(lambda _: True, judge, 1.)
    try:
        state.size, state.verified = SIZE, True
        completed(state, True)
        assert state.update(IMAGE, size, .1, transport) == (False, 0)
    finally:
        state.close()


def test_exceptions_propagate_only_for_current_generation() -> None:
    worker = ScreenWorker(lambda _: True)
    try:
        worker.future = Future()
        worker.future.set_exception(ValueError('検証失敗'))
        with pytest.raises(ValueError, match='検証失敗'):
            worker.poll()
        worker.future = Future()
        worker.future.set_exception(ValueError('古い入力'))
        worker.invalidate()
        assert worker.poll() is None
    finally:
        worker.close()


def test_async_failure_keeps_fast_first_epoch_and_recheck(session: Any) -> None:
    """遅延検証の失敗も既存の保留経路を通し、世代は一度だけ更新する。"""
    state = AsyncScreenState(lambda _: True, judge, 1.)
    warmup = session.warmup
    try:
        state.last_verify, state.verified, state.size = 0., True, SIZE
        completed(state, False)
        assert state.update(IMAGE, SIZE, .1, True)[0]
        assert session.epoch == 0
        completed(state, False)
        assert not state.update(IMAGE, SIZE, .2, True)[0]
        for _ in range(3):
            session.hold('no_puyo_screen')
        assert session.epoch == 1 and session.warmup is warmup and warmup.ready
        assert session._needs_recheck and not session.publishing_ready
        completed(state, True)
        assert state.update(IMAGE, SIZE, .3, True)[0]
        session.update(3, .3, None)
        assert session.publishing_ready and session.epoch == 1
        assert session._recheck is not None
    finally:
        state.close()
