"""遅い検査・例外・切断でも読み取りを塞がないことを確認する。"""
from __future__ import annotations

import threading
import time
from typing import Any
import numpy as np
import pytest
from src.phase_j.live_feedback_worker import FeedbackWorker
from src.phase_j.live_device import DeviceConfig, DirectShowSource
from tests.phase_j.test_feedback_capture import (
    Camera, SIZE, encoded_camera, marked_image, release_environment)
from src.phase_j.live_publish import input_hold

TIMEOUT_SEC = 3.0


def test_screen_verification_can_skip_duplicate_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import distribution_input
    from src.phase_j.live_device import PuyoScreenVerifier
    from unittest.mock import Mock
    detector = Mock(return_value=True)
    monkeypatch.setattr(distribution_input, 'has_feedback', detector)
    verifier = object.__new__(PuyoScreenVerifier)
    verifier.regions = ()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    assert not verifier(image)
    verifier.check_feedback = False
    assert verifier(image)
    assert detector.call_count == 1


def test_end_hold_uses_supplied_feedback_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import distribution_input
    from src.phase_j.live_device import PuyoScreenVerifier
    from src.phase_j.live_match_end_feed import eligible
    from unittest.mock import Mock
    detector = Mock(side_effect=AssertionError('読み取り側で検査しない'))
    monkeypatch.setattr(distribution_input, 'has_feedback', detector)
    verifier = object.__new__(PuyoScreenVerifier)
    verifier.regions = ()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    assert eligible(image, verifier, feedback_present=False)
    assert not eligible(image, verifier, feedback_present=True)
    detector.assert_not_called()


def wait_until(predicate: Any) -> None:
    deadline = time.perf_counter() + TIMEOUT_SEC
    while not predicate() and time.perf_counter() < deadline:
        time.sleep(.01)
    assert predicate()


def test_latest_slot_exception_and_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    entered, resume = threading.Event(), threading.Event()
    seen: list[int] = []
    def detect(image: Any, corrected: Any, lazy: bool) -> bool:
        seen.append(int(image[0, 0, 0]))
        if len(seen) == 1:
            entered.set()
            assert resume.wait(TIMEOUT_SEC)
            raise RuntimeError('synthetic failure')
        return True
    monkeypatch.setattr(FeedbackWorker, '_detect', staticmethod(detect))
    worker = FeedbackWorker()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    try:
        worker.offer(image, 0)
        assert entered.wait(TIMEOUT_SEC)
        for stamp in (1, 2, 3):
            image[:] = stamp
            worker.offer(image, stamp)
        image[:] = 99
        resume.set()
        wait_until(lambda: worker.result(3)[0])
        assert seen == [0, 3]
        assert not worker.result(7)[0]
    finally:
        resume.set()
        worker.close()
        worker.thread.join(TIMEOUT_SEC)
    assert worker.thread.daemon and not worker.thread.is_alive()
    assert worker.result(3)[0] is False


def test_read_does_not_wait_for_detector_and_disconnect_discards_result(monkeypatch: pytest.MonkeyPatch) -> None:
    release_environment(monkeypatch, False, True)
    entered, resume = threading.Event(), threading.Event()
    def detect(*args: Any) -> bool:
        entered.set()
        assert resume.wait(TIMEOUT_SEC)
        return True
    monkeypatch.setattr(FeedbackWorker, '_detect', staticmethod(detect))
    source = DirectShowSource(DeviceConfig('synthetic', 0), 0, lambda: None)
    camera = Camera(np.zeros(SIZE, dtype=np.uint8), lifecycle=False)
    source._read(camera)
    worker = source._feedback_worker
    try:
        assert entered.wait(TIMEOUT_SEC)
        start = time.perf_counter()
        for _ in range(5):
            source._read(camera)
        assert time.perf_counter() - start < .5
        camera.released = True
        source._read(camera)
        assert source._feedback_worker is None and not source._feedback_present
    finally:
        resume.set()
        source._close_feedback()
        worker.thread.join(TIMEOUT_SEC)
    assert not worker.result(source.clock())[0] and not worker.thread.is_alive()


@pytest.mark.parametrize('buffered', [False, True])
@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('transport', [False, True])
def test_synthetic_entrance_to_warning(monkeypatch: pytest.MonkeyPatch, buffered: bool,
                                     asynchronous: bool, transport: bool) -> None:
    release_environment(monkeypatch, buffered, asynchronous)
    image = marked_image()
    camera = Camera(encoded_camera(image) if transport else image, lifecycle=False)
    warnings: list[float] = []
    def publish(phase: str) -> None:
        payload = dict(display={})
        input_hold(payload, dict(input_calibration=dict(phase=phase, progress=0), t_sec=0), 0, None)
        if payload['display']['input_status'] == 'overlay_feedback':
            warnings.append(time.perf_counter() - camera.start)
    source = DirectShowSource(DeviceConfig('synthetic', 0, color_correction='601to709'),
        TIMEOUT_SEC, lambda: None, lambda *args: camera, verifier=lambda image: True, on_status=publish)
    list(source)
    assert warnings and warnings[0] <= TIMEOUT_SEC
    assert camera.released and source._feedback_worker is None
