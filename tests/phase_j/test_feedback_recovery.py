"""異常入力・検査の例外・形式変更後も読み取りと検査が復帰する。"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import pytest

from src.phase_j import distribution_input as inputs
from src.phase_j import live_feedback_worker as module
from src.phase_j.live_device import DeviceConfig, DirectShowSource
from tests.phase_j.test_feedback_capture import Camera, SIZE, release_environment
from tests.phase_j.test_live_feedback_worker import wait_until, TIMEOUT_SEC

INVALID = [None, np.empty((0, 0, 3), np.uint8), np.zeros((2, 2), np.uint8),
           np.zeros((2, 2, 1), np.uint8), np.zeros((2, 2, 4), np.uint8)]
SMALL_SIZE = (2, 2, 3)
SHORT_PERIOD = .01
RUN_SEC = 7.0
BAD_START, SHAPE_START, STOP_START, RESTORE_START = .7, 1.7, 2.0, 4.0


@pytest.mark.parametrize('image', INVALID)
def test_invalid_input_clears_pending_and_inflight(monkeypatch: pytest.MonkeyPatch, image: Any) -> None:
    entered, resume = threading.Event(), threading.Event()
    def detect(*args: Any) -> bool:
        entered.set()
        assert resume.wait(TIMEOUT_SEC)
        return True
    monkeypatch.setattr(module.FeedbackWorker, '_detect', staticmethod(detect))
    monkeypatch.setattr(module, 'PERIOD_SEC', SHORT_PERIOD)
    worker = module.FeedbackWorker()
    try:
        worker.offer(np.zeros(SMALL_SIZE, np.uint8), 1)
        assert entered.wait(TIMEOUT_SEC)
        worker.offer(np.zeros(SMALL_SIZE, np.uint8), 2)
        worker.offer(image, 3)
        assert worker._pending is None and worker.result(3) == (False, float('-inf'))
        resume.set()
        time.sleep(SHORT_PERIOD * 3)
        assert worker.result(3) == (False, float('-inf')) and worker.thread.is_alive()
        worker.offer(np.zeros(SMALL_SIZE, np.uint8), 4)
        wait_until(lambda: worker.result(4)[0])
    finally:
        resume.set()
        worker.close()
        worker.thread.join(TIMEOUT_SEC)


class BrokenCopy(np.ndarray):
    def copy(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('synthetic copy failure')


def test_copy_exception_is_recorded_and_worker_recovers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('PUYO_LIVE_SCREEN_ERROR_OUTPUT', str(tmp_path))
    monkeypatch.setattr(module.FeedbackWorker, '_detect', staticmethod(lambda *args: True))
    worker = module.FeedbackWorker()
    try:
        worker.offer(np.zeros(SMALL_SIZE, np.uint8).view(BrokenCopy), 1)
        assert worker.result(1) == (False, float('-inf')) and worker._pending is None
        assert 'feedback_offer' in (tmp_path/'screen_errors.jsonl').read_text('utf-8')
        worker.offer(np.zeros(SMALL_SIZE, np.uint8), 2)
        wait_until(lambda: worker.result(2)[0])
        assert worker.thread.is_alive()
    finally:
        worker.close()
        worker.thread.join(TIMEOUT_SEC)


@pytest.mark.parametrize('method', ['start', 'offer', 'result', 'close'])
def test_reader_contains_feedback_exceptions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, method: str) -> None:
    monkeypatch.setenv('PUYO_LIVE_SCREEN_ERROR_OUTPUT', str(tmp_path))
    source = DirectShowSource(DeviceConfig('synthetic', 0), 0, lambda: None)
    def fail(*args: Any) -> Any:
        raise RuntimeError('synthetic '+method)
    target = module.FeedbackWorker
    with monkeypatch.context() as patch:
        patch.setattr(module if method == 'start' else target, 'FeedbackWorker' if method == 'start' else method, fail)
        source._check_feedback(np.zeros(SMALL_SIZE, np.uint8), 1)
        worker = source._feedback_worker
        if method == 'result':
            assert not source._hold_input(True, True, np.zeros(SMALL_SIZE, np.uint8), 1, None, 0, None)
        source._close_feedback()
        assert not source._feedback_present
    if worker is not None:
        worker.close()
        worker.thread.join(TIMEOUT_SEC)
    assert ('feedback_'+('offer' if method == 'start' else method)) in (tmp_path/'screen_errors.jsonl').read_text('utf-8')


class SequenceCamera(Camera):
    def read(self) -> tuple[bool, Any]:
        elapsed = time.perf_counter()-self.start
        self.count += 1
        if BAD_START <= elapsed < SHAPE_START:
            index = min(int((elapsed-BAD_START) / ((SHAPE_START-BAD_START)/len(INVALID))), len(INVALID)-1)
            return True, INVALID[index]
        if SHAPE_START <= elapsed < STOP_START:
            return True, np.zeros((90, 160, 3), np.uint8)
        if STOP_START <= elapsed < RESTORE_START:
            return True, cv2.resize(inputs.stopped_reference(), SIZE[1::-1])
        image = np.full(SIZE, 32, dtype=np.uint8)
        image[:8, :8] = self.count % 255
        return True, image


@pytest.mark.parametrize('asynchronous', [False, True])
def test_sequence_recovers_without_stalling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, asynchronous: bool) -> None:
    release_environment(monkeypatch, False, asynchronous)
    monkeypatch.setattr(module, 'PERIOD_SEC', SHORT_PERIOD)
    camera = SequenceCamera(np.zeros(SIZE, np.uint8), lifecycle=False)
    statuses: list[tuple[float, str]] = []
    diagnostics: list[tuple[float, dict]] = []
    workers: list[Any] = []
    worker_alive: list[bool] = []
    def status(phase: str) -> None:
        statuses.append((time.perf_counter()-camera.start, phase))
        diagnostics.append((statuses[-1][0], dict(source.input_diagnostics)))
    def verify(image: Any) -> bool:
        return not inputs.is_obs_stopped(image)
    source = DirectShowSource(DeviceConfig('synthetic', 0), RUN_SEC, lambda: None,
        lambda *args: camera, verifier=verify, on_status=status)
    frames = []
    for frame in source:
        frames.append(frame.media_sec)
        if source._feedback_worker is not None:
            workers.append(source._feedback_worker)
            worker_alive.append(source._feedback_worker.thread.is_alive())
    wait_until(lambda: all(not worker.thread.is_alive() for worker in workers))
    report = dict(statuses=statuses, diagnostics=diagnostics, frames=frames,
        worker_count=len({id(w) for w in workers}), reads=camera.count,
        alive_during_reads=all(worker_alive), closed_after_run=source._feedback_worker is None,
        restored_frames=sum(stamp >= RESTORE_START for stamp in frames))
    (tmp_path/'sequence.json').write_text(json.dumps(report), 'utf-8')
    assert any(stamp < BAD_START for stamp in frames)
    assert any(stamp >= RESTORE_START for stamp in frames)
    # 停止画像は継続性の再確認 (verifying) が先に立つことがある。札はプレビューから決まる (r9)。
    assert any(STOP_START <= stamp < RESTORE_START for stamp, _ in statuses)
    assert camera.released and source._feedback_worker is None
    assert all(worker_alive)
    assert workers and all(not worker.thread.is_alive() for worker in workers)
