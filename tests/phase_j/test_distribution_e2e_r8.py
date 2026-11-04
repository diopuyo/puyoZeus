"""仮想カメラの停止・再開と、画像検査の例外後の継続を実機なしで検査する。"""
from pathlib import Path
import json
import time
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from src.phase_j import distribution_input as inputs
from src.phase_j import live_device as device
from src.phase_j import live_screen_errors as errors
from src.phase_j.live_screen_worker import AsyncScreenState

SIZE = (1920, 1080)
STOP_AT, RESUME_AT, END_AT = 1., 21., 24.
STEP_SEC, MAX_RECOVERY_SEC = .1, 2.
BAD_IMAGES = (None, np.empty((0, 0, 3), np.uint8), np.zeros((54, 96), np.uint8))


class Capture:
    def __init__(self, clock: list[float], stopped: Any) -> None:
        self.clock, self.stopped, self.released = clock, stopped, False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, Any]:
        if STOP_AT <= self.clock[0] < RESUME_AT:
            return True, self.stopped
        return True, np.full((1080, 1920, 3), int(self.clock[0]*30) % 100 + 120, np.uint8)

    def set(self, key: int, value: float) -> bool:
        return True

    def release(self) -> None:
        self.released = True


def make_source(monkeypatch: pytest.MonkeyPatch, stopped: Any, mode: str) -> tuple:
    monkeypatch.setenv('PUYO_LIVE_VERIFY_ASYNC', str(int(mode == 'async')))
    monkeypatch.setenv('PUYO_LIVE_ASYNC_SCREEN_VERIFY', str(int(mode == 'initial')))
    clock, statuses = [0.], []
    capture = Capture(clock, stopped)

    def sleep(seconds: float) -> None:
        clock[0] += max(STEP_SEC, seconds)
        time.sleep(0)  # 模擬時刻でも worker が動く機会を与える。

    source = device.DirectShowSource(device.DeviceConfig('OBS', 0), END_AT, lambda: None,
        lambda *_: capture, lambda _: not (STOP_AT <= clock[0] < RESUME_AT),
        lambda: clock[0], sleep, lambda phase: statuses.append((clock[0], phase)))
    source._buffered_capture, source._reconnector = False, None
    return source, capture, statuses


@pytest.mark.parametrize('mode', ['sync', 'initial', 'async'])
@pytest.mark.parametrize('kind', ['none', 'empty', 'gray', 'stopped'])
def test_stop_then_resume(monkeypatch: pytest.MonkeyPatch, mode: str, kind: str) -> None:
    stopped = dict(zip(('none', 'empty', 'gray'), BAD_IMAGES)).get(kind)
    if kind == 'stopped':
        stopped = cv2.resize(inputs.stopped_reference(), SIZE)
    source, capture, statuses = make_source(monkeypatch, stopped, mode)
    frames = list(source)
    holds = [phase for stamp, phase in statuses if STOP_AT + MAX_RECOVERY_SEC <= stamp < RESUME_AT]
    assert holds and set(holds) == {'verifying'}
    resumed = [frame.captured_at for frame in frames if frame.captured_at >= RESUME_AT]
    assert resumed and resumed[0] - RESUME_AT <= MAX_RECOVERY_SEC
    assert not any(STOP_AT + MAX_RECOVERY_SEC <= f.captured_at < RESUME_AT for f in frames)
    assert capture.released


@pytest.mark.parametrize('image', BAD_IMAGES)
def test_invalid_image_is_not_sent_to_opencv(image: Any) -> None:
    assert not inputs.has_feedback(image)
    assert not inputs.is_obs_stopped(image)


def fail_once(result: bool = True) -> Any:
    attempts = []

    def check(*args: Any) -> bool:
        attempts.append(True)
        if len(attempts) == 1:
            raise ValueError('検査の故障注入')
        return result

    return check


@pytest.mark.parametrize('wait_initial', [False, True])
def test_worker_logs_and_recovers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 wait_initial: bool) -> None:
    monkeypatch.setenv(errors.OUTPUT_ENV, str(tmp_path))
    state = AsyncScreenState(fail_once(), lambda passed, *_: (passed, 0), .1, wait_initial)
    image = np.zeros((54, 96, 3), np.uint8)
    try:
        assert not state.update(image, (96, 54), 0., True)[0]
        if not wait_initial:
            state.worker.future.result(timeout=2)
            assert not state.update(image, (96, 54), .01, True)[0]
        state.update(image, (96, 54), 1., True)
        if not wait_initial:
            state.worker.future.result(timeout=2)
        assert state.update(image, (96, 54), 1.01, True)[0]
    finally:
        state.close()
    rows = [json.loads(line) for line in (tmp_path / errors.LOG_FILE).read_text('utf-8').splitlines()]
    assert len(rows) == 1 and rows[0]['stage'] == 'screen_verify_worker'
    assert 'ValueError: 検査の故障注入' in rows[0]['traceback']


def test_unready_does_not_call_stopped_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: Any) -> bool:
        raise AssertionError('ready でない画像を検査してはいけない')

    monkeypatch.setattr(inputs, 'is_obs_stopped', unexpected)
    source, _, statuses = make_source(monkeypatch, None, 'sync')
    source._screen_hold(False, None, SimpleNamespace(diagnostics={}), 0, SimpleNamespace())
    assert statuses[-1][1] == 'verifying'


@pytest.mark.parametrize('stage', ['feedback', 'obs_stopped', 'screen_verify'])
def test_source_logs_and_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    monkeypatch.setenv(errors.OUTPUT_ENV, str(tmp_path))
    source, capture, _ = make_source(monkeypatch, None, 'sync')
    if stage == 'feedback':
        monkeypatch.setattr(inputs, 'has_feedback', fail_once(False))
    elif stage == 'obs_stopped':
        monkeypatch.setattr(inputs, 'is_obs_stopped', fail_once(False))
        source.verifier = lambda _: source.clock() >= STEP_SEC * 2
    else:
        source.verifier = fail_once()
    frames = list(source)
    assert any(frame.captured_at >= RESUME_AT for frame in frames)
    assert capture.released and source.clock() >= END_AT
    rows = [json.loads(line) for line in (tmp_path / errors.LOG_FILE).read_text('utf-8').splitlines()]
    assert len(rows) == 1 and rows[0]['stage'] == stage
    assert 'ValueError: 検査の故障注入' in rows[0]['traceback']


def test_output_configuration_is_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(errors.OUTPUT_ENV, 'old')
    errors.configure_output(tmp_path)
    import os
    assert os.environ[errors.OUTPUT_ENV] == str(tmp_path.resolve())
    errors.configure_output(None)
    assert errors.OUTPUT_ENV not in os.environ
