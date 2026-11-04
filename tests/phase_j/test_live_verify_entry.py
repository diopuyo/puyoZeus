"""入口だけの再試行周期と既存の合否・設定配送を確認する。"""
from __future__ import annotations

import argparse
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from src.phase_j import live_verify_entry as entry, live_speed
from src.phase_j.live_device import DeviceConfig, DirectShowSource
from src.phase_j.live_screen_worker import AsyncScreenState
from src.phase_j.live_source import RECOGNITION_HZ
from tests.phase_j.test_live_device import FakeDevice

REGULAR_SEC = 1.0
WAIT_SEC = 2.0
TICK_SEC = 0.25
IMAGE = np.zeros((1080, 1920, 3), dtype=np.uint8)
SIZE = (1920, 1080)


def judge(passed: bool, verified: bool, streak: int) -> tuple[bool, int]:
    """製品と同じ K=1 の判定を使う。"""
    source = DirectShowSource(DeviceConfig('OBS', 0), REGULAR_SEC, lambda: None)
    return source._judge(passed, verified, streak)


@pytest.mark.parametrize('fast', [False, True])
@pytest.mark.parametrize('wait_initial', [False, True])
def test_async_entry_then_regular(fast: bool, wait_initial: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(entry.FLAG_ENV, str(int(fast)))
    calls: list[bool] = []
    def verifier(image: np.ndarray) -> bool:
        calls.append(True)
        return len(calls) >= 3
    state = AsyncScreenState(verifier, judge, REGULAR_SEC, wait_initial=wait_initial)
    submitted = []
    try:
        for tick in range(13):
            captured = tick*TICK_SEC
            previous = state.last_verify
            state.update(IMAGE, SIZE, captured, True)
            if state.last_verify != previous:
                submitted.append(captured)
            if state.worker.future is not None:
                state.worker.future.result(timeout=WAIT_SEC)
                state.update(IMAGE, SIZE, captured, True)
        assert submitted == ([0, .25, .5, 1.5, 2.5] if fast else [0, 1, 2, 3])
        assert state.verified and state.fail_streak == 0
    finally:
        state.close()


def source_trace(fast: bool, monkeypatch: pytest.MonkeyPatch) -> tuple:
    monkeypatch.setenv(entry.FLAG_ENV, str(int(fast)))
    monkeypatch.setenv('PUYO_LIVE_VERIFY_ASYNC', '0')
    monkeypatch.setenv('PUYO_LIVE_ASYNC_SCREEN_VERIFY', '0')
    monkeypatch.setattr('src.phase_j.live_device.FrameContinuity.ready', lambda *args: True)
    clock, calls, holds = [0.0], [], []
    def sleep(seconds: float) -> None:
        clock[0] = round(clock[0]+seconds, 9)
    def verifier(image: np.ndarray) -> bool:
        calls.append(clock[0])
        return len(calls) >= 3
    source = DirectShowSource(DeviceConfig('OBS', 0), 3.1, lambda: holds.append(clock[0]),
                              lambda *args: FakeDevice(IMAGE), verifier, lambda: clock[0], sleep)
    frames = [(frame.media_sec, frame.index) for frame in source]
    return calls, holds, frames


def test_sync_entry_then_regular(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, _, _ = source_trace(True, monkeypatch)
    assert calls[:3] == [0, .25, .5]
    assert len(calls) == 5
    assert all(REGULAR_SEC <= right-left <= REGULAR_SEC+1/RECOGNITION_HZ
               for left, right in zip(calls[2:], calls[3:]))


def test_off_trace_matches_legacy_period(monkeypatch: pytest.MonkeyPatch) -> None:
    actual = source_trace(False, monkeypatch)
    monkeypatch.setattr(entry, 'verification_period', lambda verified, regular, fast: regular)
    assert source_trace(False, monkeypatch) == actual


def test_pending_worker_never_queues_more_verifications(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(entry.FLAG_ENV, '1')
    entered, release = Event(), Event()
    calls = []
    def verifier(image: np.ndarray) -> bool:
        calls.append(True)
        entered.set()
        assert release.wait(WAIT_SEC)
        return False
    state = AsyncScreenState(verifier, judge, REGULAR_SEC, wait_initial=False)
    try:
        state.update(IMAGE, SIZE, 0, True)
        assert entered.wait(WAIT_SEC)
        pending = state.worker.future
        for captured in (.25, .5, .75):
            assert state.update(IMAGE, SIZE, captured, True) == (False, 0)
            assert state.worker.future is pending
        assert len(calls) == 1 and state.last_verify == 0
    finally:
        release.set()
        state.close()


@pytest.mark.parametrize('fail_limit,expected', [
    (1, [0, 1, 1.25, 1.5, 2.5, 3.5]), (2, [0, 1, 2, 2.25, 3.25]),
])
def test_retry_after_hold_preserves_failure_limit(
        fail_limit: int, expected: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(entry.FLAG_ENV, '1')
    calls = []
    def verifier(image: np.ndarray) -> bool:
        calls.append(True)
        return len(calls) not in (2, 3)
    source = DirectShowSource(DeviceConfig('OBS', 0, verify_fail_streak=fail_limit),
                              REGULAR_SEC, lambda: None)
    state = AsyncScreenState(verifier, source._judge, REGULAR_SEC)
    submitted = []
    try:
        for tick in range(15):
            previous = state.last_verify
            state.update(IMAGE, SIZE, tick*TICK_SEC, True)
            if state.last_verify != previous:
                submitted.append(tick*TICK_SEC)
            if state.worker.future is not None:
                state.worker.future.result(timeout=WAIT_SEC)
                state.update(IMAGE, SIZE, tick*TICK_SEC, True)
        assert submitted == expected
        assert state.verified and state.fail_streak == 0
    finally:
        state.close()


@pytest.mark.parametrize('passed', [False, True])
def test_pass_fail_and_feedback_do_not_change(passed: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(entry.FLAG_ENV, '1')
    def verifier(image: np.ndarray) -> bool:
        return passed
    verifier.feedback = not passed
    state = AsyncScreenState(verifier, judge, REGULAR_SEC)
    try:
        assert state.update(IMAGE, SIZE, 0, True) == (passed, 0 if passed else 1)
        assert state.feedback is (not passed)
        assert state.latest_result.passed is passed
    finally:
        state.close()


def test_flag_configuration_and_release_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.launcher import UserConfig, build_pipeline_config, parse_user_config
    from src.phase_j.release_config import release_user_config
    key = 'live_verify_entry_fast_retry'
    assert live_speed.SPEED_DEFAULTS[key] is False
    assert UserConfig.__dataclass_fields__[key].default is False
    # 配布の既定は ON (2026-11-03 user 承認)。明示の OFF はそのまま通る。
    assert release_user_config({}, tmp_path)[key] is True
    assert release_user_config({key: False}, tmp_path)[key] is False
    assert release_user_config({key: True}, tmp_path)[key] is True
    user = parse_user_config(dict(source='dshow', device_name='OBS', **{key: True}), tmp_path)
    assert build_pipeline_config(user)[key] is True
    parser = argparse.ArgumentParser()
    live_speed.add_arguments(parser)
    default = parser.parse_args([])
    assert getattr(default, key) is False
    assert 'entry_verification' not in live_speed.trial_flags_record(default)
    options = parser.parse_args(['--live-verify-entry-fast-retry'])
    options.output, options.source, options.cpu_threads = tmp_path, 'dshow', 1
    monkeypatch.setattr('os.environ', {})
    live_speed.configure(options, parser)
    assert entry.enabled()
    assert live_speed.trial_flags_record(options)['entry_verification'] == dict(fast_retry=True, period_sec=.25)
    with pytest.raises(ValueError, match='bool'):
        live_speed.validate({key: 1})
