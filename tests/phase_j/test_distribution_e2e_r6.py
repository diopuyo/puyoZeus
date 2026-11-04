"""6回目の修正を模擬プロセスと合成画像だけで確認する。"""
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from playwright.sync_api import Browser

from src.phase_j.distribution_runtime import DistributionRuntime
from src.phase_j import distribution_runtime as module
from src.phase_j.distribution_input import discover, marker_image, MESSAGES
from src.phase_j.live_device import DirectShowSource, VERIFY_PERIOD_SEC
from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_overlay_browser import browser


def dead_child() -> Mock:
    return Mock(stdout=BytesIO(b'synthetic crash\n'), wait=Mock(return_value=1), poll=Mock(return_value=1))


def test_restart_limit_and_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = DistributionRuntime(tmp_path)
    clock, delays, launches = [0.0], [], []
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(runtime.cancel_start, 'wait', lambda delay: delays.append(delay) or False)
    monkeypatch.setattr(runtime, 'start', lambda: launches.append(True))
    runtime.settings['auto_input'] = False
    for _ in range(module.RESTART_LIMIT + 1):
        runtime.child = dead_child()
        runtime.latest = dict(display=dict(input_status='ready'))
        runtime._watch_child(runtime.child)
    assert len(launches) == 3 and delays == [5, 10, 15]
    assert runtime.error['type'] == 'abnormal_exit' and not runtime.restarting
    log = (runtime.logs / 'runtime.log').read_text('utf-8')
    assert log.count('reason=abnormal_exit') == 4 and 'allowed=False' in log
    clock[0] = module.RESTART_WINDOW_SECONDS
    runtime.child = dead_child()
    runtime._watch_child(runtime.child)
    assert len(launches) == 4 and len(runtime.restart_times) == 1
    assert runtime.restarting and not runtime.error
    runtime.accept_snapshot(dict(display=dict(input_status='ready')))
    assert not runtime.restarting


@pytest.mark.parametrize('during_wait', [False, True])
def test_stop_prevents_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, during_wait: bool) -> None:
    runtime = DistributionRuntime(tmp_path)
    child = dead_child()
    runtime.child = child
    launch = Mock()
    monkeypatch.setattr(runtime, 'start', launch)
    if during_wait:
        monkeypatch.setattr(runtime.cancel_start, 'wait', lambda delay: runtime.stop(final=False) or True)
    else:
        runtime.stop(final=False)
    runtime._watch_child(child)
    assert launch.call_count == 0 and not runtime.restarting and not runtime.error


def test_stale_child_cannot_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = DistributionRuntime(tmp_path)
    runtime.child = Mock()
    launch = Mock()
    monkeypatch.setattr(runtime, 'start', launch)
    runtime._watch_child(dead_child())
    assert not runtime.restart_times and launch.call_count == 0


def test_restart_creates_new_analysis_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = DistributionRuntime(tmp_path)
    runtime.settings['auto_input'] = False
    runtime.child = dead_child()
    replacement = Mock()
    replacement.poll.return_value = None
    monkeypatch.setattr(runtime.cancel_start, 'wait', lambda delay: False)
    monkeypatch.setattr(runtime, 'preview_environment', lambda: {})
    monkeypatch.setattr(module, 'distribution_user_config', lambda *args: {})
    monkeypatch.setattr(module, 'prepare_output', lambda *args: None)
    monkeypatch.setattr(module, 'create_job', lambda: None)
    monkeypatch.setattr(module, 'assign_job', lambda *args: None)
    monkeypatch.setattr(module.threading, 'Thread', Mock())
    spawn = Mock(return_value=replacement)
    monkeypatch.setattr(module.subprocess, 'Popen', spawn)
    runtime._watch_child(runtime.child)
    assert runtime.child is replacement and spawn.call_count == 1
    assert 'src.phase_j.launcher' in spawn.call_args.args[0]
    assert not runtime.error and runtime.restarting


def test_feedback_blocks_accepted_frames_then_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_device import DeviceConfig, FrameContinuity
    from tests.phase_j.test_dshow_color import FakeDevice
    clock, statuses = [0.0], []
    clean = np.zeros((1080, 1920, 3), dtype=np.uint8)
    marked = clean.copy()
    marker = marker_image()
    marked[20:20 + marker.shape[0], 948:948 + marker.shape[1]] = marker
    device = FakeDevice(clean)
    device.read = lambda: (True, marked if 1 <= clock[0] < 2 else clean)
    monkeypatch.setenv('PUYO_LIVE_VERIFY_ASYNC', '0')
    monkeypatch.setenv('PUYO_LIVE_ASYNC_SCREEN_VERIFY', '0')
    monkeypatch.setattr(FrameContinuity, 'ready', lambda *args: True)
    source = DirectShowSource(DeviceConfig('synthetic', 0, color_correction='off'), 3,
        lambda: None, lambda *args: device, lambda image: True,
        lambda: clock[0], lambda delay: clock.__setitem__(0, clock[0] + delay),
        on_status=statuses.append)
    frames = list(source)
    assert 'overlay_feedback' in statuses and device.released
    assert any(frame.media_sec < 1 for frame in frames)
    assert any(frame.media_sec >= 2 for frame in frames)


def test_feedback_on_accepted_synthetic_input() -> None:
    source = DirectShowSource.__new__(DirectShowSource)
    source._feedback_checked_at = float('-inf')
    source._feedback_present = False
    source.on_status, source.sleep = Mock(), Mock()
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    marker = marker_image()
    image[20:20 + marker.shape[0], 948:948 + marker.shape[1]] = marker
    source._check_feedback(image, 0)
    assert source._hold_input(True, True, image, 0, Mock(), 0, Mock())
    source.on_status.assert_called_with('overlay_feedback')
    clean = np.zeros_like(image)
    source._check_feedback(clean, VERIFY_PERIOD_SEC / 2)
    assert source._feedback_present
    source._check_feedback(clean, VERIFY_PERIOD_SEC)
    assert not source._hold_input(True, True, clean, VERIFY_PERIOD_SEC, Mock(), 0, Mock())


@pytest.mark.parametrize('name', ['HD Webcam', 'OBS Virtual Camera'])
def test_wrong_size_guidance(name: str) -> None:
    result = discover([dict(index=0, name=name)], lambda index: dict(reason='wrong_size'))
    assert result['reason'] == 'wrong_size'
    if name == 'OBS Virtual Camera':
        assert result['message'] == MESSAGES['wrong_size']
    else:
        assert name in result['message'] and 'Select OBS Virtual Camera' in result['message']
        assert 'ゲームの映像ではないようです' in result['message'] and '1920' not in result['message']


def test_feedback_diagnostics_clear(tmp_path: Path) -> None:
    runtime = DistributionRuntime(tmp_path)
    runtime.input_probe = dict(reason='ready', message=MESSAGES['ready'])
    runtime.accept_snapshot(dict(display=dict(input_status='overlay_feedback')))
    assert runtime.diagnostics()['input']['message'] == MESSAGES['feedback']
    runtime.accept_snapshot(dict(display=dict(input_status='ready')))
    assert runtime.diagnostics()['input']['reason'] == 'ready'


@pytest.mark.parametrize('language', ['ja', 'en'])
@pytest.mark.parametrize('restarting', [False, True])
def test_recovery_and_feedback_labels(distribution: tuple, browser: Browser, language: str, restarting: bool) -> None:
    _, url = distribution
    page = browser.new_page()
    page.goto(url + '/settings')
    page.locator('#tab-other').click()
    page.locator('[name=language]').select_option(language)
    state = dict(running=True, settings=dict(device_name='synthetic'),
        diagnostics=dict(restarting=restarting, calibration='overlay_feedback',
        input=dict(reason='feedback', message=MESSAGES['feedback'])))
    rendered = page.evaluate('''state=>{window.renderDistributionUX(state);return [
        document.getElementById('runtime-state').textContent,
        document.getElementById('app-notice').textContent]}''', state)
    expected = (['Waiting for a match', 'Restarting analysis'] if language == 'en'
                else ['試合待ち', '解析を再開しています'])
    if restarting:
        assert rendered == expected
    else:
        assert rendered[0] == ('Needs attention' if language == 'en' else '要対応')
        assert ('overlay appears' if language == 'en' else '表示が映り込んでいます') in rendered[1]
    page.close()
