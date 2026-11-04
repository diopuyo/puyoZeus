"""映像の判定順を保ち、設定窓だけで仮想カメラ停止を案内する。"""
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from playwright.sync_api import Browser, expect

from src.phase_j import distribution_input as inputs
from src.phase_j import distribution_runtime as module
from src.phase_j import live_preview
from src.phase_j.live_device import DirectShowSource, VERIFY_PERIOD_SEC
from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_distribution_e2e_r8 import make_source, BAD_IMAGES
from tests.phase_j.test_overlay_browser import browser

NORMAL_IMAGE = np.zeros((180, 320, 3), np.uint8)


def jpeg(stopped: bool) -> bytes:
    image = inputs.stopped_reference() if stopped else NORMAL_IMAGE
    return live_preview.thumbnail(image)


def publish(runtime: module.DistributionRuntime, encoded: bytes) -> None:
    memory = runtime.preview_memory
    assert memory is not None
    live_preview.HEADER.pack_into(memory.buf, 0, 1, 0)
    memory.buf[live_preview.HEADER.size:live_preview.HEADER.size + len(encoded)] = encoded
    live_preview.HEADER.pack_into(memory.buf, 0, 2, len(encoded))


@pytest.mark.parametrize('ready', [False, True])
@pytest.mark.parametrize('stopped', [False, True])
@pytest.mark.parametrize('feedback', [False, True])
def test_screen_hold_matches_third_revision(monkeypatch: pytest.MonkeyPatch,
        ready: bool, stopped: bool, feedback: bool) -> None:
    source, _, statuses = make_source(monkeypatch, None, 'sync')
    calls: list[Any] = []

    def detect(image: Any) -> bool:
        calls.append(image)
        return stopped

    monkeypatch.setattr(inputs, 'is_obs_stopped', detect)
    source._screen_hold(ready, NORMAL_IMAGE, SimpleNamespace(diagnostics={}), 0,
                        SimpleNamespace(feedback=feedback))
    expected = ('verifying' if not ready else 'obs_stopped' if stopped else
                'overlay_feedback' if feedback else 'no_puyo_screen')
    assert statuses[-1][1] == expected
    assert len(calls) == int(ready)


def test_screen_hold_without_status_keeps_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    source, _, _ = make_source(monkeypatch, None, 'sync')
    calls: list[bool] = []
    source.on_status, source.on_hold = None, lambda: calls.append(True)
    for ready in (False, True):
        source._screen_hold(ready, NORMAL_IMAGE, SimpleNamespace(diagnostics={}), 0, None)
    assert calls == [True, True]


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('field', ['input_status', 'calibration'])
def test_preview_warning_recovers_without_input_change(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, shared: bool, field: str) -> None:
    runtime = module.DistributionRuntime(tmp_path)
    clock = [0.]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    runtime.latest = dict(display={field: 'verifying'})
    runtime.input_probe = dict(reason='ready', message='正常')
    snapshot, probe = dict(runtime.latest), dict(runtime.input_probe)
    if shared:
        runtime.preview_environment()
    try:
        for stopped in (True, False, True):
            runtime.preview_image = jpeg(not stopped) if shared else jpeg(stopped)
            if shared:
                publish(runtime, jpeg(stopped))
            clock[0] += module.PREVIEW_STOP_CHECK_SECONDS
            result = runtime.diagnostics()
            assert result['input']['reason'] == ('obs_stopped' if stopped else 'ready')
            if stopped:
                assert result['input']['message'] == inputs.MESSAGES['obs_stopped']
            assert runtime.latest == snapshot and runtime.input_probe == probe
        runtime.latest = dict(display=dict(input_status='ready', match_phase='playing'))
        assert runtime.diagnostics()['input']['reason'] == 'ready'
    finally:
        runtime.close_preview()


def test_preview_check_is_throttled_and_resets(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = module.DistributionRuntime(tmp_path)
    runtime.latest = dict(display=dict(input_status='verifying'))
    runtime.preview_image = jpeg(True)
    clock, calls = [0.], []
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    detector = inputs.is_obs_stopped

    def detect(frame: np.ndarray) -> bool:
        calls.append(clock[0])
        return detector(frame)

    monkeypatch.setattr(inputs, 'is_obs_stopped', detect)
    for stamp in (0., 1., 2., 3.):
        clock[0] = stamp
        assert runtime.diagnostics()['input']['reason'] == 'obs_stopped'
    assert calls == [0., 3.]
    runtime.latest = dict(display=dict(input_status='ready'))
    runtime.diagnostics()
    runtime.latest = dict(display=dict(input_status='verifying'))
    runtime.preview_image = jpeg(False)
    assert runtime.diagnostics()['input'].get('reason') != 'obs_stopped'
    assert calls == [0., 3., 3.]


@pytest.mark.parametrize('encoded', [b'', b'invalid JPEG'])
def test_invalid_preview_keeps_diagnostics(tmp_path: Path, encoded: bytes) -> None:
    runtime = module.DistributionRuntime(tmp_path)
    runtime.latest = dict(display=dict(input_status='verifying'))
    runtime.preview_image = encoded
    assert runtime.diagnostics()['input'].get('reason') != 'obs_stopped'


def test_preview_detector_failure_keeps_diagnostics(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = module.DistributionRuntime(tmp_path)
    runtime.latest = dict(display=dict(input_status='verifying'))
    runtime.preview_image = jpeg(True)

    def fail(frame: Any) -> bool:
        raise ValueError('停止画像判定の故障注入')

    monkeypatch.setattr(inputs, 'is_obs_stopped', fail)
    assert runtime.diagnostics()['input'].get('reason') != 'obs_stopped'


def test_feedback_check_clears_on_bad_and_stopped_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    source, _, _ = make_source(monkeypatch, None, 'sync')
    for index, image in enumerate((*BAD_IMAGES, inputs.stopped_reference(), NORMAL_IMAGE), 1):
        source._feedback_present = True
        source._check_feedback(image, index * VERIFY_PERIOD_SEC)
        assert not source._feedback_present
        assert not source._feedback_hold()


@pytest.mark.parametrize('language', ['ja', 'en'])
def test_settings_stopped_preview_and_resume(browser: Browser, distribution: tuple,
        language: str) -> None:
    runtime, url = distribution
    runtime.configure({'overlay': {'language': language}})
    runtime.preview_environment()
    publish(runtime, jpeg(True))
    runtime.latest = dict(display=dict(input_status='verifying'))
    runtime.child = SimpleNamespace(poll=lambda: None)
    context = browser.new_context()
    page = context.new_page()
    try:
        page.goto(url + '/settings')
        expect(page.locator('#runtime-state')).to_have_text('要対応' if language == 'ja' else 'Needs attention')
        expect(page.locator('#app-notice')).to_contain_text(
            '仮想カメラが止まっています' if language == 'ja' else 'Virtual Camera has stopped')
        assert runtime.latest['display']['input_status'] == 'verifying'
        publish(runtime, jpeg(False))
        expect(page.locator('#runtime-state')).not_to_have_text(
            '要対応' if language == 'ja' else 'Needs attention', timeout=10000)
        runtime.latest = dict(display=dict(input_status='ready', match_phase='playing'))
        expect(page.locator('#runtime-state')).to_have_text('解析中' if language == 'ja' else 'Analyzing')
    finally:
        runtime.child = None
        runtime.close_preview()
        context.close()
