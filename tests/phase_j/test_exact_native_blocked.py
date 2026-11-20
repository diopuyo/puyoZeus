"""exact DLLの拒否を起動前に検出し、既存の案内と状態へ渡す。"""
from __future__ import annotations

import json
import subprocess
from urllib.request import urlopen

import pytest
from playwright.sync_api import Browser, expect

from src import puyo_core_bridge as bridge
from src.phase_j import live_native
from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_overlay_browser import browser


@pytest.fixture
def blocked_dll(monkeypatch: pytest.MonkeyPatch) -> OSError:
    error = OSError('test policy rejection')
    error.winerror = bridge.APPLICATION_CONTROL_BLOCKED

    def fail() -> None:
        raise error

    monkeypatch.setenv('PUYO_LIVE_EXACT_NATIVE', '1')
    monkeypatch.setattr(bridge, 'NATIVE_IMPORT_FAILURE', None)
    monkeypatch.setattr(live_native, 'load_library', fail)
    return error


@pytest.mark.parametrize('autostart', [False, True])
def test_exact_blocked_before_capture_or_spawn(distribution: tuple,
        monkeypatch: pytest.MonkeyPatch, blocked_dll: OSError, autostart: bool) -> None:
    runtime, url = distribution
    calls: list[str] = []
    original_popen = subprocess.Popen

    def popen(command: list, *args: object, **kwargs: object) -> object:
        if 'src.phase_j.launcher' in command:
            calls.append('spawn')
            raise AssertionError('拒否後に解析子を起動した')
        return original_popen(command, *args, **kwargs)

    monkeypatch.setattr(runtime, 'scan_inputs', lambda: calls.append('scan'))
    monkeypatch.setattr(runtime, '_schedule_restart', lambda code: calls.append('retry'))
    monkeypatch.setattr('subprocess.Popen', popen)
    runtime._wait_for_input() if autostart else runtime.start()
    with urlopen(url + '/api/state') as response:
        state = json.load(response)
    assert state['error'] == 'native_blocked'
    assert not state['running']
    assert runtime.child is None
    assert runtime.restart_times == []
    assert calls == []
    assert 'puyo_live_exact.dll' in runtime.error['stack_trace']
    assert str(blocked_dll) in runtime.error['stack_trace']


@pytest.mark.parametrize('language', ['ja', 'en'])
def test_exact_blocked_notice(browser: Browser, distribution: tuple,
        blocked_dll: OSError, language: str) -> None:
    runtime, url = distribution
    runtime.configure({'overlay': {'language': language}})
    runtime.start()
    context = browser.new_context()
    try:
        page = context.new_page()
        page.goto(url + '/settings')
        expect(page.locator('#runtime-state')).to_have_text(
            '要対応' if language == 'ja' else 'Needs attention')
        expect(page.locator('#app-notice')).to_contain_text(
            '計算部品を止めたため' if language == 'ja' else 'blocked a required component')
    finally:
        context.close()


@pytest.mark.parametrize('error', [OSError('missing DLL'), AttributeError('missing ABI')])
def test_non_policy_errors_keep_existing_fallback(monkeypatch: pytest.MonkeyPatch,
        error: Exception) -> None:
    def fail() -> None:
        raise error

    monkeypatch.setenv('PUYO_LIVE_EXACT_NATIVE', '1')
    monkeypatch.setattr(live_native, 'load_library', fail)
    assert live_native.native_policy_error() is None
    with live_native.configured_native():
        pass


def test_disabled_native_is_not_loaded(monkeypatch: pytest.MonkeyPatch,
        blocked_dll: OSError) -> None:
    monkeypatch.setenv('PUYO_LIVE_EXACT_NATIVE', '0')
    assert live_native.native_policy_error() is None


def test_successful_native_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIVE_EXACT_NATIVE', '1')
    monkeypatch.setattr(live_native, 'load_library', lambda: object())
    assert live_native.native_policy_error() is None


def test_distribution_uses_child_config_not_parent_environment(distribution: tuple,
        monkeypatch: pytest.MonkeyPatch, blocked_dll: OSError) -> None:
    runtime, _ = distribution
    monkeypatch.setenv('PUYO_LIVE_EXACT_NATIVE', '0')
    assert 'puyo_live_exact.dll' in runtime._native_policy_failure()
    monkeypatch.setattr('src.phase_j.distribution_runtime.distribution_user_config',
                        lambda *args: {'live_exact_native': False})
    assert runtime._native_policy_failure() is None
