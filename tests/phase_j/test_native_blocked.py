"""計算部品の制御ポリシーによる停止だけを起動前に案内する。"""
from __future__ import annotations

import builtins
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from urllib.request import urlopen

import pytest
from playwright.sync_api import Browser, expect

from src import puyo_core_bridge as bridge
from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_overlay_browser import browser


@pytest.mark.parametrize('message,code,blocked', [
    ('DLL load failed: アプリケーション制御ポリシーによってこのファイルがブロックされました。', None, True),
    ('DLL load failed: An Application Control policy has blocked this file.', None, True),
    ('別の言語の説明', 4551, True),
    ('No module named puyo_core', None, False),
    ('DLL load failed: 指定されたモジュールが見つかりません。', 126, False),
    ('DLL load failed: invalid image hash', 577, False),
])
def test_import_failure_retained(monkeypatch: pytest.MonkeyPatch, message: str,
                                 code: int | None, blocked: bool) -> None:
    error = ImportError(message)
    if code is not None:
        error.winerror = code
    original = builtins.__import__

    def importing(name: str, *args: object, **kwargs: object) -> ModuleType:
        if name == 'puyo_core':
            raise error
        return original(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location('_bridge_policy_test', bridge.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    monkeypatch.setattr(builtins, '__import__', importing)
    spec.loader.exec_module(module)
    assert module.NATIVE_AVAILABLE is False
    assert module.NATIVE_IMPORT_ERROR is error
    assert module.NATIVE_IMPORT_FAILURE == ('native_blocked' if blocked else 'import_error')


def test_nested_error_and_cycle() -> None:
    cause = OSError('他言語')
    cause.winerror = 4551
    error = ImportError('DLL load failed')
    error.__cause__ = cause
    assert bridge.is_native_policy_blocked(error)
    cause.winerror = 126
    cause.__context__ = error
    assert not bridge.is_native_policy_blocked(error)


def test_localized_windows_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge.sys, 'platform', 'win32')
    monkeypatch.setattr(bridge.ctypes, 'FormatError', lambda code: 'Andere Sprache.\r\n', raising=False)
    assert bridge.is_native_policy_blocked(ImportError('DLL load failed: Andere Sprache.'))
    assert not bridge.is_native_policy_blocked(ImportError('DLL load failed: Access denied.'))


@pytest.mark.parametrize('autostart', [False, True])
def test_blocked_state_without_spawn_or_retry(distribution: tuple,
        monkeypatch: pytest.MonkeyPatch, autostart: bool) -> None:
    runtime, url = distribution
    monkeypatch.setattr(bridge, 'NATIVE_IMPORT_FAILURE', 'native_blocked')
    monkeypatch.setattr(bridge, 'NATIVE_IMPORT_ERROR', ImportError('blocked'))
    calls: list[str] = []
    monkeypatch.setattr(runtime, 'scan_inputs', lambda: calls.append('scan'))
    monkeypatch.setattr(runtime, '_schedule_restart', lambda code: calls.append('retry'))
    monkeypatch.setattr('subprocess.Popen', lambda *a, **kw: calls.append('spawn'))
    if autostart:
        runtime._wait_for_input()
    else:
        runtime.start()
    with urlopen(url + '/api/state') as response:
        state = json.load(response)
    assert state['error'] == 'native_blocked'
    assert not state['running']
    assert runtime.child is None
    assert runtime.restart_times == []
    assert calls == []


@pytest.mark.parametrize('reason', [None, 'import_error'])
def test_other_startup_failure_unchanged(distribution: tuple,
        monkeypatch: pytest.MonkeyPatch, reason: str | None) -> None:
    runtime, url = distribution
    monkeypatch.setattr(bridge, 'NATIVE_IMPORT_FAILURE', reason)

    def broken_scan() -> dict:
        raise RuntimeError('ordinary startup failure')

    runtime.settings['auto_input'] = True
    monkeypatch.setattr(runtime, 'scan_inputs', broken_scan)
    runtime.start()
    with urlopen(url + '/api/state') as response:
        assert json.load(response)['error'] == 'startup_failure'
    assert 'ordinary startup failure' in runtime.error['stack_trace']


@pytest.mark.parametrize('language', ['ja', 'en'])
def test_native_notice_language(browser: Browser, distribution: tuple,
                               language: str) -> None:
    runtime, url = distribution
    runtime.configure({'overlay': {'language': language}})
    runtime.record_failure('native_blocked', 'アプリケーション制御ポリシー')
    context = browser.new_context()
    try:
        page = context.new_page()
        page.goto(url + '/settings')
        expect(page.locator('#runtime-state')).to_have_text(
            '要対応' if language == 'ja' else 'Needs attention')
        notice = page.locator('#app-notice')
        expect(notice).to_contain_text('スマート アプリ コントロール' if language == 'ja'
                                     else 'Windows Smart App Control')
        expect(notice).to_contain_text('コード署名' if language == 'ja' else 'code signing')
        expect(notice).to_contain_text('配布サイト' if language == 'ja' else 'distribution site')
        if language == 'en':
            assert not re.search(r'[ぁ-んァ-ヶ一-龯]', notice.inner_text())
        runtime.record_failure('startup_failure', 'ordinary error')
        expect(notice).to_have_text('解析を始められませんでした。「くわしく」を確認してください。'
                                   if language == 'ja' else 'Could not start. Check Details.')
    finally:
        context.close()
