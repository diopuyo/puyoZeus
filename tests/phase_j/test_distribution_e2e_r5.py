"""実物のカメラ・OBS・窓を使わず、5回目の不具合を確認する。"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from playwright.sync_api import Browser, expect
from src.phase_j.distribution_runtime import DistributionRuntime
from src.phase_j.distribution_obs import integrate
from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_obs_integration import MockObs


def test_autostart_obs_only_after_ninety_seconds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import distribution_config as config, distribution_input as inp, distribution_runtime as module
    runtime = DistributionRuntime(tmp_path)
    elapsed, opened = [], []
    monkeypatch.setattr(module.time, 'monotonic', lambda: sum(elapsed))
    devices = [dict(index=0, name='Webcam'), dict(index=1, name='OBS Virtual Camera'),
               dict(index=2, name='Capture card')]
    monkeypatch.setattr(config, 'devices', lambda: devices)
    monkeypatch.setattr(runtime, 'take_preview', lambda: None)
    monkeypatch.setattr(runtime, 'configure', lambda changes: None)
    def probe(index: int) -> dict:
        opened.append(index)
        return dict(reason='ready' if sum(elapsed) >= 90 else 'obs_stopped')
    discover = inp.discover
    monkeypatch.setattr(inp, 'discover', lambda items: discover(items, probe))
    launches = []
    monkeypatch.setattr(runtime, '_watch_child', lambda *args: None)
    monkeypatch.setattr(runtime, 'preview_environment', lambda: {})
    monkeypatch.setattr(module, 'distribution_user_config', lambda *args: {})
    monkeypatch.setattr(module, 'prepare_output', lambda *args: None)
    monkeypatch.setattr(module, 'create_job', lambda: None)
    monkeypatch.setattr(module, 'assign_job', lambda *args: None)
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *args, **kwargs: launches.append(args[0]) or Mock())
    def wait(delay: float) -> None:
        assert runtime.input_probe['reason'] == 'waiting'
        assert 'OBS の仮想カメラが始まるのを待っています' in runtime.input_probe['message']
        elapsed.append(delay)
        assert len(elapsed) <= 18
    monkeypatch.setattr(runtime.cancel_start, 'wait', wait)
    runtime.start_autostart()
    assert elapsed == [5] * 18
    assert opened == [1] * 19
    assert len(launches) == 1 and 'src.phase_j.launcher' in launches[0]
    assert runtime.child is not None and not runtime.input_waiting
    assert not runtime.autostart_waiting


def test_autostart_stop_cancels_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = DistributionRuntime(tmp_path)
    calls = []
    monkeypatch.setattr(runtime, 'start', lambda: calls.append(True))
    monkeypatch.setattr(runtime.cancel_start, 'wait', lambda delay: runtime.stop(final=False))
    runtime.start_autostart()
    assert calls == [True]
    assert runtime.cancel_start.is_set() and not runtime.input_waiting
    assert not runtime.autostart_waiting and runtime.child is None


@pytest.mark.parametrize('autostart', [False, True])
def test_startup_window_visibility(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, autostart: bool) -> None:
    from src.phase_j import distribution_app as app
    runtime = Mock(root=tmp_path, port=8080)
    runtime.settings = dict(port=8080, completed=True, distribution_app_window=True, check_updates=False)
    window = Mock()
    monkeypatch.setattr(app, 'migrate_shortcuts', Mock())
    monkeypatch.setattr(app, 'make_server', Mock(return_value=Mock(next_server=None)))
    monkeypatch.setattr(app, 'AppWindow', window)
    monkeypatch.setattr(app.threading, 'Thread', Mock())
    monkeypatch.setattr(app.threading, 'Timer', Mock())
    monkeypatch.setattr('src.phase_j.distribution_tray.start_tray', Mock())
    monkeypatch.setattr('src.phase_j.distribution_obs.refresh_on_startup', Mock())
    monkeypatch.setattr(app.signal, 'signal', Mock())
    app.run_application(runtime, SimpleNamespace(port=8080, setup=False, no_browser=False, autostart=autostart))
    assert window.call_args.args[-1] is autostart
    window.return_value.run.assert_called_once()


@pytest.mark.parametrize('changed', [False, True])
def test_obs_url_change_result(monkeypatch: pytest.MonkeyPatch, changed: bool) -> None:
    obs = MockObs()
    name = 'puyoZeus Live 12345678'
    url = 'http://127.0.0.1:8766/'
    obs.items = [dict(sourceName=name, inputKind='browser_source')]
    obs.inputs = [dict(inputName=name)]
    obs.settings[name] = dict(url='http://127.0.0.1:8765/' if changed else url)
    client = Mock()
    client.request.side_effect = lambda kind, **data: obs.respond(kind, data)
    monkeypatch.setattr('src.phase_j.distribution_obs.ObsClient', lambda *args: client)
    result = integrate(dict(obs_enabled=True, obs_port=4455, obs_password='secret'), url, apply=True)
    assert result['url_updated'] is changed
    assert result['url'] == url and result['already_added']
    assert obs.settings[name]['url'] == url


@pytest.mark.parametrize('language', ['ja', 'en'])
@pytest.mark.parametrize('sent', [False, True])
def test_log_done_button(distribution: tuple, browser: Browser, language: str, sent: bool) -> None:
    _, url = distribution
    page = browser.new_page()
    page.route('**/api/logs/preview', lambda route: route.fulfill(json=dict(token='test', local_only=not sent)))
    page.route('**/api/logs/send', lambda route: route.fulfill(json=dict(sent=sent, status='saved', receipt='test')))
    page.goto(url + '/settings?language=' + language)
    page.locator('#tab-other').click()
    page.locator('[name=language]').select_option(language)
    page.locator('#send-logs').click()
    expect(page.locator('#confirm-logs')).to_be_enabled()
    page.locator('#confirm-logs').click()
    expect(page.locator('#cancel-logs')).to_have_text('Close' if language == 'en' else '閉じる')
    page.locator('#cancel-logs').click()
    expect(page.locator('#log-dialog')).not_to_be_visible()
    page.locator('#send-logs').click()
    expect(page.locator('#cancel-logs')).to_have_text('Cancel' if language == 'en' else 'キャンセル')
    page.close()


@pytest.mark.parametrize('language', ['ja', 'en'])
def test_obs_updated_notice(distribution: tuple, browser: Browser, language: str) -> None:
    _, url = distribution
    page = browser.new_page()
    page.route('**/api/obs', lambda route: route.fulfill(json=dict(
        scene='配信', already_added=True, url_updated=True, url='http://127.0.0.1:8766/')))
    page.goto(url + '/settings?language=' + language)
    page.locator('#tab-other').click()
    page.locator('[name=language]').select_option(language)
    page.evaluate("async()=>{await obsAction(true)}")
    expect(page.locator('#obs-result')).to_contain_text('Updated the overlay URL' if language == 'en' else '表示の URL を')
    expect(page.locator('#obs-result')).to_contain_text('http://127.0.0.1:8766/')
    page.close()


@pytest.mark.parametrize('language', ['ja', 'en'])
def test_autostart_waiting_label(distribution: tuple, browser: Browser, language: str) -> None:
    _, url = distribution
    page = browser.new_page()
    page.goto(url + '/settings')
    page.locator('#tab-other').click()
    page.locator('[name=language]').select_option(language)
    state = dict(running=False, diagnostics=dict(input_waiting=True, scanning=True, input=dict(
        reason='waiting', message='OBS の仮想カメラが始まるのを待っています / Waiting for OBS Virtual Camera to start')))
    rendered = page.evaluate('''(state)=>{window.renderDistributionUX(state);return [
        document.getElementById('runtime-state').textContent,
        document.getElementById('app-notice').textContent]}''', state)
    assert rendered == (['Waiting for a match', 'Waiting for OBS Virtual Camera to start'] if language == 'en'
                        else ['試合待ち', 'OBS の仮想カメラが始まるのを待っています'])
    page.close()
