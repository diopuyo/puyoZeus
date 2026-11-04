"""配布版の自動反映、入力画像の非永続化、3タブを検証する。"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pytest
from playwright.sync_api import Browser, expect

from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_overlay_browser import browser
from src.phase_j import live_preview


@pytest.mark.parametrize('language', ['ja', 'en'])
def test_stopped_camera_and_error_message(browser: Browser, distribution: tuple, language: str) -> None:
    from types import SimpleNamespace
    runtime, url = distribution
    runtime.configure({'overlay': {'language': language}})
    context = browser.new_context()
    page = context.new_page()
    errors: list[str] = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    runtime.child = SimpleNamespace(poll=lambda: None)
    runtime.latest = dict(display=dict(input_status='obs_stopped'))
    try:
        page.goto(url + '/settings')
        expect(page.locator('#runtime-state')).to_have_text('要対応' if language == 'ja' else 'Needs attention')
        expect(page.locator('#app-notice')).to_contain_text('仮想カメラが止まっています' if language == 'ja' else 'Virtual Camera has stopped')
        expect(page.locator('#input-reason')).to_contain_text('仮想カメラが止まっています' if language == 'ja' else 'Virtual Camera has stopped')
        page.evaluate("setLocalMessage('obs-result',errorMessage(new Error('パスワードが違います / Incorrect password')))")
        expect(page.locator('#obs-result')).to_have_text('パスワードが違います' if language == 'ja' else 'Incorrect password')
        runtime.latest = dict(display=dict(input_status='ready', match_phase='playing'))
        expect(page.locator('#runtime-state')).to_have_text('解析中' if language == 'ja' else 'Analyzing')
        assert not errors
    finally:
        runtime.child = None
        context.close()


@pytest.mark.parametrize('language', ['ja', 'en'])
def test_checking_placement_and_port_notice(browser: Browser, distribution: tuple, language: str) -> None:
    runtime, url = distribution
    runtime.configure({'overlay': {'language': language}})
    context = browser.new_context()
    page = context.new_page()
    errors: list[str] = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(url + '/settings')
    expect(page.locator('#tab-stream')).to_be_visible()
    runtime.scanning = True
    expect(page.locator('#runtime-state')).to_have_text('確認中' if language == 'ja' else 'Checking')
    expect(page.locator('#app-notice')).to_have_text('映像の入力を探しています…' if language == 'ja' else 'Looking for video input…')
    runtime.scanning = False
    runtime.input_probe = dict(selected=dict(name='Fake video', index=0))
    expect(page.locator('#runtime-state')).to_have_text('停止中' if language == 'ja' else 'Stopped')
    expect(page.locator('#input-device')).to_have_text('Fake video')
    runtime.port_conflict = dict(kind='different', root='C:/another-copy', port=8765)
    expect(page.locator('#port-conflict')).to_be_visible()
    expect(page.locator('#port-conflict')).to_contain_text('C:/another-copy')
    expect(page.locator('#port-conflict button')).to_be_visible()
    runtime.port_conflict = dict(kind='other', port=8765)
    expect(page.locator('#port-conflict button')).to_be_hidden()
    expect(page.locator('#port-conflict')).to_contain_text(url + '/')
    page.locator('#tab-appearance').click()
    page.locator('[name=layout]').select_option('l-top-left')
    expect(page.locator('#placement-hint')).to_be_visible()
    expect(page.locator('#placement-hint')).to_contain_text('%')
    expect(page.locator('#placement-hint')).to_contain_text('In OBS' if language == 'en' else 'OBS でゲーム')
    page.locator('[name=layout]').select_option('center')
    expect(page.locator('#placement-hint')).to_be_hidden()
    assert not errors
    context.close()


@pytest.mark.parametrize('language,already', [('ja', False), ('ja', True), ('en', False), ('en', True)])
def test_obs_success_message(browser: Browser, distribution: tuple, language: str, already: bool) -> None:
    runtime, url = distribution
    runtime.configure({'overlay': {'language': language}})
    context = browser.new_context()
    page = context.new_page()
    page.route('**/api/obs', lambda route: route.fulfill(json=dict(scene='Test scene', applied=True,
               already_added=already, browser_sources=[], virtual_camera=True)))
    page.goto(url + '/settings')
    page.locator('#obs-apply').click()
    expect(page.locator('#obs-result')).to_contain_text('Test scene')
    expect(page.locator('#obs-result')).to_contain_text('すでに追加済み' if language == 'ja' and already else
        '表示を追加' if language == 'ja' else 'already' if already else 'Added')
    expect(page.locator('#obs-result')).to_contain_text('ソース' if language == 'ja' else 'Source')
    context.close()


def test_preview_crosses_recognition_process(distribution: tuple) -> None:
    runtime, _ = distribution
    environment = runtime.preview_environment()
    root = Path(__file__).resolve().parents[2]
    script = (f'import sys;sys.path.insert(0,{str(root)!r});'
              'from src.phase_j.live_preview import publish_image;publish_image(b"frame from recognition")')
    run = subprocess.run([sys.executable, '-c', script], env=environment, capture_output=True, check=True)
    assert run.returncode == 0
    assert live_preview.shared_image(runtime.preview_memory) == b'frame from recognition'
    live_preview.HEADER.pack_into(runtime.preview_memory.buf, 0, 3, 8)
    assert live_preview.shared_image(runtime.preview_memory) == b''


def test_preview_is_opt_in_and_throttled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_preview, '_image', b'')
    monkeypatch.setattr(live_preview, '_updated', 0.0)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    monkeypatch.delenv('PUYO_INPUT_PREVIEW', raising=False)
    live_preview.offer_frame(frame)
    assert live_preview.current_image() == b''
    monkeypatch.setenv('PUYO_INPUT_PREVIEW', '1')
    live_preview.offer_frame(frame)
    first = live_preview.current_image()
    assert first.startswith(b'\xff\xd8')
    live_preview.offer_frame(np.full_like(frame, 255))
    assert live_preview.current_image() == first


def test_scan_preview_never_enters_diagnostics(distribution: tuple) -> None:
    runtime, url = distribution
    image = live_preview.thumbnail(np.zeros((1080, 1920, 3), dtype=np.uint8))
    selected = dict(index=0, name='Camera', preview=base64.b64encode(image).decode())
    runtime.input_probe = dict(devices=[selected], selected=selected)
    runtime.take_preview()
    assert 'preview' not in json.dumps(runtime.diagnostics())
    with urlopen(url + '/api/input-preview.jpg') as response:
        assert response.read() == image
        assert response.headers['Cache-Control'] == 'no-store'
    assert not list(runtime.root.rglob('*.jpg'))


def test_device_switch_keeps_existing_waiter(distribution: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, url = distribution
    monkeypatch.setattr('src.phase_j.distribution_config.devices', lambda: [dict(index=2, name='Second input')])
    runtime.input_waiting = True
    runtime.switch_device(2)
    assert runtime.preferred_device == 2 and runtime.input_waiting
    assert runtime.start_thread is None and not runtime.cancel_start.is_set()
    with urlopen(url + '/api/state') as response:
        assert json.load(response)['selected_input'] == 2
    with pytest.raises(ValueError):
        runtime.switch_device(3)
    runtime.input_waiting = False


def test_running_preview_only_reads_child_http(distribution: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    from types import SimpleNamespace
    runtime, url = distribution
    image = live_preview.thumbnail(np.zeros((180, 320, 3), dtype=np.uint8))
    monkeypatch.setattr('src.phase_j.distribution_http.urlopen', lambda *a, **k: io.BytesIO(image))
    monkeypatch.setattr('cv2.VideoCapture', lambda *a, **k: pytest.fail('二重に入力を開いた'))
    runtime.child = SimpleNamespace(poll=lambda: None)
    runtime.inner_port = 12345
    try:
        with urlopen(url + '/api/input-preview.jpg') as response:
            assert response.read() == image
    finally:
        runtime.child = None
        runtime.inner_port = 0


def test_autosave_updates_existing_long_url(browser: Browser, distribution: tuple) -> None:
    runtime, url = distribution
    context = browser.new_context(viewport=dict(width=1100, height=800))
    page = context.new_page()
    errors: list[str] = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    overlay = context.new_page()
    overlay.on('pageerror', lambda error: errors.append(str(error)))
    overlay.goto(url + '/?theme=pop-cream&layout=l-bottom-right')
    overlay.evaluate('window.identityToken=123')
    page.goto(url + '/settings')
    expect(page.locator('#tab-stream')).to_be_visible()
    page.locator('#tab-appearance').click()
    expect(page.locator('#overlay')).to_be_visible()
    with page.expect_response(lambda response: response.url.endswith('/api/settings') and response.request.post_data and 'pop-cream' in response.request.post_data):
        page.locator('[name=theme]').select_option('pop-cream')
        page.locator('[name=layout]').select_option('l-top-left')
    expect(overlay.locator('#stage')).to_have_attribute('data-pop', 'cream')
    expect(overlay.locator('#panel')).to_have_attribute('data-layout', 'l-top-left')
    assert overlay.evaluate('window.identityToken') == 123
    assert page.locator('#url').input_value() == url + '/'
    page.locator('[name=layout]').select_option('center')
    expect(overlay.locator('#panel')).to_have_attribute('data-layout', 'center')
    assert overlay.locator('#panel').evaluate('(node)=>node.style.width') == ''
    assert runtime.settings['overlay']['theme'] == 'pop-cream'
    assert not errors
    context.close()


@pytest.mark.parametrize('language', ['ja', 'en'])
def test_tabs_and_input_preview(browser: Browser, distribution: tuple, language: str) -> None:
    runtime, url = distribution
    runtime.settings['language'] = language
    runtime.settings['onboarding_seen'] = True
    runtime.preview_image = live_preview.thumbnail(np.zeros((180, 320, 3), dtype=np.uint8))
    page = browser.new_page(viewport=dict(width=1100, height=800))
    errors: list[str] = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(url + '/settings')
    expect(page.locator('#tab-stream')).to_have_text('Stream' if language == 'en' else '配信')
    expect(page.locator('#input-preview')).to_be_visible()
    page.wait_for_function('document.querySelector("#input-preview").naturalWidth===320')
    directory = Path(os.environ.get('SETUP_REMOVAL_JOB', str(runtime.root))) / 'ux_screenshots'
    directory.mkdir(exist_ok=True)
    for tab in ('stream', 'appearance', 'other'):
        page.locator('#tab-' + tab).click()
        expect(page.locator('#page-' + tab)).to_be_visible()
        page.screenshot(path=str(directory / f'{language}_{tab}.png'))
        assert page.evaluate('document.documentElement.scrollHeight') <= 830
    assert not errors
    page.close()


def test_packaged_launchers_use_parent_root(tmp_path: Path) -> None:
    from build_release import arrange_launchers
    sources = Path(__file__).resolve().parents[2] / 'packaging'
    for source, name in [('PuyoLive.vbs', 'puyoZeus Live.vbs'), ('PuyoLive.vbs', 'PuyoLive.vbs'),
                         ('PuyoLive-distribution.bat', 'PuyoLive.bat'), ('Uninstall.bat', 'Uninstall.bat'),
                         ('Uninstall.bat', 'アンインストール.bat')]:
        (tmp_path / name).write_bytes((sources / source).read_bytes())
    arrange_launchers(tmp_path)
    assert [p for p in tmp_path.iterdir() if p.suffix in ('.bat', '.vbs')] == []
    assert (tmp_path / 'tools/puyoZeus Live.vbs').is_file()
    assert (tmp_path / 'tools/アンインストール.bat').is_file()
    assert '%~dp0..\\' in (tmp_path / 'tools/PuyoLive.bat').read_text('utf-8')
    assert 'fs.GetParentFolderName(fs.GetParentFolderName(' in (tmp_path / 'tools/PuyoLive.vbs').read_text('utf-8')
    assert '%~dp0..\\' in (tmp_path / 'tools/Uninstall.bat').read_text('utf-8')


@pytest.mark.parametrize('language', ['ja', 'en'])
def test_status_badge_and_sentence_share_state(browser: Browser, distribution: tuple,
                                              language: str) -> None:
    runtime, url = distribution
    runtime.settings.update(language=language, onboarding_seen=True)
    page = browser.new_page()
    page.goto(url + '/settings')
    page.wait_for_function('Boolean(window.lastDistributionState)')
    page.route('**/api/state', lambda route: route.abort())
    labels = ['Stopped', 'Waiting for a match', 'Analyzing'] if language == 'en' else ['停止中', '試合待ち', '解析中']
    texts = ['Press Start when you are ready.', 'Waiting for a match to appear', 'Watching the game video.'] if language == 'en' else [
        '準備ができたら「起動」を押してください。', '試合が映るのを待っています', 'ゲームの映像を見ています。']
    # A successful input scan must not overwrite the waiting sentence, including on language/form refresh.
    for running, calibration, phase, index in [(False, '', '', 0), (True, 'verifying', '', 1),
                                               (True, 'ready', 'between_matches', 1), (True, 'ready', 'playing', 2)]:
        state = dict(running=running, settings={}, diagnostics=dict(calibration=calibration, match_phase=phase,
                     input=dict(reason='ready', message='ゲーム画面を確認しました / Game input verified')))
        page.evaluate('(state)=>{window.lastDistributionState=state;renderAppNotice();}', state)
        expect(page.locator('#runtime-state')).to_have_text(labels[index])
        expect(page.locator('#app-notice')).to_have_text(texts[index])
    page.close()


@pytest.mark.parametrize('language', ['ja', 'en'])
def test_visible_url_and_one_click_obs(browser: Browser, distribution: tuple, language: str) -> None:
    runtime, url = distribution
    runtime.settings.update(language=language, onboarding_seen=True, obs_enabled=False)
    page = browser.new_page(viewport=dict(width=1100, height=800))
    page.add_init_script('navigator.clipboard.writeText=async value=>{window.copiedURL=value;}')
    enabled_at_setup: list[bool] = []
    def setup(route: Any) -> None:
        enabled_at_setup.append(runtime.settings['obs_enabled'])
        assert route.request.post_data_json == {'apply': True}
        route.fulfill(json=dict(scene='puyoZeus Live', browser_sources=['Overlay'], warning=''))
    page.route('**/api/obs', setup)
    page.goto(url + '/settings')
    expect(page.locator('#stream-url')).to_have_value(url + '/')
    expect(page.locator('#copy-obs-url')).to_be_in_viewport()
    expect(page.locator('#obs-enabled')).to_be_hidden()
    page.locator('#copy-obs-url').click()
    page.wait_for_function('window.copiedURL===location.origin+"/"')
    page.locator('#obs-apply').click()
    expect(page.locator('#obs-result')).to_contain_text('puyoZeus Live')
    assert enabled_at_setup == [True]
    assert json.loads((runtime.root / 'distribution.json').read_text('utf-8'))['obs_enabled'] is True
    page.locator('#tab-appearance').click()
    expect(page.locator('#reset')).to_have_text('Reset appearance' if language == 'en' else '見た目を初期に戻す')
    page.close()
