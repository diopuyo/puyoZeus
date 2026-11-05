"""配信の準備を模擬OBSと模擬状態で確認する。ゲーム画像は使わない。"""
from pathlib import Path
import os
import threading

import pytest
from playwright.sync_api import Browser, Page, expect
from websockets.sync.server import serve

from src.phase_j.distribution_config import load_settings
from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_obs_integration import MockObs

FRAMES = Path(os.environ.get('STREAM_SCREENSHOTS_PATH', 'test_outputs/stream'))
SIZES = [(1100, 800), (900, 700)]


@pytest.mark.parametrize('language,separator', [('ja', ' ・ '), ('en', ' · ')])
def test_obs_status_separator(browser: Browser, distribution: tuple, language: str, separator: str) -> None:
    """模擬状態の連携表示で、言語に合う区切りを確認する。"""
    runtime, url = distribution
    runtime.configure({'language': language})
    page = browser.new_page()
    try:
        page.goto(url + '/settings')
        page.wait_for_function('window.distributionReady')
        page.evaluate('''() => {
          window.streamObsStatus={already_added:true,virtual_camera:true};
          renderObsStatus(true);
        }''')
        parts = ['OBS ✓ Connected', 'Browser source ✓', 'Virtual camera ✓ Running'] if language == 'en' else [
            'OBS ✓ 連携済み', 'ブラウザソース ✓', '仮想カメラ ✓ 開始中']
        expect(page.locator('#obs-connection')).to_have_text(separator.join(parts))
    finally:
        page.close()


def frame(page: Page, name: str) -> None:
    FRAMES.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(FRAMES / (name + '.png')), full_page=True)


def publish(page: Page, calibration: str, running: bool = True) -> None:
    page.evaluate('''([calibration,running])=>{
      const state=window.lastDistributionState;
      state.running=running;state.diagnostics={calibration,match_phase:'playing'};
      renderDistributionUX(state);
    }''', [calibration, running])


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('language', ['ja', 'en'])
def test_stream_journey(browser: Browser, distribution: tuple, size: tuple, language: str) -> None:
    runtime, url = distribution
    runtime.configure({'language': language})
    obs = MockObs()
    with serve(obs.handle, '127.0.0.1', 0, subprotocols=['obswebsocket.json']) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        runtime.configure({'obs_port': server.socket.getsockname()[1]})
        page = browser.new_page(viewport={'width': size[0], 'height': size[1]})
        errors = [];page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url + '/settings');page.wait_for_function('window.distributionReady')
        expect(page.locator('#stream-steps')).to_have_attribute('open', '')
        expect(page.locator('#toggle-runtime')).to_have_text('Start analysis' if language == 'en' else '解析を始める')
        expect(page.locator('#step-obs-mark')).to_have_text('①')
        box = page.locator('#obs-apply').bounding_box()
        assert box['y'] + box['height'] <= size[1]
        box = page.locator('.ux-obs').bounding_box()
        assert box['y'] + box['height'] <= size[1]
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert page.locator('#page-stream summary', has_text='詳細設定').count() == 0
        page.locator('#stream-details summary').click()
        expect(page.locator('#live-diagnostics')).to_be_visible()
        page.locator('#stream-details summary').click()
        frame(page, f'{language}_{size[0]}_initial')
        page.locator('#obs-password').fill('secret');page.locator('#obs-apply').click()
        expect(page.locator('#step-obs-mark')).to_have_text('✓')
        expect(page.locator('#obs-connection')).to_contain_text('✓')
        assert runtime.settings['stream_obs_done']
        frame(page, f'{language}_{size[0]}_obs_done')
        verify_analysis(page, runtime, language, size)
        assert errors == []
        page.close();server.shutdown()


def verify_analysis(page: Page, runtime: object, language: str, size: tuple) -> None:
    page.route('**/api/state', lambda route: route.fulfill(json={
        **page.evaluate('window.lastDistributionState'), 'running': True,
        'settings': {**runtime.settings, 'obs_password': None},
        'diagnostics': {'calibration': 'ready', 'match_phase': 'playing'}}))
    publish(page, 'ready')
    expect(page.locator('#step-camera-mark')).to_have_text('✓')
    expect(page.locator('#stream-steps')).not_to_have_attribute('open', '')
    expect(page.locator('#toggle-runtime')).to_have_text('Stop analysis' if language == 'en' else '解析を止める')
    frame(page, f'{language}_{size[0]}_playing')
    assert load_settings(runtime.root)['stream_camera_done']
    publish(page, 'overlay_feedback')
    expect(page.locator('#step-camera')).to_have_class('output needs-attention')
    expect(page.locator('#stream-steps')).to_have_attribute('open', '')
    expect(page.locator('#app-notice a')).to_have_attribute('href', '#step-camera')
    frame(page, f'{language}_{size[0]}_feedback')
    runtime.settings = load_settings(runtime.root)
    page.unroute('**/api/state');page.reload();page.wait_for_function('window.distributionReady')
    expect(page.locator('#stream-steps')).not_to_have_attribute('open', '')
    frame(page, f'{language}_{size[0]}_returning')
    page.locator('#stream-steps summary').first.click()
    expect(page.locator('#step-camera-mark')).to_have_text('✓')


def test_obs_status_read_only_and_failure(browser: Browser, distribution: tuple) -> None:
    runtime, url = distribution
    obs = MockObs()
    with serve(obs.handle, '127.0.0.1', 0, subprotocols=['obswebsocket.json']) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        runtime.configure({'obs_enabled': True, 'obs_password': 'secret',
                           'obs_port': server.socket.getsockname()[1], 'language': 'ja'})
        page = browser.new_page();page.goto(url + '/settings')
        expect(page.locator('#obs-connection')).to_contain_text('✓ 連携済み')
        assert 'GetVirtualCamStatus' in obs.calls
        calls = len(obs.calls)
        page.evaluate('refreshObsStatus(window.lastDistributionState)')
        assert len(obs.calls) == calls
        page.evaluate('window.streamObsChecked-=12000;refreshObsStatus(window.lastDistributionState)')
        assert len(obs.calls) > calls
        assert not any(c.startswith(('Create', 'Start', 'Set')) for c in obs.calls)
        assert 'obs_password' not in page.request.get(url + '/api/state').json()['settings']
        server.shutdown()
        page.evaluate('window.streamObsChecked=0;refreshObsStatus(window.lastDistributionState)')
        expect(page.locator('#obs-connection')).to_contain_text('OBS につながりません')
        expect(page.locator('#toggle-runtime')).to_be_enabled()
        frame(page, 'obs_disconnected');page.close()

@pytest.mark.parametrize('key,diagnostics,label', [
    ('checking', {'scanning': True}, '確認中'),
    ('waiting', {'input_waiting': True}, '試合待ち'),
    ('attention', {'calibration': 'obs_stopped'}, '要対応'),
])
def test_status_and_keyboard(browser: Browser, distribution: tuple, key: str,
                             diagnostics: dict, label: str) -> None:
    runtime, url = distribution
    runtime.configure({'language': 'ja'})
    page = browser.new_page(viewport={'width': 900, 'height': 700})
    page.goto(url + '/settings');page.wait_for_function('window.distributionReady')
    page.evaluate('''d=>{const s=window.lastDistributionState;
        s.running=false;s.diagnostics=d;renderDistributionUX(s);}''', diagnostics)
    expect(page.locator('#runtime-state')).to_have_text(label)
    assert not runtime.settings['stream_camera_done']
    frame(page, key)
    page.locator('#stream-details summary').focus();page.keyboard.press('Enter')
    expect(page.locator('#live-diagnostics')).to_be_visible()
    assert page.locator('#stream-details summary').evaluate(
        "e=>getComputedStyle(e).outlineStyle") != 'none'
    page.locator('#stream-details summary').press('Enter')
    expect(page.locator('#scan-input')).not_to_be_visible()
    page.locator('.ux-video summary').click()
    expect(page.locator('#scan-input-help')).to_contain_text('解析を止めてから')
    page.close()


def test_failed_setup_does_not_complete(browser: Browser, distribution: tuple) -> None:
    runtime, url = distribution
    runtime.configure({'language': 'ja'})
    page = browser.new_page();page.goto(url + '/settings')
    page.route('**/api/obs', lambda route: route.fulfill(status=400, json={'error': 'OBS につながりません'}))
    page.locator('#obs-password').fill('synthetic');page.locator('#obs-apply').click()
    expect(page.locator('#obs-result')).to_have_text('OBS につながりません')
    expect(page.locator('#step-obs-mark')).to_have_text('①')
    assert not runtime.settings['stream_obs_done']
    assert not runtime.settings['stream_camera_done']
    page.close()
