"""実ブラウザで言語・マスコット・設定の確認操作を検証する。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Browser, Page, expect

from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_overlay_options_browser import server
from src.phase_j.live_publish import LiveOverlayServer
from scripts.render_overlay_design import snapshot


@pytest.mark.parametrize('distribution_mode', [False, True])
@pytest.mark.parametrize('fragment', ['', '#logs'])
def test_settings_actions_match_server(browser: Browser, distribution: tuple,
                                       server: LiveOverlayServer, distribution_mode: bool,
                                       fragment: str) -> None:
    """試験版では専用操作を隠し、配布版ではログ確認の直リンクも維持する。"""
    _, distribution_url = distribution
    url = distribution_url if distribution_mode else f'http://127.0.0.1:{server.address()[1]}'
    page = browser.new_page()
    errors: list[str] = []
    posts: list[str] = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('request', lambda request: posts.append(request.url) if request.method == 'POST' else None)
    with page.expect_response(lambda response: response.url.endswith('/api/state')):
        page.goto(url + '/settings' + fragment)
    for selector in ('#save-settings', '#send-logs', '#stop-app', 'a[href="/setup"]',
                     '#uninstall-app', '#live-diagnostics', '#rescan-input', '#obs-enabled',
                     '#obs-check', '#obs-apply'):
        if distribution_mode:
            expect(page.locator(selector)).to_be_visible()
        else:
            expect(page.locator(selector)).to_be_hidden()
    if distribution_mode and fragment:
        expect(page.locator('#log-dialog')).to_be_visible()
    else:
        expect(page.locator('#log-dialog')).to_be_hidden()
    expect(page.locator('#copy')).to_be_visible()
    expect(page.locator('#overlay')).to_be_visible()
    assert not errors
    if not distribution_mode:
        assert not posts
    page.close()


def test_setup_uses_current_catalog(browser: Browser, distribution: tuple) -> None:
    """初回導線でも旧テーマやdata案を再び選ばせない。"""
    _, url = distribution
    page = browser.new_page()
    page.goto(url+'/setup')
    expect(page.locator('#theme option')).to_have_count(2)
    # 採用した四配置を出す。旧配置URLの読み込み互換は別試験で確認する。
    expect(page.locator('#layout option')).to_have_count(4)
    expect(page.locator('#theme')).to_have_value('pop')
    expect(page.locator('#mascot')).to_be_checked()
    page.locator('#language').select_option('en')
    expect(page.locator('#theme option[value=pop]')).to_have_text('Pop (black)')
    page.close()


@pytest.mark.parametrize('language,label', [('ja', 'アンインストールを開始'), ('en', 'Start uninstall')])
def test_uninstall_cancel_keeps_app(browser: Browser, distribution: tuple, language: str, label: str) -> None:
    runtime, url = distribution
    page = browser.new_page()
    page.goto(url + f'/settings?language={language}')
    page.locator('[name=language]').select_option(language)
    expect(page.locator('#uninstall-app')).to_have_text(label)
    expect(page.locator('#uninstall-help')).to_contain_text('zip' if language == 'ja' else 'ZIP')
    page.on('dialog', lambda dialog: dialog.dismiss())
    page.locator('#uninstall-app').click()
    assert not runtime.stopping
    page.close()


@pytest.mark.parametrize('p,winner', [(.5,None),(.47,None),(.53,None),(.8,None),(.2,None),
    (1,'1P'),(0,'2P'),(1,None)])
def test_mascot_stays_same_across_results(browser: Browser, distribution: tuple, p: float, winner: str | None) -> None:
    _, url = distribution
    page = browser.new_page(viewport={'width':1920,'height':1080})
    # 表情の既定ON化後も、明示OFFなら同じ画像を保つ。
    page.goto(url + '/?preview=sample&language=en&mascot=true&mascot_expressions=false')
    data = snapshot(p)
    if winner:
        data['evaluations']['practical']['source'] = 'confirmed_death'
    page.evaluate('renderSnapshot', data)
    expect(page.locator('#mascot img')).to_have_attribute('src', '/mascot/overlay.webp')
    page.wait_for_function("document.querySelector('#mascot img').naturalWidth === 320")
    expect(page.locator('#mascot img')).to_have_js_property('complete', True)
    page.evaluate("unavailable('接続待ち')")
    expect(page.locator('#mascot img')).to_have_attribute('src', '/mascot/overlay.webp')
    page.close()


@pytest.mark.parametrize('language,expected', [('ja','試合待ち'), ('en','Waiting for a match')])
def test_live_state_language(browser: Browser, distribution: tuple, language: str, expected: str) -> None:
    _, url = distribution
    page = browser.new_page()
    page.goto(url + f'/?preview=sample&language={language}')
    page.evaluate("renderSnapshot({display:{input_status:'no_puyo_screen'}})")
    expect(page.locator('#state')).to_have_text(expected)
    assert page.locator('#mascot').is_visible()
    page.close()


def test_all_themes_and_layouts_in_english(browser: Browser, distribution: tuple) -> None:
    _, url = distribution
    page = browser.new_page(viewport={'width':1920,'height':1080})
    errors: list[str] = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    for theme in ('classic','neon','arena','pop','noir'):
        for layout in ('card','ribbon','minimal','data','l'):
            page.goto(url + f'/?preview=sample&language=en&theme={theme}&layout={layout}&mascot=true&details=true')
            page.evaluate('renderSnapshot', snapshot(.73))
            expect(page.locator('#state')).to_have_text('Playing')
            visible = page.locator('#stage').inner_text()
            assert not re.search(r'[ぁ-んァ-ヶ一-龠]', visible), visible
            assert page.locator('#panel').evaluate('(e)=>e.scrollWidth<=e.clientWidth+1')
    assert errors == []
    page.close()



def assert_dynamic_language_messages(page: Page, url: str) -> None:
    """APIの全文メッセージと保存済み結果が言語切り替えに追従する。"""
    state = page.request.get(url + '/api/state').json()
    state['diagnostics']['input'] = {'message': '入力を確認しました / Input checked'}
    state['diagnostics']['scanning'] = False
    state['diagnostics']['calibration'] = None
    page.route('**/api/state', lambda route: route.fulfill(json=state))
    page.route('**/api/obs', lambda route: route.fulfill(json={
        'scene': 'Test scene', 'browser_sources': ['Overlay'],
        'warning': '手元で確認してください / Review locally'}))
    page.evaluate('appNotice()')
    page.locator('#obs-check').click()
    expect(page.locator('#obs-result')).to_contain_text('Test scene: Overlay')
    page.locator('#send-logs').click()
    expect(page.locator('#confirm-logs')).to_be_enabled()
    page.locator('#confirm-logs').click()
    expect(page.locator('#log-status')).to_contain_text('logs/reports')
    page.locator('#cancel-logs').click()
    for language in ('en', 'ja', 'en'):
        page.locator('[name=language]').select_option(language)
        english = language == 'en'
        expect(page.locator('#report-local-notice')).to_have_text(
            'The destination is being prepared for 0.6.0. Reviewed reports are saved locally (logs/reports).'
            if english else '0.6.0 では送信先を準備中です。確認した内容は手元 (logs/reports) に保存します')
        expect(page.locator('#log-status')).to_have_text(
            'Saved locally (logs/reports).' if english else '手元 (logs/reports) に保存しました')
        expect(page.locator('#input-reason')).to_have_text('Input checked' if english else '入力を確認しました')
        expect(page.locator('#obs-result')).to_have_text(
            'Test scene: Overlay\n' + ('Review locally' if english else '手元で確認してください'))


def test_settings_language_save_and_log_review(browser: Browser, distribution: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, url = distribution
    monkeypatch.setattr('src.phase_j.distribution_runtime.shortcuts', lambda *args: None)
    page = browser.new_page()
    page.goto(url + '/settings')
    page.locator('[name=language]').select_option('en')
    expect(page.locator('h1')).to_have_text('puyoZeus Live Display Settings')
    expect(page).to_have_title('puyoZeus Live Display Settings')
    assert not re.search(r'[ぁ-んァ-ヶ一-龠]', page.locator('main').inner_text()), page.locator('main').inner_text()
    page.locator('[name=mascot]').check()
    page.locator('[name=x]').fill('730')
    page.locator('[name=y]').fill('594')
    page.locator('#save-settings').click()
    expect(page.locator('#save-status')).to_have_text('Saved')
    assert runtime.settings['overlay']['mascot'] is True
    assert runtime.settings['overlay']['x'] == 730
    assert runtime.settings['overlay']['y'] == 594
    page.locator('#send-logs').click()
    expect(page.locator('#log-dialog h2')).to_have_text('puyoZeus Live — Review report')
    expect(page.locator('#log-preview')).to_contain_text('installation_id')
    assert not list(runtime.store.directory.glob('pending-*.json'))
    page.locator('#cancel-logs').click()
    expect(page.locator('#log-dialog')).not_to_be_visible()
    page.locator('[name=language]').select_option('ja')
    expect(page.locator('h1')).to_have_text('puyoZeus Live 表示設定')
    expect(page).to_have_title('puyoZeus Live 表示設定')
    assert_dynamic_language_messages(page, url)
    page.close()


@pytest.mark.parametrize('position,size', [('left','small'),('right','large'),('card','medium')])
def test_mascot_position_and_size(browser: Browser, distribution: tuple, position: str, size: str) -> None:
    _, url = distribution
    page = browser.new_page(viewport={'width':1920,'height':1080})
    page.goto(url + f'/?preview=sample&mascot=true&mascot_position={position}&mascot_size={size}')
    box = page.locator('#mascot').bounding_box()
    assert box['width'] == {'small':96,'medium':120,'large':224}[size]
    assert 0 <= box['x'] <= 1920-box['width'] and 0 <= box['y'] <= 1080-box['height']
    panel = page.locator('#panel').bounding_box()
    assert box['x'] == panel['x']+(panel['width']-box['width'])/2
    assert box['y'] == panel['y']+panel['height']+8
    page.close()


@pytest.mark.parametrize('reduce,enabled,count', [('reduce',True,0),('no-preference',False,0),('no-preference',True,0)])
def test_mascot_motion_respects_preferences(browser: Browser, distribution: tuple, reduce: str, enabled: bool, count: int) -> None:
    _, url = distribution
    page = browser.new_page(reduced_motion=reduce)
    page.goto(url + f'/?preview=sample&mascot=true&mascot_motion={str(enabled).lower()}')
    page.evaluate("()=>{window.pulses=0;document.getElementById('mascot').animate=()=>{window.pulses++;};}")
    page.evaluate('renderSnapshot', snapshot(.5))
    changed = snapshot(.8)
    changed['identity']['stream_seq'] += 1
    changed['identity']['match_state_seq'] += 1
    page.evaluate('renderSnapshot', changed)
    assert page.evaluate('window.pulses') == count
    page.close()


def test_os_default_and_url_override(browser: Browser, distribution: tuple) -> None:
    runtime, url = distribution
    runtime.os_language='en'
    page = browser.new_page(locale='ja-JP')
    page.goto(url + '/?preview=sample')
    assert page.locator('html').get_attribute('lang') == 'en'
    page.goto(url + '/?preview=sample&language=ja')
    assert page.locator('html').get_attribute('lang') == 'ja'
    page.close()


@pytest.mark.parametrize('display', [dict(visibility='hidden'), dict(status='integrity_fault')])
def test_hidden_probability_does_not_drive_mascot(browser: Browser, distribution: tuple, display: dict) -> None:
    _, url = distribution
    page = browser.new_page()
    page.goto(url + '/?preview=sample&mascot=true')
    data = snapshot(.9)
    data['display'].update(display)
    page.evaluate('renderSnapshot', data)
    expect(page.locator('#mascot img')).to_have_attribute('src', '/mascot/50.webp')
    page.close()
