"""日本語の設定画面と旧URL・新URLの描画を実ブラウザで確かめる。"""
import json
import os
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, Route, expect

from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_distribution_browser import select_language, open_appearance

LAYOUTS = ('center', 'minimal', 'card', 'vertical', 'l-top-left', 'l-top-right',
           'l-bottom-left', 'l-bottom-right')
REMOVED = ('x', 'y', 'corner', 'mascot_flip_right', 'mascot_band_edges',
           'numbers', 'recent', 'full', 'mascot_motion', 'mascot_expressions')


def capture(page: Page, name: str) -> bytes:
    page.wait_for_function('''() => [...document.querySelectorAll('iframe')].every(frame =>
      frame.contentDocument?.readyState === 'complete' && frame.contentWindow.location.href === frame.src
      && frame.contentDocument.querySelector('#stage')?.dataset.layout)''')
    for frame in page.frames:
        frame.wait_for_function('''() => [...document.querySelectorAll('#mascot img')].every(
          img => img.complete && img.naturalWidth > 0)''')
    directory = os.environ.get('SETTINGS_SCREENSHOTS_PATH')
    path = None
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = str(Path(directory) / (name + '.png'))
    return page.screenshot(path=path, full_page=True, animations='disabled')


def settings_page(browser: Browser, url: str) -> Page:
    page = browser.new_page(viewport=dict(width=1100, height=800), locale='ja-JP')
    page.goto(url + '/settings')
    select_language(page, 'ja')
    expect(page.locator('#toggle-runtime')).to_be_visible()
    return page


def test_controls_follow_mode_eyes_and_layout(browser: Browser, distribution: tuple) -> None:
    _, url = distribution
    page = settings_page(browser, url)
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    open_appearance(page)
    assert page.locator('[name=layout] option').evaluate_all('(es)=>es.map(e=>e.value)') == list(LAYOUTS)
    for name in REMOVED:
        assert page.locator(f'[name="{name}"]').count() == 0
    for enabled, eyes in ((False, False), (False, True), (True, False), (True, True)):
        page.locator('[name=mascot_dopamine_mode]').set_checked(enabled)
        page.locator('[name=mascot_eyes]').set_checked(eyes)
        assert page.locator('[name=mascot_dopamine]').is_enabled() is enabled
        assert page.locator('[name=mascot_speed]').is_enabled() is eyes
        for layout in ('center', 'l-bottom-right'):
            page.locator('[name=layout]').select_option(layout)
            assert page.locator('[name=scale]').is_enabled() is layout.startswith('l-')
            assert page.locator('[name=band]').is_enabled() is layout.startswith('l-')
            capture(page, f'settings_mode_{enabled}_eyes_{eyes}_{layout}')
            resolved = page.frames[1].evaluate('overlayOptions')
            assert resolved['mascot_motion'] is enabled and resolved['mascot_expressions'] is enabled
        options = page.evaluate('readControls()')
        assert not set(REMOVED).intersection(options)
    assert not errors
    page.close()


def test_header_start_stop_restart(browser: Browser, distribution: tuple) -> None:
    _, url = distribution
    page = settings_page(browser, url)
    initial = page.request.get(url + '/api/state').json()
    running = False
    calls = []
    def state(route: Route) -> None:
        route.fulfill(json={**initial, 'running': running})
    def action(route: Route) -> None:
        nonlocal running
        calls.append(route.request.url.rsplit('/', 1)[-1])
        running = calls[-1] == 'start'
        route.fulfill(json=dict(started=running, stopped=not running))
    page.route('**/api/state', state)
    page.route('**/api/start', action)
    page.route('**/api/pause', action)
    button = page.locator('#toggle-runtime')
    expect(button).to_have_text('解析を始める')
    assert calls == []
    capture(page, 'settings_before_start')
    button.click()
    expect(button).to_have_text('解析を止める')
    capture(page, 'settings_running')
    button.click()
    expect(button).to_have_text('解析を始める')
    button.click()
    expect(button).to_have_text('解析を止める')
    assert calls == ['start', 'pause', 'start']
    page.close()


def test_setup_saves_and_moves_to_settings(browser: Browser, distribution: tuple,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, url = distribution
    monkeypatch.setattr('src.phase_j.distribution_runtime.shortcuts', lambda *args: None)
    page = browser.new_page(viewport=dict(width=1100, height=800), locale='ja-JP')
    device = dict(index=0, name='OBS Virtual Camera')
    page.route('**/api/discover', lambda route: route.fulfill(json=dict(
        devices=[device], selected=device, message='入力を確認しました')))
    page.goto(url + '/setup')
    expect(page).to_have_url(url + '/settings')
    select_language(page, 'ja')
    open_appearance(page)
    assert page.locator('[name=layout] option').evaluate_all('(es)=>es.map(e=>e.value)') == list(LAYOUTS)
    page.locator('[name=layout]').select_option('l-bottom-right')
    page.locator('[name=mascot_dopamine_mode]').uncheck()
    capture(page, 'setup_full')
    expect(page.locator('#save-status')).to_have_text('保存しました')
    expect(page).to_have_url(url + '/settings')
    expect(page.locator('#toggle-runtime')).to_have_text('解析を始める')
    assert not runtime.settings['completed'] and runtime.child is None
    assert runtime.settings['overlay']['layout'] == 'l-bottom-right'
    assert runtime.settings['overlay']['mascot_dopamine_mode'] is False
    page.close()


@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('theme', ['pop', 'pop-cream'])
def test_legacy_and_new_urls_have_identical_pixels(browser: Browser, distribution: tuple,
                                                 layout: str, theme: str) -> None:
    _, url = distribution
    page = browser.new_page(viewport=dict(width=1920, height=1080), locale='ja-JP')
    legacy = 'layout=l&corner=' + layout[2:] if layout.startswith('l-') else 'layout=' + layout
    common = f'&theme={theme}&preview=sample&language=ja&mascot_eyes=false'
    queries = [legacy + '&x=123&y=456&mascot_flip_right=true&mascot_band_edges=20,40,60,80'
               '&numbers=false&recent=false&full=true&mascot_motion=false&mascot_expressions=false',
               'layout=' + layout + '&mascot_dopamine_mode=false']
    images = []
    for kind, query in zip(('old', 'new'), queries):
        page.goto(url + '/?' + query + common)
        page.wait_for_function('typeof renderSnapshot === "function"')
        page.evaluate('''() => {
          renderSnapshot({identity:{session_id:'sample',match_id:'sample'},
            timing:{source_available_ms:0},display:{input_status:'ready',match_phase:'playing'},
            evaluations:{practical:{availability:'available',p1_win_probability:.68,
              source:'static',calibration_id:'b3-stage2-display'}}});
        }''')
        page.evaluate('document.fonts.ready')
        is_l = layout.startswith('l-')
        assert page.locator('#whole').is_visible() is (is_l or layout == 'card')
        assert page.locator('.trend').is_visible() is (layout not in ('minimal', 'card'))
        if layout not in ('minimal', 'card'):
            graph, panel = (page.locator(selector).bounding_box() for selector in ('.trend', '#panel'))
            assert panel['x'] <= graph['x'] and panel['y'] <= graph['y']
            assert graph['x'] + graph['width'] <= panel['x'] + panel['width']
            assert graph['y'] + graph['height'] <= panel['y'] + panel['height']
        options = page.evaluate('overlayOptions')
        assert options['numbers'] and options['recent'] is (layout not in ('minimal', 'card'))
        assert options['full'] is (is_l or layout == 'card')
        images.append(capture(page, f'overlay_{layout}_{theme}_{kind}'))
    assert images[0] == images[1], f'{layout}/{theme}'
    page.close()
