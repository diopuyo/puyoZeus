"""初回案内・追加配置で、入力なしや保留を勝率と取り違えない。"""
from __future__ import annotations
from typing import Any

import pytest
from playwright.sync_api import Browser, expect
from scripts.render_overlay_design import new_page, snapshot
from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_distribution_app import distribution


@pytest.mark.parametrize('layout', ['ribbon', 'vertical', 'l-left'])
@pytest.mark.parametrize('theme', ['battle-a', 'esports', 'normal'])
def test_wait_and_hold_preserve_probability(browser: Browser, layout: str, theme: str) -> None:
    page = new_page(browser, layout+'&theme='+theme)
    page.evaluate('renderSnapshot', snapshot(.68))
    expect(page.locator('#p1')).to_have_text('68')
    expect(page.locator('#battle-speech')).to_be_hidden()
    held = snapshot(.68, 1000)
    held['display']['status'] = 'hold'
    page.evaluate('renderSnapshot', held)
    expect(page.locator('#stage')).to_have_attribute('data-state', 'hold')
    expect(page.locator('#p1')).to_have_text('68')
    held['display']['input_status'] = 'no_puyo_screen'
    page.evaluate('renderSnapshot', held)
    expect(page.locator('#stage')).to_have_attribute('data-state', 'wait')
    expect(page.locator('#p1')).to_have_text('—')
    expect(page.locator('#trace')).to_have_attribute('d', '')
    expect(page.locator('#battle-speech')).to_have_text('次の試合まだかな')
    expect(page.locator('#battle-speech')).to_be_visible()
    page.clock.install()
    page.clock.fast_forward(60000)
    expect(page.locator('#battle-speech')).to_be_visible()
    held['display']['input_status'] = 'verifying'
    held['display']['match_phase'] = 'between_matches'
    page.evaluate('renderSnapshot', held)
    expect(page.locator('#battle-speech')).to_be_hidden()
    page.close()


def test_vertical_bar_is_visible_inside_panel(browser: Browser) -> None:
    page = new_page(browser, 'vertical&theme=battle-a')
    page.evaluate('renderSnapshot', snapshot(.68))
    panel, bar = (page.locator(selector).bounding_box() for selector in ['#panel', '#bar'])
    assert panel and bar
    assert bar['width'] > bar['height']  # 見本のサイド柱もゲージは横向き。
    assert panel['x'] <= bar['x'] and bar['x']+bar['width'] <= panel['x']+panel['width']
    assert panel['y'] <= bar['y'] and bar['y']+bar['height'] <= panel['y']+panel['height']
    page.close()


@pytest.mark.parametrize('scene,title', [('no_puyo_screen', '試合待ち'), ('ready', '認識中'),
                                        ('verifying', '入力を確かめています'), ('overlay_feedback', '入力の設定を確認')])
def test_setup_status_and_missing_camera(browser: Browser, distribution: tuple, scene: str, title: str) -> None:
    runtime, url = distribution
    page = browser.new_page()
    state: dict[str, Any] = dict(settings=runtime.settings, os_language='ja', root='test', url=url,
                                running=True, error=None, diagnostics=dict(calibration=scene))
    page.route('**/api/state', lambda route: route.fulfill(json=state))
    page.route('**/api/discover', lambda route: route.fulfill(json=dict(devices=[], selected=None, message='入力なし')))
    page.goto(url+'/setup')
    expect(page.locator('#runtime-title')).to_have_text(title)
    expect(page.locator('#camera-help')).to_be_visible()
    expect(page.locator('#test')).to_be_disabled()
    page.locator('#layout').select_option('vertical')
    # PROMPT 6: 配置の変更で選択済みのテーマを変えない。
    expect(page.locator('#theme')).to_have_value('pop')
    page.locator('#layout').select_option('l-left')
    expect(page.locator('#preview-placement')).to_contain_text('1536 × 864 / (384, 0)')
    page.close()


def test_setup_stop_and_error_next_action(browser: Browser, distribution: tuple) -> None:
    runtime, url = distribution
    page = browser.new_page()
    state = dict(settings=runtime.settings, os_language='ja', root='test', url=url,
                 running=False, error='runtime', diagnostics={})
    page.route('**/api/state', lambda route: route.fulfill(json=state))
    page.route('**/api/discover', lambda route: route.fulfill(json=dict(devices=[], selected=None, message='入力なし')))
    page.route('**/api/stop', lambda route: route.fulfill(json=dict(stopped=True)))
    page.goto(url+'/setup')
    expect(page.locator('#runtime-title')).to_have_text('起動・実行エラー')
    expect(page.locator('#retry')).to_be_visible()
    page.locator('#stop-setup').click()
    expect(page.locator('#runtime-title')).to_have_text('終了しました')
    expect(page.locator('#start')).to_be_disabled()
    page.close()
