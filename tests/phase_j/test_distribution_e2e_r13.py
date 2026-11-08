"""保存した公開記録を SSE 受信口へ流し、カードと全配置の表示を確認する。"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, expect

from scripts.render_overlay_design import MOCK_STREAM
from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_distribution_app import distribution

LAYOUTS = ('center', 'minimal', 'card', 'vertical', 'l-top-left', 'l-top-right',
           'l-bottom-left', 'l-bottom-right')
ROWS = json.loads((Path(__file__).parent / 'fixtures/card_publications.json').read_text('utf-8'))['rows']


def overlaps(a: dict, b: dict) -> bool:
    return (a['x'] < b['x'] + b['width'] and b['x'] < a['x'] + a['width']
            and a['y'] < b['y'] + b['height'] and b['y'] < a['y'] + a['height'])


@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('theme', ('pop', 'pop-cream'))
def test_recorded_sse_and_win(browser: Browser, tmp_path: Path, layout: str, theme: str, distribution: tuple) -> None:
    runtime, url = distribution
    runtime.configure({'overlay': {'layout': layout, 'theme': theme, 'mascot_eyes': False}})
    page = browser.new_page(viewport=dict(width=1920, height=1080), reduced_motion='reduce')
    page.add_init_script(MOCK_STREAM)
    page.goto(url + '/?preview=live')
    page.wait_for_function("document.querySelector('#mascot img')?.naturalWidth>0")
    errors: list[str] = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.evaluate('(rows)=>rows.forEach(data=>testStream.emit(data))', ROWS[:-1])
    page.evaluate('document.fonts.ready')
    probability = ROWS[-2]['evaluations']['practical']['p1_win_probability']
    expect(page.locator('#p1')).to_have_text(str(int(probability * 100 + .5)))
    assert page.evaluate("Math.abs(Number(el('fill').style.transform.slice(7,-1))-shown)<1e-8")
    output = Path(os.environ.get('CARD_SCREENSHOTS_PATH', str(tmp_path)))
    output.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(output / f'{layout}_{theme}_live.png'), omit_background=True)
    if layout == 'card':
        check_card(page)
    page.evaluate('data=>testStream.emit(data)', ROWS[-1])
    expect(page.locator('#stage')).to_have_attribute('data-state', 'result')
    expect(page.locator('.tab')).to_be_visible()
    assert 'WIN!' in page.locator('.tab').inner_text()
    for selector in ('#p1', '#p2'):
        assert not overlaps(page.locator('.tab').bounding_box(), page.locator(selector).bounding_box())
    page.screenshot(path=str(output / f'{layout}_{theme}_win.png'), omit_background=True)
    if layout == 'card':
        next_match = deepcopy(ROWS[0])
        next_match['identity']['match_id'] = 'next'
        page.evaluate('data=>testStream.emit(data)', next_match)
        assert page.evaluate('fullHistory.length') == 1
    assert not errors
    page.close()


def check_card(page: Page) -> None:
    panel = page.locator('#panel').bounding_box()
    assert 700 <= panel['x'] and panel['x'] + panel['width'] <= 1220
    assert panel['y'] + panel['height'] <= 932
    graph = page.locator('#whole').bounding_box()
    assert graph['width'] > graph['height'] > 0
    assert page.locator('#whole-trace').get_attribute('d').count('L') > 1
    assert page.evaluate('fullHistory.at(-1).time-fullStart') > 30000
    assert page.locator('#whole .guide').evaluate("e=>getComputedStyle(e).stroke!=='none' && Number(getComputedStyle(e).opacity)>0 && e.getBBox().width>0")
    assert page.locator('#mascot').bounding_box()['y'] < panel['y']
    for selector in ('.plate.one', '#bar', '.plate.two'):
        rect = page.locator(selector).bounding_box()
        assert rect['y'] + rect['height'] <= graph['y']
    assert not overlaps(page.locator('#bar').bounding_box(), page.locator('#p1').bounding_box())
    assert not overlaps(page.locator('#bar').bounding_box(), page.locator('#p2').bounding_box())


@pytest.mark.parametrize('language', ('ja', 'en'))
def test_appearance_and_status_words(browser: Browser, distribution: tuple, language: str) -> None:
    runtime, url = distribution
    runtime.configure({'overlay': {'language': language, 'layout': 'card'}})
    page = browser.new_page()
    page.goto(url + '/settings')
    page.locator('#tab-appearance').click()
    expect(page.locator('[name=layout]')).to_have_value('card')
    assert page.locator('[name=layout] option').evaluate_all('(es)=>es.map(e=>e.value)') == list(LAYOUTS)
    for selector in ('#url', '#copy', '#json', '#placement'):
        expect(page.locator(selector)).to_be_hidden()
    expect(page.locator('#placement-hint')).to_be_hidden()
    page.locator('[name=layout]').select_option('l-top-left')
    expect(page.locator('#placement-hint')).to_be_visible()
    page.evaluate("window.lastDiagnostics={calibration:'ready',delay_ms:50,fps:30};renderDiagnostics()")
    text = page.locator('#live-diagnostics').text_content()
    assert ('画面の調整: 済み' if language == 'ja' else 'Screen adjustment: Complete') in text
    assert ('約 0.05 秒' if language == 'ja' else 'About 0.05 seconds') in text
    page.close()
