"""採用した表示の値なし・保留・決着と設定の互換性を検査する。"""
from __future__ import annotations

import pytest
from playwright.sync_api import Browser, expect
from scripts.render_overlay_design import new_page, snapshot
from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_distribution_app import distribution

THEMES = ['pop', 'pop-cream']
LAYOUTS = ['center', 'minimal', 'card', 'vertical', 'l-left']
GRAPH_HEIGHT = 54


@pytest.mark.parametrize('theme', THEMES)
@pytest.mark.parametrize('layout', LAYOUTS)
def test_adopted_states(browser: Browser, theme: str, layout: str) -> None:
    page = new_page(browser, layout+'&theme='+theme)
    page.clock.install()
    page.clock.pause_at(page.evaluate('Date.now()'))
    page.evaluate('renderSnapshot', snapshot(.68))
    expect(page.locator('#p1')).to_have_text('68')
    expect(page.locator('#p2')).to_have_text('32')
    expect(page.locator('.meta')).to_be_hidden()
    expect(page.locator('.name').first).to_be_hidden()
    held = snapshot(.68, 31000)
    held['display']['status'] = 'hold'
    page.evaluate('renderSnapshot', held)
    page.clock.run_for(599)
    assert page.locator('#stage').get_attribute('data-state') == 'live'
    page.clock.run_for(1)
    assert page.locator('#stage').get_attribute('data-state') == 'hold'
    page.evaluate('renderSnapshot', snapshot(.68, 32000))
    page.clock.run_for(799)
    assert page.locator('#stage').get_attribute('data-state') == 'hold'
    page.clock.run_for(1)
    assert page.locator('#stage').get_attribute('data-state') == 'live'
    held['terminal'] = dict(state='confirmed', winner='2P')
    page.evaluate('renderSnapshot', held)
    expect(page.locator('#result-text')).to_have_text('WIN! ▶')
    expect(page.locator('#p1')).to_have_text('0')
    held['display']['input_status'] = 'no_puyo_screen'
    page.evaluate('renderSnapshot', held)
    expect(page.locator('#battle-speech')).to_have_text('次の試合まだかな')
    expect(page.locator('#p1')).to_have_text('—')
    page.close()


@pytest.mark.parametrize('old,new', [('battle-a','pop'),('esports','pop'),('normal','pop'),('stylish','pop'),('plain','pop')])
def test_old_theme_urls(browser: Browser, old: str, new: str) -> None:
    page = new_page(browser, 'ribbon&theme='+old)
    assert page.evaluate('overlayOptions.theme') == new
    page.close()


def test_eyes_can_stop_without_disabling_dopamine(browser: Browser, distribution: tuple) -> None:
    _, url = distribution
    page = browser.new_page(viewport={'width':1920,'height':1080})
    page.goto(url+'/?preview=sample&mascot_eyes=false')
    page.evaluate('renderSnapshot', snapshot(.95))
    expect(page.locator('#mascot img')).to_have_attribute('src', '/mascot/battle-still.webp')
    assert page.locator('#mascot img').evaluate('e=>e.getAnimations().length') == 1
    assert page.locator('#mascot').evaluate('e=>e.getAnimations().length') == 0
    page.close()

@pytest.mark.parametrize('theme', ['pop', 'pop-cream'])
@pytest.mark.parametrize('layout', ['center', 'vertical'])
def test_vertical_history_is_fixed(browser: Browser, theme: str, layout: str) -> None:
    page = new_page(browser, layout+'&theme='+theme+'&vgraph=0&graph_autoscale=0')
    for seconds, probability in [(1, .2), (2, .8), (3, .4)]:
        page.evaluate('renderSnapshot', snapshot(probability, seconds*1000))
    before = page.locator('#trace').get_attribute('d')
    page.evaluate('renderSnapshot', snapshot(.6, 4000))
    after = page.locator('#trace').get_attribute('d')
    import re
    points = lambda value: [float(x) for x in re.findall(r'-?\d+\.?\d*', value)]
    a, b = points(before), points(after)
    assert a[::2] == b[:-2:2]
    # 値の横座標を保ち、追加後も時刻順に縦枠全体へ配置する。
    assert a[1::2] == [GRAPH_HEIGHT, GRAPH_HEIGHT / 2, 0]
    assert b[1::2] == [GRAPH_HEIGHT, GRAPH_HEIGHT * 2 / 3, GRAPH_HEIGHT / 3, 0]
    assert page.locator('#stage').get_attribute('data-vgraph') == '1'
    assert page.locator('#trend .history-one').get_attribute('d')
    expect(page.locator('#trend .mid-top')).to_be_hidden()
    assert page.locator('#trace').evaluate('e=>getComputedStyle(e).strokeLinejoin') == 'round'
    expect(page.locator('#trend .mid-bottom')).to_be_hidden()
    assert float(page.locator('#tip').get_attribute('cx')) == pytest.approx(4+.6*404)
    assert page.evaluate("getComputedStyle(el('panel'),'::before').display") == 'none'
    assert page.locator('#panel').bounding_box()['height'] == (560 if layout == 'center' else 610)
    page.close()


@pytest.mark.parametrize('query', ['', '&theme=pop-cream', '&vgraph=1', '&theme=plain&vgraph=1'])
def test_legacy_ribbon_uses_center(browser: Browser, query: str) -> None:
    page = new_page(browser, 'ribbon'+query)
    assert page.locator('#stage').get_attribute('data-vgraph') == '1'
    assert page.evaluate('OverlayOptions.resolve().layout') == 'center'
    page.close()

@pytest.mark.parametrize('path,theme_selector,layout_selector,preview', [
    ('/settings', '[name=theme]', '[name=layout]', '#overlay'),
    ('/setup', '[name=theme]', '[name=layout]', '#overlay'),
])
def test_two_designs_four_layouts(browser: Browser, distribution: tuple, path: str,
                                 theme_selector: str, layout_selector: str, preview: str) -> None:
    ui_layouts = ['center', 'minimal', 'card', 'vertical', 'l-top-left', 'l-top-right',
                  'l-bottom-left', 'l-bottom-right']
    _, url = distribution
    page = browser.new_page(viewport={'width':1500,'height':1100})
    page.goto(url+path)
    page.locator('#tab-appearance').click()
    themes = page.locator(theme_selector)
    layouts = page.locator(layout_selector)
    expect(themes).to_have_value('pop')
    expect(layouts).to_have_value('center')
    assert themes.locator('option').evaluate_all('(nodes)=>nodes.map(e=>e.value)') == THEMES
    assert layouts.locator('option').evaluate_all('(nodes)=>nodes.map(e=>e.value)') == ui_layouts
    assert page.locator('[name=vgraph]').count() == 0
    for theme in THEMES:
        themes.select_option(theme)
        for layout in ui_layouts:
            layouts.select_option(layout)
            # 連続選択で前の iframe を検査しないよう、指定 URL の読み込み完了を待つ。
            page.wait_for_function('''selector=>{const frame=document.querySelector(selector);
                return frame.contentDocument?.readyState==='complete'
                    && frame.contentWindow.location.href===frame.src
                    && Boolean(frame.contentDocument.querySelector('#stage')?.dataset.layout);
            }''', arg=preview)
            stage = page.frame_locator(preview).locator('#stage')
            expect(stage).to_have_attribute('data-pop', 'cream' if theme == 'pop-cream' else 'black')
            expect(stage).to_have_attribute('data-layout', 'l' if layout.startswith('l-') else {'center':'side','vertical':'side','minimal':'mini','card':'ribbon'}[layout])
            expect(stage).to_have_attribute('data-vgraph', '1' if layout.startswith('l-') or layout in ['center','vertical'] else '0')
    page.close()
