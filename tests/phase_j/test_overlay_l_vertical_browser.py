"""L字の直近の線は縦、全試合の線は横であることを確認する。"""
import re

import pytest
from playwright.sync_api import Browser, Page

from scripts.render_overlay_design import new_page, snapshot
from tests.phase_j.test_overlay_browser import browser

LAYOUTS = ('l', 'l-left', 'l-right', 'l-top-left', 'l-top-right',
           'l-bottom-left', 'l-bottom-right')
THEMES = ('pop', 'pop-cream', 'battle-a', 'esports', 'normal', 'classic',
          'noir', 'neon', 'arena', 'stylish', 'plain')
SAMPLES = ((.35, 1000), (.65, 11000), (.45, 21000))
VIEWPORT_WIDTH, VIEWPORT_HEIGHT = 1920, 1080
FRAME_ADVANCE_MS = 400
GRAPH_HEIGHT = 54


def coordinates(page: Page, selector: str) -> list[float]:
    """SVGの折れ線を座標の配列へ戻す。"""
    return list(map(float, re.findall(r'-?\d+\.?\d*',
                                     page.locator(selector).get_attribute('d') or '')))


@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('theme', THEMES)
@pytest.mark.parametrize('mode', ('probability', 'evaluation'))
@pytest.mark.parametrize('autoscale', ('0', '1'))
def test_l_vertical_history(browser: Browser, layout: str, theme: str,
                            mode: str, autoscale: str) -> None:
    page = new_page(browser, f'{layout}&theme={theme}&value_display={mode}'
                    f'&graph_autoscale={autoscale}')
    page.clock.install(time=0)
    for probability, time_ms in SAMPLES:
        page.evaluate('renderSnapshot', snapshot(probability, time_ms))
        page.clock.run_for(FRAME_ADVANCE_MS)
    assert page.locator('#stage').get_attribute('data-vgraph') == '1'
    x0, y0, x1, y1, x2, y2 = coordinates(page, '#trace')
    assert x0 < x2 < x1, '右ほど1P有利'
    assert y0 > y1 > y2, '下ほど過去'
    assert (y0, y1, y2) == (GRAPH_HEIGHT, GRAPH_HEIGHT / 2, 0), '履歴が縦枠の上下端まで届く'
    for selector in ('#trend .history-one', '#trend .history-two'):
        area_y = coordinates(page, selector)[1::2]
        assert min(area_y) == 0 and max(area_y) == GRAPH_HEIGHT
    label = page.locator('#trend').get_attribute('aria-label')
    assert '右ほど1P有利' in label and '下ほど過去' in label
    assert ('評価値' if mode == 'evaluation' else '勝率') in label
    a0, b0, a1, b1, a2, b2 = coordinates(page, '#whole-trace')
    assert a0 < a1 < a2 and b0 > b2 > b1, '全試合は横向き'
    for selector in ('#trend', '#whole'):
        box = page.locator(selector).bounding_box()
        assert box and box['width'] > 0 and box['height'] > 0
        assert 0 <= box['x'] < box['x'] + box['width'] <= VIEWPORT_WIDTH
        assert 0 <= box['y'] < box['y'] + box['height'] <= VIEWPORT_HEIGHT
    assert page.locator('#band #whole').count() == 1
    page.close()
