"""柱の縦履歴は30秒内の点を枠の高さ全体へ配置する。"""
import pytest
from playwright.sync_api import Browser

from scripts.render_overlay_design import new_page, snapshot
from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_overlay_l_vertical_browser import coordinates

GRAPH_HEIGHT = 54
HISTORY_WINDOW_MS = 30000
FRAME_ADVANCE_MS = 400
PROBABILITIES = (.35, .65, .45)


@pytest.mark.parametrize('layout', ('center', 'vertical'))
@pytest.mark.parametrize('theme', ('pop', 'pop-cream'))
@pytest.mark.parametrize('mode', ('probability', 'evaluation'))
@pytest.mark.parametrize('autoscale', ('0', '1'))
@pytest.mark.parametrize('duration_ms', (20000, 30000, 40000))
def test_vertical_fills_height(browser: Browser, layout: str, theme: str,
                               mode: str, autoscale: str, duration_ms: int) -> None:
    page = new_page(browser, f'{layout}&theme={theme}&value_display={mode}'
                    f'&graph_autoscale={autoscale}')
    page.clock.install(time=0)
    for probability, time_ms in zip(PROBABILITIES, (0, duration_ms // 2, duration_ms)):
        page.evaluate('renderSnapshot', snapshot(probability, time_ms))
        page.clock.run_for(FRAME_ADVANCE_MS)
    points = coordinates(page, '#trace')
    expected_y = ([GRAPH_HEIGHT, 0] if duration_ms > HISTORY_WINDOW_MS
                  else [GRAPH_HEIGHT, GRAPH_HEIGHT / 2, 0])
    assert points[1::2] == expected_y, '30秒内の履歴が下から上へ枠全体を使う'
    assert points[-4] > points[-2], '右ほど1P有利'
    for selector in ('#trend .history-one', '#trend .history-two'):
        area_y = coordinates(page, selector)[1::2]
        assert min(area_y) == 0 and max(area_y) == GRAPH_HEIGHT
    label = page.locator('#trend').get_attribute('aria-label')
    assert '右ほど1P有利' in label and '下ほど過去' in label
    assert ('評価値' if mode == 'evaluation' else '勝率') in label
    page.close()
