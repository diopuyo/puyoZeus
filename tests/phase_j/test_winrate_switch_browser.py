"""試作の線1を実ブラウザで描く。"""
from pathlib import Path

import pytest

pytest.importorskip('playwright.sync_api')
from playwright.sync_api import sync_playwright
from scripts.render_overlay_design import new_page, snapshot

ROOT = Path(__file__).resolve().parents[2]


def test_line1_graph_and_number() -> None:
    with sync_playwright() as runner:
        browser = runner.chromium.launch(channel='msedge', headless=True)
        page = new_page(browser)
        page.add_script_tag(path=str(ROOT/'src/phase_j/overlay_winrate_switch.js'))
        for index, value in enumerate((.4, .7, .3)):
            data = snapshot(.4, 30000+index*1000)
            data['evaluations']['winrate_switch'] = dict(line1=value, b3=.4, source1='v4.9f')
            page.evaluate('data => renderSnapshot(data)', data)
        page.wait_for_timeout(250)
        assert '撃ち合い+B3' == page.locator('#heading').inner_text()
        assert '30' in page.locator('#p1').inner_text()
        assert page.locator('#whole-trace').get_attribute('d').count('L') == 2
        assert 'B3: 40.0%' in page.locator('#heading').get_attribute('title')
        browser.close()
