"""テーマの安全領域・実データ表示・演出の状態遷移を実ブラウザで確認する。"""
from __future__ import annotations

import io
from itertools import product

from PIL import Image
from playwright.sync_api import Browser, Page
import pytest

from scripts.render_overlay_design import new_page, snapshot
from src.phase_j.overlay_config import THEMES, validate_overlay
from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_overlay_options_browser import server


@pytest.mark.parametrize('theme', THEMES)
def test_theme_config_and_safe_layouts(browser: Browser, theme: str) -> None:
    assert validate_overlay({'theme': theme}) == {'theme': theme}
    for layout, value in product(['card', 'ribbon', 'minimal', 'data'], [0., 1.]):
        page = new_page(browser, f'{layout}&theme={theme}')
        page.evaluate('renderSnapshot', snapshot(value))
        assert page.locator('#stage').get_attribute('data-theme') == 'pop'
        box = page.locator('#panel').bounding_box()
        assert 704 <= box['x'] and box['x']+box['width'] <= 1216
        assert box['y']+box['height'] <= (1060 if layout == 'data' else 932)
        for selector in ['.num', '.meta']:
            assert page.locator(selector).first.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1'), (theme, layout)
        page.close()


@pytest.mark.parametrize('theme', THEMES[1:])
def test_theme_l_game_pixels_and_options(browser: Browser, theme: str) -> None:
    for corner, scale, band in product(['top-left', 'bottom-right'], [70, 85], ['true', 'false']):
        page = new_page(browser, f'l&theme={theme}&corner={corner}&scale={scale}&band={band}&font=large')
        page.evaluate('renderSnapshot', snapshot(1.))
        g = page.evaluate('OverlayOptions.geometry(overlayOptions)')
        pixels = Image.open(io.BytesIO(page.screenshot(omit_background=True)))
        assert pixels.getchannel('A').crop((g['x'], g['y'], g['x']+g['width'], g['y']+g['height'])).getextrema() == (0, 0)
        for selector in ['#panel', '#band', '.num', '.meta']:
            node = page.locator(selector).first
            if node.is_visible():
                assert node.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1 && e.scrollHeight<=e.clientHeight+1'), (theme, corner, scale, band, selector)
        page.close()


def motion_page(browser: Browser) -> Page:
    page = new_page(browser, 'card&theme=neon')
    page.emulate_media(reduced_motion='no-preference')
    page.clock.install(time=0)
    page.clock.pause_at(100)
    return page


def test_roll_swing_cooldown_and_result_once(browser: Browser) -> None:
    page = motion_page(browser)
    page.evaluate('renderSnapshot', snapshot(.5, 0))
    page.evaluate('renderSnapshot', snapshot(.8, 1000))
    assert page.locator('#stage').get_attribute('data-motion') is None
    page.clock.run_for(180)
    assert page.locator('#p1').inner_text() == '80'
    page.clock.run_for(2000)
    assert page.locator('#p1').inner_text() == '80'
    result = snapshot(1., 2000)
    result['evaluations']['practical']['source'] = 'unavoidable_death'
    page.evaluate('renderSnapshot', result)
    assert page.locator('#stage').get_attribute('data-winner') is None
    result['evaluations']['practical']['source'] = 'confirmed_death'
    page.evaluate('renderSnapshot', result)
    assert page.locator('#stage').get_attribute('data-winner') == '1P'
    page.clock.run_for(2000)
    page.evaluate('renderSnapshot', result)
    assert page.locator('#stage').get_attribute('data-motion') is None
    assert page.locator('#p1').inner_text() == '100'
    result['display']['match_phase'] = 'between_matches'
    page.evaluate('renderSnapshot', result)
    assert page.locator('#stage').get_attribute('data-winner') is None
    assert page.locator('#p1').inner_text() == '—'
    page.close()


def test_reduced_motion_and_new_match_reset(browser: Browser) -> None:
    page = new_page(browser, 'card&theme=pop')
    page.evaluate('renderSnapshot', snapshot(.2, 0))
    page.evaluate('renderSnapshot', snapshot(.8, 1000))
    assert page.locator('#stage').get_attribute('data-motion') is None
    assert page.locator('#p1').inner_text() == '80'
    page.emulate_media(reduced_motion='no-preference')
    data = snapshot(.1, 0)
    data['identity']['match_id'] = 'next'
    page.evaluate('renderSnapshot', data)
    assert page.locator('#stage').get_attribute('data-motion') is None
    assert page.locator('#p1').inner_text() == '10'
    page.close()


def test_small_updates_accumulate_but_hold_does_not_flash(browser: Browser) -> None:
    page = motion_page(browser)
    for index in range(8):
        page.evaluate('renderSnapshot', snapshot(.5+index*.02, index*100))
        page.clock.run_for(100)
    assert page.locator('#stage').get_attribute('data-motion') is None
    page.clock.run_for(2000)
    hold = snapshot(.2, 3000)
    hold['display']['status'] = 'hold'
    page.evaluate('renderSnapshot', hold)
    assert page.locator('#stage').get_attribute('data-motion') is None
    page.close()


def test_theme_settings_http_and_fallback(browser: Browser, server: object) -> None:
    page = browser.new_page()
    origin = f'http://127.0.0.1:{server.address()[1]}'
    page.goto(origin+'/settings')
    page.locator('[name=theme]').select_option('pop-cream')
    assert 'theme=pop-cream' in page.locator('#url').input_value()
    frame = page.frame_locator('#overlay')
    from playwright.sync_api import expect
    expect(frame.locator('#stage')).to_have_attribute('data-pop', 'cream')
    for path in ['/overlay_themes.css', '/overlay_themes.js']:
        assert page.request.get(origin+path).status == 200
    page.goto(origin+'/?preview=sample&theme=unknown')
    assert page.locator('#stage').get_attribute('data-theme') == 'pop'
    page.close()
