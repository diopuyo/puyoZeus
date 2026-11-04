"""L字の非重複、設定ページ、HTTP経由の既定値を実ブラウザで検査する。"""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Iterator

import pytest
from playwright.sync_api import Browser, Page, expect

from scripts.render_overlay_design import new_page, snapshot, SCENES
from src.phase_j.live_publish import LiveOverlayServer, LiveStreamState
from tests.phase_j.test_overlay_browser import browser, page


@pytest.fixture
def server() -> Iterator[LiveOverlayServer]:
    instance = LiveOverlayServer(LiveStreamState(), '127.0.0.1', 0,
                                 overlay=dict(layout='l-left', scale=75, numbers=False))
    instance.start()
    yield instance
    instance.stop()


def test_corners_scale_fonts_and_band_never_cover_game(browser: Browser) -> None:
    for corner, scale, font, band in product(
        ['top-left', 'top-right', 'bottom-left', 'bottom-right'], [70, 80, 85],
        ['small', 'medium', 'large'], ['true', 'false']):
        query = f'l&corner={corner}&scale={scale}&font={font}&band={band}'
        page = new_page(browser, query)
        page.evaluate('renderSnapshot', snapshot(1.))
        game = page.evaluate('OverlayOptions.geometry(overlayOptions)')
        for selector in ['#panel', '#band']:
            element = page.locator(selector)
            if not element.is_visible():
                continue
            box = element.bounding_box()
            assert box['x'] >= 0 and box['y'] >= 0, query
            assert box['x']+box['width'] <= 1920 and box['y']+box['height'] <= 1080, query
            assert (box['x'] >= game['x']+game['width'] or box['x']+box['width'] <= game['x']
                    or box['y'] >= game['y']+game['height'] or box['y']+box['height'] <= game['y']), query
            assert element.evaluate('(e)=>e.scrollHeight<=e.clientHeight+1'), query
        for selector in ['.num', '.meta', '#details', '#whole']:
            assert page.locator(selector).first.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1'), query
        page.close()


@pytest.mark.parametrize('key,selector', [('numbers', '#p1'), ('recent', '.trend'),
    ('full', '#whole'), ('status', '.state'), ('details', '#details')])
def test_switches_work_on_every_layout(browser: Browser, key: str, selector: str) -> None:
    for layout in ['card', 'ribbon', 'minimal', 'data', 'l-right', 'l-left']:
        for enabled in ['true', 'false']:
            page = new_page(browser, f'{layout}&{key}={enabled}')
            # 廃止された切替指定より、配置ごとの 0.6.0 初期値を優先する。
            visible = key == 'numbers' or (key == 'recent' and layout != 'minimal')
            visible = visible or (key == 'full' and layout.startswith('l'))
            assert page.locator(selector).is_visible() == visible
            page.close()


def test_full_history_survives_short_window_and_calculation_gap(page: Page) -> None:
    for time in [0, 30000, 60000, 90000, 90000]:
        page.evaluate('renderSnapshot', snapshot(.6, time))
    assert page.evaluate('fullHistory.length') == 4
    assert page.evaluate('fullStart') == 0
    assert page.evaluate('history.length') == 2
    data = snapshot(.6, 91000)
    data['evaluations']['practical']['availability'] = 'unavailable'
    page.evaluate('renderSnapshot', data)
    assert page.evaluate('fullHistory.length') == 4
    data['identity']['match_id'] = 'next'
    page.evaluate('renderSnapshot', data)
    assert page.evaluate('fullHistory.length') == 0


def test_full_history_bound_keeps_start(page: Page) -> None:
    page.evaluate('()=>{for(let i=0;i<10000;i++) updateFullHistory(.5,i*1000)}')
    assert page.evaluate('fullHistory.length') <= 2048
    assert page.evaluate('fullHistory[0].time') == 0


def test_http_defaults_url_override_and_settings(browser: Browser, server: LiveOverlayServer) -> None:
    page = browser.new_page()
    errors: list[str] = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    origin = f'http://127.0.0.1:{server.address()[1]}'
    page.goto(origin + '/?preview=sample&scale=85&numbers=true')
    assert page.evaluate('overlayOptions.layout') == 'l-top-right'
    assert page.evaluate('overlayOptions.scale') == 85
    assert page.locator('#p1').is_visible()
    page.goto(origin + '/settings')
    page.locator('#image').set_input_files(SCENES['early'][1])
    page.locator('[name=layout]').select_option('l-bottom-right')
    page.locator('[name=scale]').fill('80')
    assert page.locator('[name=full]').count() == 0
    assert page.locator('[name=midline]').is_hidden()
    assert 'midline=adaptive' in page.locator('#url').input_value()
    assert '"midline": "adaptive"' in page.locator('#json').input_value()
    assert '位置 (384, 216)' in page.locator('#placement').inner_text()
    assert 'numbers=' not in page.locator('#url').input_value()
    frame = page.frame_locator('#overlay')
    frame.locator('#p1').wait_for(state='attached')
    # 見本は iframe の load 後に描画されるため、要素の存在だけでは待ち足りない。
    expect(frame.locator('#state')).to_have_text('対戦中')
    assert frame.locator('#stage').get_attribute('data-midline') == 'adaptive'
    assert page.locator('#frame').is_visible()
    page.locator('#scene').select_option('result')
    expect(frame.locator('#state')).to_have_text('決着 · 2P')
    assert 'preview=' not in page.locator('#url').input_value()
    assert not errors
    page.close()


@pytest.mark.parametrize('corner', ['top-left', 'top-right', 'bottom-left', 'bottom-right'])
def test_game_pixels_are_transparent(browser: Browser, corner: str) -> None:
    import io
    from PIL import Image
    page = new_page(browser, f'l&corner={corner}')
    game = page.evaluate('OverlayOptions.geometry(overlayOptions)')
    image = Image.open(io.BytesIO(page.screenshot(omit_background=True)))
    alpha = image.getchannel('A').crop((game['x'], game['y'],
        game['x']+game['width'], game['y']+game['height']))
    assert alpha.getextrema() == (0, 0)
    page.close()


@pytest.mark.parametrize('layout', ['card', 'ribbon', 'minimal', 'data'])
def test_overlay_all_items_large_fit_viewport(browser: Browser, layout: str) -> None:
    page = new_page(browser, f'{layout}&font=large&recent=true&details=true&full=true')
    page.evaluate('renderSnapshot', snapshot(1.))
    page.wait_for_timeout(50)
    box = page.locator('#panel').bounding_box()
    assert box['y']+box['height'] <= 1080
    for selector in ['.scores', '.num', '#details', '.meta']:
        assert page.locator(selector).first.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1')
    page.close()


def test_invalid_url_uses_valid_file_defaults(browser: Browser, server: LiveOverlayServer) -> None:
    page = browser.new_page()
    page.goto(f'http://127.0.0.1:{server.address()[1]}/?preview=sample&scale=100&layout=bad&numbers=bad')
    assert page.evaluate('overlayOptions.scale') == 75
    assert page.evaluate('overlayOptions.layout') == 'l-top-right'
    assert page.locator('.scores').is_visible()
    assert page.request.get(f'http://127.0.0.1:{server.address()[1]}/puyo_live.json').status == 404
    page.close()


def test_settings_image_shows_through_iframe(browser: Browser, server: LiveOverlayServer) -> None:
    import io
    from PIL import Image, ImageChops
    page = browser.new_page(viewport=dict(width=1500, height=1400), reduced_motion='reduce')
    page.goto(f'http://127.0.0.1:{server.address()[1]}/settings')
    page.locator('#image').set_input_files(SCENES['exchange'][1])
    page.wait_for_function("document.getElementById('frame').naturalWidth>0")
    page.frame_locator('#overlay').locator('#state').wait_for(state='attached')
    # HTMLの到着だけでなく、L字の配置が反映されてから透明領域を比較する。
    from playwright.sync_api import expect
    expect(page.frame_locator('#overlay').locator('#stage')).to_have_attribute('data-layout', 'l')
    page.locator('#frame').evaluate('''async image=>{
      await image.decode();
      await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
    }''')
    box = page.locator('#frame').bounding_box()
    clip = dict(x=box['x']+2, y=box['y']+2, width=box['width']-4, height=box['height']-4)
    with_overlay = Image.open(io.BytesIO(page.screenshot(clip=clip)))
    page.locator('#overlay').evaluate("e=>e.style.visibility='hidden'")
    without_overlay = Image.open(io.BytesIO(page.screenshot(clip=clip)))
    assert ImageChops.difference(with_overlay, without_overlay).getbbox() is None
    page.evaluate("navigator.clipboard.writeText=async text=>window.copiedUrl=text")
    page.locator('#copy').click()
    assert page.evaluate('window.copiedUrl') == page.locator('#url').input_value()
    page.close()
