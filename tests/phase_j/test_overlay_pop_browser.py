"""テーマ選択・4配置、保存座標、ドラッグ、ゲーム外の席を製品HTTPで確認する。"""
from __future__ import annotations

import io
import json
from itertools import product

from PIL import Image
from playwright.sync_api import Browser, expect
import pytest

from src.phase_j.overlay_config import validate_overlay
from scripts.render_overlay_design import snapshot
from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_overlay_options_browser import server


@pytest.mark.parametrize('key,value', [('x', -1), ('x', 1921), ('y', 1081),
    ('y', float('nan')), ('x', float('inf')), ('x', True), ('y', '20')])
def test_invalid_position(key: str, value: object) -> None:
    with pytest.raises(ValueError):
        validate_overlay({key: value})


@pytest.mark.parametrize('old,new', [('classic', 'pop'), ('noir', 'pop'),
    ('arena', 'pop'), ('neon', 'pop'), ('pop', 'pop')])
def test_legacy_theme(browser: Browser, server: object, old: str, new: str) -> None:
    page = browser.new_page()
    page.goto(f'http://127.0.0.1:{server.address()[1]}/?preview=sample&theme={old}&layout=data')
    assert page.evaluate('overlayOptions.theme') == new
    assert page.evaluate('overlayOptions.layout') == 'card'
    assert page.evaluate('overlayOptions.details') is True
    page.close()


def test_settings_drag_reset_and_language(browser: Browser, server: object) -> None:
    page = browser.new_page(viewport=dict(width=1500, height=1100))
    page.goto(f'http://127.0.0.1:{server.address()[1]}/settings')
    # 採用された黒とクリームの二つ。
    assert page.locator('[name=theme] option').count() == 2
    # 採用された中央・小さい帯・縦・L 字の四つ。
    assert page.locator('[name=layout] option').count() == 4
    page.locator('[name=layout]').select_option('center')
    expect(page.locator('#overlap-warning')).to_contain_text('主要なゲーム表示を避けた位置')
    handle = page.locator('#drag-handle')
    expect(handle).to_be_visible()
    handle.scroll_into_view_if_needed()
    b = handle.bounding_box()
    page.mouse.move(b['x']+b['width']/2, b['y']+b['height']/2)
    page.mouse.down()
    page.mouse.move(b['x']-150, b['y']-100, steps=8)
    page.mouse.up()
    expect(page.locator('#url')).to_have_value(__import__('re').compile(r'.*[?&]x=\d+.*'))
    saved = json.loads(page.locator('#json').input_value())['overlay']
    assert isinstance(saved['x'], int) and isinstance(saved['y'], int)
    page.locator('[data-position=top-left]').click()
    expect(page.locator('[name=x]')).to_have_value('0')
    page.locator('[data-position=center]').click()
    expect(page.locator('[name=x]')).to_have_value('')
    page.locator('[name=x]').fill('300')
    page.locator('[name=y]').fill('300')
    expect(page.locator('#overlap-warning')).to_contain_text('注意')
    page.locator('[name=language]').select_option('en')
    expect(page.locator('[name=theme] option[value=pop]')).to_have_text('Pop (black)')
    expect(page.locator('[name=theme] option[value=pop-cream]')).to_have_text('Pop (cream)')
    expect(page.locator('[name=mascot]').locator('..')).to_contain_text('Zeus-kun')
    page.close()


@pytest.mark.parametrize('layout', ['card', 'ribbon', 'minimal'])
def test_position_and_perch(browser: Browser, server: object, layout: str) -> None:
    page = browser.new_page(viewport=dict(width=1920, height=1080))
    page.goto(f'http://127.0.0.1:{server.address()[1]}/?preview=sample&layout={layout}&x=42&y=420')
    page.wait_for_function("document.querySelector('#mascot img').naturalWidth>0")
    box = page.locator('#panel').bounding_box()
    mascot = page.locator('#mascot').bounding_box()
    assert box['x'] == 42 and box['y'] == 420
    if layout == 'ribbon':
        assert mascot['y'] == box['y']+box['height']+8
    else:
        assert mascot['y']+mascot['height'] == box['y']+2
    assert box['x'] <= mascot['x'] <= box['x']+box['width']-mascot['width']
    page.close()


@pytest.mark.parametrize('theme,corner', list(product(['normal', 'esports', 'pop'],
    ['top-left', 'top-right', 'bottom-left', 'bottom-right'])))
def test_l_mascot_outside_game(browser: Browser, server: object, theme: str, corner: str) -> None:
    page = browser.new_page(viewport=dict(width=1920, height=1080), reduced_motion='reduce')
    page.goto(f'http://127.0.0.1:{server.address()[1]}/?preview=sample&layout=l&theme={theme}&corner={corner}&numbers=true&scale=85&debug_panel=false')  # 既定のデバッグ表示は連鎖の中身欄を隠すため切る
    page.evaluate('renderSnapshot', snapshot(.6))
    page.wait_for_function("document.querySelector('#mascot img').naturalWidth>0")
    g = page.evaluate('OverlayOptions.geometry(overlayOptions)')
    alpha = Image.open(io.BytesIO(page.screenshot(omit_background=True))).getchannel('A')
    assert alpha.crop((g['x'], g['y'], g['x']+g['width'], g['y']+g['height'])).getextrema() == (0, 0)
    mascot = page.locator('#mascot').bounding_box()
    for selector in ['#contents-extra', '#details', '#whole']:
        content = page.locator(selector).bounding_box()
        if content is None:  # 採用表示では説明の欄を表示しない。
            continue
        assert (content['y']+content['height'] <= mascot['y'] or mascot['y']+mascot['height'] <= content['y']
                or content['x']+content['width'] <= mascot['x'] or mascot['x']+mascot['width'] <= content['x'])
    assert page.locator('#mascot img').get_attribute('alt') == 'ゼウスくん'
    page.close()


@pytest.mark.parametrize('corner,scale', list(product(
    ['top-left', 'top-right', 'bottom-left', 'bottom-right'], [70, 85])))
def test_no_band_mascot_uses_empty_space(browser: Browser, server: object, corner: str, scale: int) -> None:
    page = browser.new_page(viewport=dict(width=1920, height=1080))
    page.goto(f'http://127.0.0.1:{server.address()[1]}/?preview=sample&layout=l&band=false&corner={corner}&scale={scale}&mascot_size=large')
    page.wait_for_function("document.querySelector('#mascot img').naturalWidth>0")
    g = page.evaluate('OverlayOptions.geometry(overlayOptions)')
    mascot = page.locator('#mascot').bounding_box()
    panel = page.locator('#panel').bounding_box()
    assert mascot['y']+mascot['height'] <= panel['y'] or panel['y']+panel['height'] <= mascot['y']
    assert mascot['y'] >= g['y']+g['height'] or mascot['y']+mascot['height'] <= g['y']
    assert 0 <= mascot['y'] and mascot['y']+mascot['height'] <= 1080
    page.close()
