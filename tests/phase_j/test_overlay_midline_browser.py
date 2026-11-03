"""50%表示の設定・縮小・テーマ色と、履歴への非干渉を実ブラウザで確認する。"""
from __future__ import annotations

import io

from PIL import Image, ImageChops
from playwright.sync_api import Browser
import pytest

from scripts.render_overlay_design import new_page, snapshot
from src.phase_j.overlay_config import MIDLINES, THEMES, validate_overlay
from tests.phase_j.test_overlay_browser import browser

VIEWPORTS = ((1920, 1080), (390, 844))
MIN_LABEL_PIXELS = 6


@pytest.mark.parametrize('option', MIDLINES)
@pytest.mark.parametrize('theme', THEMES)
def test_midlines_survive_mobile_and_backgrounds(browser: Browser, theme: str, option: str) -> None:
    page = new_page(browser, f'l-left&theme={theme}&midline={option}&full=true')
    page.evaluate('renderSnapshot', snapshot(.8))
    for width, height in VIEWPORTS:
        page.set_viewport_size(dict(width=width, height=height))
        page.evaluate('resizeStage()')
        for background in ('#ffffff', '#000000'):
            page.evaluate('color=>document.body.style.background=color', background)
            for selector in ('.trend .mid-label', '#whole .mid-label'):
                label = page.locator(selector)
                assert label.inner_text() == '50'
                assert label.is_hidden()  # 採用表示ではグラフの文字を出さない。
            # 背景が白でも黒でも、50%の破線が実際に描画される。
            bar = page.locator('#trend')
            before = Image.open(io.BytesIO(bar.screenshot()))
            page.add_style_tag(content='#trend .guide{visibility:hidden}')
            after = Image.open(io.BytesIO(bar.screenshot()))
            assert ImageChops.difference(before, after).getbbox() is not None
            page.locator('style').last.evaluate('(e)=>e.remove()')
    assert page.locator('.guide').first.evaluate('(e)=>getComputedStyle(e).strokeDasharray') == '7px, 5px'
    page.close()


@pytest.mark.parametrize('layout', ['card', 'ribbon', 'minimal', 'data', 'l-right', 'l-left', 'l'])
def test_styles_preserve_probability_and_paths(browser: Browser, layout: str) -> None:
    paths = []
    for option in MIDLINES:
        page = new_page(browser, f'{layout}&midline={option}&recent=true&full=true')
        for index, probability in enumerate((.2, .5, .8)):
            page.evaluate('renderSnapshot', snapshot(probability, index * 1000))
        paths.append([page.locator(selector).get_attribute('d') for selector in ('#trace', '#whole-trace')])
        assert page.locator('#p1').inner_text() == '80'
        assert page.locator('#stage').get_attribute('data-midline') == option
        for selector in ('#trace', '#whole-trace'):
            assert page.locator(selector).evaluate('(e)=>getComputedStyle(e).strokeWidth') == '4px'
        page.close()
    assert all(path == paths[0] for path in paths)


@pytest.mark.parametrize('option', MIDLINES)
def test_config_accepts_midline(option: str) -> None:
    assert validate_overlay({'midline': option}) == {'midline': option}


def test_config_rejects_unknown_midline() -> None:
    with pytest.raises(ValueError, match='midline'):
        validate_overlay({'midline': 'dashed'})


def test_url_precedence_and_default(browser: Browser) -> None:
    page = new_page(browser)
    assert page.evaluate('overlayOptions.midline') == 'adaptive'
    assert page.evaluate("OverlayOptions.resolve({midline:'glow'},'?midline=split').midline") == 'split'
    assert page.evaluate("OverlayOptions.resolve({midline:'glow'},'?midline=invalid').midline") == 'glow'
    page.close()
