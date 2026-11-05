"""見本の初回表示と配置切替を、画像比較ではなく外接矩形で確認する。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, expect

from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_overlay_browser import browser

LAYOUTS = ('center', 'minimal', 'card', 'vertical', 'l-top-left', 'l-top-right',
           'l-bottom-left', 'l-bottom-right')
SIZES = ((1920, 1050), (1280, 720), (1100, 800), (900, 700))
ZOOMS = (1, 1.25, 1.5)
PREVIEW_TIMEOUT_MS = 5000
BOUNDS = """() => {
  const ROUNDING_EPSILON=.5;
  const rect=n=>{const r=n.getBoundingClientRect();return [r.left,r.top,r.right,r.bottom]};
  const inside=(a,b)=>a[0]>=b[0]-ROUNDING_EPSILON&&a[1]>=b[1]-ROUNDING_EPSILON&&
    a[2]<=b[2]+ROUNDING_EPSILON&&a[3]<=b[3]+ROUNDING_EPSILON;
  const frame=document.querySelector('#overlay'),doc=frame.contentDocument;
  const mascot=doc?.querySelector('#mascot');
  if(!mascot||!frame.clientWidth||doc.URL!==frame.src||doc.readyState!=='complete'||
    doc.fonts.status!=='loaded')return {ready:false};
  const portrait=mascot.querySelector('img');
  if(!portrait.complete||!portrait.naturalWidth)return {ready:false};
  const bounds=[0,0,frame.clientWidth,frame.clientHeight],m=rect(mascot),img=rect(portrait);
  const numbers=[...doc.querySelectorAll('.plate .num')].map(n=>{
    const range=doc.createRange();range.selectNodeContents(n);
    const r=rect(range),p=rect(n.closest('.plate'));
    const small=n.querySelector('small');
    const digits=rect(n.querySelector('[id]'));
    return {r,p,digits,unit:small&&rect(small),ok:inside(r,p)&&
      (!small||(inside(rect(small),p)&&digits[2]<=rect(small)[0]))};
  });
  return {ready:true,mascot:m,img,bounds,numbers,
    ok:inside(m,bounds)&&inside(img,bounds)&&numbers.length===2&&numbers.every(n=>n.ok)};
}"""


def record_preview_failure(page: Page, events: list[dict]) -> None:
    """通信失敗もジョブの記録に残し、矩形の失敗と区別する。"""
    if path := os.environ.get('PREVIEW_GEOMETRY_LOG'):
        record = dict(case=os.environ.get('PYTEST_CURRENT_TEST'), events=events,
                      frames=[frame.url for frame in page.frames],
                      iframe=page.locator('#overlay').evaluate('(e)=>({html:e.outerHTML,rect:e.getBoundingClientRect().toJSON()})'))
        with Path(path).with_suffix('.errors.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')


def assert_preview(page: Page) -> None:
    """読み込みと配置が確定した時点の座標を検査する。"""
    expect(page.frame_locator('#overlay').locator('#p1')).to_contain_text('56')
    result = page.wait_for_function('() => {const r=(' + BOUNDS + ')();return r.ok&&r}',
                                    timeout=PREVIEW_TIMEOUT_MS).json_value()
    assert result['ok'], result
    if path := os.environ.get('PREVIEW_GEOMETRY_LOG'):
        result['case'] = os.environ.get('PYTEST_CURRENT_TEST')
        with Path(path).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result) + '\n')


@pytest.mark.parametrize('language', ['ja', 'en'])
def test_settings_labels(distribution: tuple, browser: Browser, language: str) -> None:
    runtime, url = distribution
    runtime.configure({'overlay': {'language': language}})
    page = browser.new_page()
    try:
        page.goto(url + '/settings?language=' + language)
        expect(page.locator('h1+p')).to_have_count(0)
        expect(page.locator('#desktop-shortcut, #start-menu-shortcut')).to_have_count(0)
        label = '詳細設定' if language == 'ja' else 'Advanced settings'
        expect(page.locator('summary').filter(has_text=label)).to_have_count(1)
    finally:
        page.close()


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('zoom', ZOOMS)
@pytest.mark.parametrize('theme', ['pop', 'pop-cream'])
@pytest.mark.parametrize('layout', LAYOUTS)
def test_preview_bounds(distribution: tuple, browser: Browser, size: tuple[int, int],
                        zoom: float, theme: str, layout: str) -> None:
    runtime, url = distribution
    runtime.configure({'overlay': {'layout': layout, 'theme': theme}})
    page = browser.new_page(viewport={'width': round(size[0]/zoom), 'height': round(size[1]/zoom)},
                            device_scale_factor=zoom)
    events: list[dict] = []
    page.on('requestfailed', lambda request: events.append(dict(url=request.url, error=request.failure)))
    try:
        page.goto(url + '/settings')
        page.locator('#tab-appearance').click()
        assert_preview(page)
        page.locator('[name=layout]').select_option(LAYOUTS[(LAYOUTS.index(layout)+1) % len(LAYOUTS)])
        assert_preview(page)
        page.set_viewport_size({'width': 900, 'height': 700})
        assert_preview(page)
    except Exception:
        record_preview_failure(page, events)
        raise
    finally:
        page.close()


def test_preview_navigation_is_not_repeated(distribution: tuple, browser: Browser) -> None:
    _, url = distribution
    page = browser.new_page()
    try:
        page.goto(url + '/settings')
        page.locator('#tab-appearance').click()
        assert_preview(page)
        page.evaluate("""() => {
          window.previewChanges=[];
          new MutationObserver(records=>previewChanges.push(...records.map(r=>r.attributeName)))
            .observe(document.querySelector('#overlay'),{attributes:true,attributeFilter:['src']});
        }""")
        page.locator('[name=layout]').select_option('minimal')
        assert_preview(page)
        assert page.evaluate('previewChanges.length') == 1
        page.locator('#scene').select_option('result')
        expect(page.frame_locator('#overlay').locator('#p1')).to_have_text('0')
        assert page.evaluate('previewChanges.length') == 1
    finally:
        page.close()
