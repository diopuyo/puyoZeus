"""表情の境界・保持時間・表示値連動を実ブラウザで検証する。"""
from __future__ import annotations

from typing import Iterator

import pytest
from playwright.sync_api import Browser, Page, expect

from scripts.render_overlay_design import snapshot
from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_overlay_browser import browser


@pytest.fixture
def face_page(browser: Browser, distribution: tuple) -> Iterator[Page]:
    """仮想時計で保持の境界を正確に検査する。"""
    _, url = distribution
    page = browser.new_page(viewport={'width': 1920, 'height': 1080})
    page.clock.install()
    page.clock.pause_at(page.evaluate('Date.now()'))
    page.goto(url + '/?preview=sample&mascot=true&mascot_expressions=true')
    yield page
    page.close()


def expression(page: Page) -> str:
    return page.locator('#mascot').get_attribute('data-expression')


@pytest.mark.parametrize('p,expected', [
    (0, '10'), (.09999, '10'), (.1, '30'), (.10001, '30'),
    (.29999, '30'), (.3, '50'), (.30001, '50'), (.69999, '50'),
    (.7, '70'), (.70001, '70'), (.89999, '70'), (.9, '90'), (.90001, '90'), (1, '90')])
def test_band_edges(face_page: Page, p: float, expected: str) -> None:
    face_page.evaluate('renderSnapshot', snapshot(p))
    face_page.clock.run_for(1500)
    assert expression(face_page) == expected
    assert face_page.locator('#mascot img').get_attribute('src') == f'/mascot/{expected}.webp'
    expect(face_page.locator('#mascot img')).to_have_js_property('naturalWidth', 320)


def test_hold_uses_latest_band_without_more_messages(face_page: Page) -> None:
    face_page.evaluate('renderSnapshot', snapshot(.05))
    face_page.clock.run_for(1500)
    assert expression(face_page) == '10'
    face_page.evaluate('renderSnapshot', snapshot(.9))
    face_page.clock.run_for(500)
    face_page.evaluate('renderSnapshot', snapshot(.7))
    face_page.clock.run_for(999)
    assert expression(face_page) == '10'
    face_page.clock.run_for(1)
    assert expression(face_page) == '70'


def test_reads_animated_display_not_raw_probability(face_page: Page) -> None:
    face_page.evaluate('renderSnapshot', snapshot(.05))
    face_page.clock.run_for(3000)
    face_page.evaluate('renderSnapshot', snapshot(.9))
    assert expression(face_page) == '90'
    # 描画コールバックに途中値を渡し、端末の描画フレーム間隔に依存させない。
    face_page.evaluate('cancelAnimationFrame(animation);paintProbability(.2)')
    assert expression(face_page) == '90'
    face_page.clock.run_for(1500)
    assert expression(face_page) == '30'
    face_page.evaluate('paintProbability(.9)')
    face_page.clock.run_for(1499)
    assert expression(face_page) == '30'
    face_page.clock.run_for(1)
    assert expression(face_page) == '90'


@pytest.mark.parametrize('p,winner,expected', [(1, '1P', 'win'), (0, '2P', 'lose')])
@pytest.mark.parametrize('source', ['confirmed_death', 'terminal'])
def test_confirmed_result(face_page: Page, p: float, winner: str, expected: str, source: str) -> None:
    data = snapshot(p)
    if source == 'terminal':
        data['terminal'] = dict(state='confirmed', winner=winner)
    else:
        data['evaluations']['practical']['source'] = source
    face_page.evaluate('renderSnapshot', data)
    assert expression(face_page) == expected
    band = '90' if p else '10'
    assert face_page.locator('#mascot img').get_attribute('src') == f'/mascot/{band}.webp'


@pytest.mark.parametrize('state', ['calibrating', 'no_puyo_screen', 'between_matches', 'offline'])
def test_waiting_cancels_pending_expression(face_page: Page, state: str) -> None:
    face_page.evaluate('renderSnapshot', snapshot(.9))
    face_page.clock.run_for(1500)
    face_page.evaluate('renderSnapshot', snapshot(.05))
    data = snapshot(.05)
    if state == 'offline':
        face_page.evaluate("unavailable('接続待ち','offline')")
    else:
        key = 'match_phase' if state == 'between_matches' else 'input_status'
        data['display'][key] = state
        face_page.evaluate('renderSnapshot', data)
    assert expression(face_page) == '50'
    face_page.clock.run_for(3000)
    assert expression(face_page) == '50'


@pytest.mark.parametrize('position,enabled,scale', [
    ('right', False, '1'), ('right', True, '-1 1'), ('left', True, '1')])
def test_mirror_is_optional(face_page: Page, position: str, enabled: bool, scale: str) -> None:
    face_page.evaluate('([position,enabled])=>{document.getElementById("panel").style.left=position==="right"?"1300px":"100px";'
        'overlayOptions.mascot_flip_right=enabled;PuyoMascot.update({});}', [position, enabled])
    assert face_page.locator('#mascot img').evaluate('(e)=>e.style.scale') == scale


def test_disable_restores_original_and_cancels_timer(face_page: Page) -> None:
    face_page.evaluate('renderSnapshot', snapshot(.9))
    face_page.evaluate('overlayOptions.mascot_expressions=false;PuyoExpressions.update()')
    face_page.clock.run_for(3000)
    assert expression(face_page) == 'off'
    assert face_page.locator('#mascot img').get_attribute('src') == '/mascot/overlay.webp'


def test_settings_save_reload_and_url_override(browser: Browser, distribution: tuple) -> None:
    runtime, url = distribution
    page = browser.new_page()
    page.goto(url + '/settings')
    for name in ('mascot_expressions', 'mascot_flip_right'):
        assert page.locator(f'[name={name}]').is_checked() == (name == 'mascot_expressions')
        page.locator(f'[name={name}]').check()
    page.locator('[name=mascot_band_edges]').fill('20,40,60,80')
    page.locator('#save-settings').click()
    page.wait_for_function("document.querySelector('#save-status').textContent.includes('保存')")
    page.reload()
    for name in ('mascot_expressions', 'mascot_flip_right'):
        assert page.locator(f'[name={name}]').is_checked()
        assert runtime.settings['overlay'][name] is True
    assert runtime.settings['overlay']['mascot_band_edges'] == [20, 40, 60, 80]
    assert page.locator('[name=mascot_band_edges]').input_value() == '20,40,60,80'
    page.goto(url + '/?preview=sample&mascot=true&mascot_expressions=true')
    assert page.evaluate('PuyoExpressions.band(.3)') == '30'
    page.goto(url + '/?preview=sample&mascot_band_edges=10,30,70,90')
    assert page.evaluate('PuyoExpressions.band(.3)') == '50'
    page.goto(url + '/?preview=sample&mascot_band_edges=90,70,30,10')
    assert page.evaluate('PuyoExpressions.band(.3)') == '30'
    page.goto(url + '/?preview=sample&mascot=true&mascot_expressions=false&mascot_flip_right=false')
    assert expression(page) == 'off'
    assert page.locator('#mascot img').evaluate('(e)=>e.style.scale') == '1'
    page.close()
