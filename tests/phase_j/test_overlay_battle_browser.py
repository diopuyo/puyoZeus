"""A対戦ゲージは選択済みの一本を保ち、状態と演出だけを変える。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterator

from PIL import Image
from playwright.sync_api import Browser, Page
import pytest

from scripts.render_overlay_design import new_page, snapshot
from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_overlay_options_browser import server

ROOT = Path(__file__).resolve().parents[2] / 'src' / 'phase_j'


@pytest.fixture
def battle(browser: Browser) -> Iterator[Page]:
    page = new_page(browser, 'l&theme=battle-a')
    page.add_script_tag(path=str(ROOT / 'overlay_winrate_switch.js'))
    yield page
    page.close()


def selected(p: float, time_ms: int = 30000) -> dict:
    data = snapshot(.23, time_ms)
    data['evaluations']['winrate_switch'] = dict(line1=p, v49f_active=False)
    return data


@pytest.mark.parametrize('p', [0., .1234, .5, .8766, 1.])
def test_selected_line_drives_numbers_gauge_and_graphs(battle: Page, p: float) -> None:
    data = selected(p)
    before = deepcopy(data)
    battle.evaluate('renderSnapshot', data)
    percent = int(p*100+.5)
    assert battle.locator('#p1').inner_text() == str(percent)
    assert battle.locator('#p2').inner_text() == str(100-percent)
    assert battle.locator('#fill').evaluate('(e)=>e.style.transform') == f'scaleX({p:g})'
    assert battle.locator('#fill').evaluate('(e)=>e.style.width') == ''
    assert battle.evaluate('fullHistory.at(-1).p') == p
    assert battle.locator('#trace').get_attribute('d').endswith(f'{4+(1-p)*46:.1f}')
    assert data == before


def test_delta_uses_thirty_seconds_not_last_180_events(battle: Page) -> None:
    rows = [selected(.2+i/1000, i*100) for i in range(401)]
    battle.evaluate('(rows)=>rows.forEach(renderSnapshot)', rows)
    assert battle.locator('#battle-change').count() == 0
    assert battle.locator('#trace').get_attribute('d').startswith('M4.0 36.2')
    # 同じ時刻の更新は二重登録せず、最新の値を使う。
    battle.evaluate('renderSnapshot', selected(.1, 40000))
    assert battle.locator('#trace').get_attribute('d').endswith('408.0 45.4')
    assert battle.locator('#trace').get_attribute('d').count('L') == 300


@pytest.mark.parametrize('reset_kind', ['match', 'backward', 'unavailable'])
def test_delta_resets_at_discontinuities(battle: Page, reset_kind: str) -> None:
    battle.evaluate('renderSnapshot', selected(.2, 0))
    battle.evaluate('renderSnapshot', selected(.8, 30000))
    data = selected(.6, 31000)
    if reset_kind == 'match':
        data['identity']['match_id'] = 'next'
    elif reset_kind == 'backward':
        data['timing']['source_available_ms'] = 0
    else:
        battle.evaluate("unavailable('更新待ち')")
        assert battle.locator('#trace').get_attribute('d') == ''
    battle.evaluate('renderSnapshot', data)
    assert battle.locator('#trace').get_attribute('d') == 'M408.0 22.4'


def test_exchange_pending_and_confirmed_result_use_existing_fields(battle: Page) -> None:
    data = selected(.8)
    data['evaluations']['winrate_switch']['v49f_active'] = True
    battle.evaluate('renderSnapshot', data)
    assert not battle.locator('#state').is_visible()
    assert battle.locator('#p1').inner_text() == '80'
    data['display']['status'] = 'hold'
    battle.evaluate('renderSnapshot', data)
    assert battle.locator('#state').inner_text() == '判定中'
    data['display']['status'] = 'live'
    data['evaluations']['winrate_switch']['line1'] = 0.
    battle.evaluate('renderSnapshot', data)
    assert 'WIN' not in battle.locator('#state').inner_text()
    data['evaluations']['practical']['source'] = 'confirmed_death'
    battle.evaluate('renderSnapshot', data)
    assert battle.locator('#result-text').inner_text() == 'WIN! ▶'
    assert battle.locator('.player:not(.right) .num').evaluate('(e)=>getComputedStyle(e).opacity') == '0.6'


def test_extra_winrate_cards_are_hidden(battle: Page) -> None:
    for name in ['overlay_v49f.js', 'overlay_ama.js']:
        battle.add_script_tag(path=str(ROOT / name))
    battle.evaluate('renderSnapshot', selected(.7))
    assert not battle.locator('#v49f-card').is_visible()
    assert not battle.locator('#ama-card').is_visible()


def test_no_perpetual_motion_and_reduced_motion(battle: Page) -> None:
    battle.emulate_media(reduced_motion='no-preference')
    battle.evaluate('renderSnapshot', selected(.5, 0))
    data = selected(.9, 1000)
    data['evaluations']['winrate_switch']['v49f_active'] = True
    battle.evaluate('renderSnapshot', data)
    assert battle.locator('.right').evaluate('(e)=>e.getAnimations().length') == 0
    assert battle.locator('.player:not(.right)').evaluate('(e)=>e.getAnimations().length') == 0
    battle.wait_for_timeout(1900)
    assert battle.locator('#mascot img').evaluate('(e)=>e.getAnimations().length') == 1
    assert Image.open(ROOT / 'brand/overlay_mascot_still.webp').n_frames == 1
    battle.emulate_media(reduced_motion='reduce')
    battle.evaluate('renderSnapshot', selected(.3, 2000))
    assert battle.evaluate('document.getAnimations().length') == 0


def test_settings_and_local_assets(browser: Browser, server: object) -> None:
    page = browser.new_page()
    base = f'http://127.0.0.1:{server.address()[1]}'
    page.goto(base+'/settings')
    page.locator('[name=theme]').select_option('pop')
    assert 'theme=pop' in page.locator('#url').input_value()
    for name in ['DelaGothicOne-Regular', 'NotoSansJP']:
        response = page.request.get(base+f'/fonts/{name}.woff2')
        assert response.status == 200
        assert response.headers['content-type'] == 'font/woff2'
        assert response.body()[:4] == b'wOF2'
    page.goto(base+'/?preview=sample&theme=battle-a&layout=l&debug_panel=true')
    assert page.evaluate('PuyoDebug.enabled')  # 明示ONの試験用表示はデザインによらず使える。
    assert page.locator('#mascot img').get_attribute('src') == '/mascot/50.webp'
    page.close()


def test_font_fallback_keeps_numbers_inside_players(browser: Browser, server: object) -> None:
    page = browser.new_page(viewport=dict(width=1920, height=1080))
    page.route('**/*.woff2', lambda route: route.abort())
    page.goto(f'http://127.0.0.1:{server.address()[1]}/?preview=sample&theme=battle-a&layout=l&scale=85')
    page.evaluate('renderSnapshot', snapshot(1.))
    for node in page.locator('.player').all():
        assert node.evaluate('(e)=>e.scrollWidth<=e.clientWidth')
    page.close()


def test_numbers_off_keeps_gauge_and_no_band_keeps_mascot_outside_game(browser: Browser, server: object) -> None:
    page = browser.new_page(viewport=dict(width=1920, height=1080))
    page.goto(f'http://127.0.0.1:{server.address()[1]}/?preview=sample&theme=battle-a&layout=l&band=false&numbers=false')
    page.evaluate('renderSnapshot', snapshot(.8))
    assert page.locator('#bar').is_visible()
    assert page.locator('.player').first.is_visible()
    game = page.evaluate('OverlayOptions.geometry(overlayOptions)')
    box = page.locator('#mascot').bounding_box()
    assert box['y'] >= game['y']+game['height']
    assert box['y']+box['height'] <= 1080
    page.close()
