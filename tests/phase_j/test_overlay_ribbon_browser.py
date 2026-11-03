"""旧 ribbon URL の中央の柱への互換と寸法・線・設定画面での位置追従を確かめる。"""
from __future__ import annotations

import json
from playwright.sync_api import Browser, expect
import pytest

from scripts.render_overlay_design import new_page
from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_overlay_options_browser import server
from tests.phase_j.test_overlay_battle_browser import ROOT, selected


@pytest.mark.parametrize('p', [0., .5, 1.])
def test_ribbon_bounds_numbers_and_recent(browser: Browser, p: float) -> None:
    page=new_page(browser,'ribbon&theme=battle-a&mascot=true')
    page.add_script_tag(path=str(ROOT / 'overlay_winrate_switch.js'))
    page.evaluate('document.fonts.ready')
    page.evaluate('renderSnapshot',selected(p))
    expect(page.locator('#trend')).to_be_visible()
    expect(page.locator('#battle-vs')).to_have_count(0)
    expect(page.locator('#p1')).to_have_text(str(int(p*100)))
    assert page.locator('#panel').bounding_box()==dict(x=838,y=180,width=244,height=560)
    assert page.locator('#mascot').bounding_box()==dict(x=900,y=748,width=120,height=120)
    expect(page.locator('#battle-speech')).to_be_hidden()
    assert page.locator('.num').evaluate_all('(nodes)=>nodes.every(e=>e.scrollWidth<=e.clientWidth+1&&e.scrollHeight<=e.clientHeight+1)')
    assert page.locator('#trace').get_attribute('d').endswith('4.0')
    page.close()


def test_ribbon_explicit_options_and_position(browser: Browser) -> None:
    page=new_page(browser,'ribbon&theme=battle-a&x=900&y=650&recent=false&mascot=false')
    page.evaluate('renderSnapshot',selected(.5))
    assert page.locator('#panel').bounding_box()==dict(x=900,y=650,width=244,height=560)
    expect(page.locator('#trend')).to_be_hidden()
    expect(page.locator('#mascot')).to_be_hidden()
    expect(page.locator('#battle-speech')).to_be_hidden()
    page.close()


def test_ribbon_finite_motion_and_result(browser: Browser) -> None:
    page=new_page(browser,'ribbon&theme=battle-a')
    page.add_script_tag(path=str(ROOT / 'overlay_winrate_switch.js'))
    page.evaluate('document.fonts.ready')
    page.emulate_media(reduced_motion='no-preference')
    page.evaluate('renderSnapshot',selected(.5,0))
    data=selected(.9,1000)
    data['evaluations']['winrate_switch']['v49f_active']=True
    page.evaluate('renderSnapshot',data)
    assert page.locator('.right').evaluate('(e)=>e.getAnimations().length')==0
    assert page.locator('.player:not(.right)').evaluate('(e)=>e.getAnimations().length')==0
    expect(page.locator('.battle-state')).to_have_count(0)
    page.wait_for_timeout(1900)
    assert page.locator('#mascot img').evaluate('(e)=>e.getAnimations().length')==1
    page.emulate_media(reduced_motion='reduce')
    data=selected(0.,2000)
    data['evaluations']['practical']['source']='confirmed_death'
    page.evaluate('renderSnapshot',data)
    expect(page.locator('#result-text')).to_have_text('WIN! ▶')
    expect(page.locator('.player:not(.right) .num')).to_have_css('opacity','0.6')
    assert page.evaluate('document.getAnimations().length')==0
    page.close()


def test_ribbon_settings_drag_and_overlap(browser: Browser, server: object) -> None:
    page=browser.new_page(viewport=dict(width=1500,height=1100))
    page.goto(f'http://127.0.0.1:{server.address()[1]}/settings?layout=ribbon&theme=battle-a&mascot=true')
    page.locator('[name=layout]').select_option('center')
    page.locator('[name=theme]').select_option('pop')
    page.locator('[name=mascot]').check()
    frame=page.frame_locator('#overlay')
    expect(frame.locator('#panel')).to_have_attribute('data-layout','center')
    expect(frame.locator('#stage')).to_have_attribute('data-theme','pop')
    expect(page.locator('#overlap-warning')).to_contain_text('主要なゲーム表示を避けた位置')
    handle=page.locator('#drag-handle')
    handle.scroll_into_view_if_needed()
    b=handle.bounding_box()
    page.mouse.move(b['x']+b['width']/2,b['y']+b['height']/2)
    page.mouse.down()
    page.mouse.move(b['x']+b['width']/2+70,b['y']+b['height']/2-20,steps=8)
    page.mouse.up()
    expect(page.locator('[name=x]')).not_to_have_value('')
    saved=json.loads(page.locator('#json').input_value())['overlay']
    expect(frame.locator('#panel')).to_have_css('left',f"{saved['x']}px")
    expect(frame.locator('#mascot')).to_have_css('left',f"{saved['x']+62}px")
    expect(frame.locator('#mascot')).to_have_css('top',f"{saved['y']+568}px")
    expect(frame.locator('#battle-speech')).to_have_css('left','50%')
    assert saved['layout']=='center' and saved['theme']=='pop' and saved['mascot']
    page.close()
