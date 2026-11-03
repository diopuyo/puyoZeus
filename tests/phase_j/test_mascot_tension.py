"""画像だけの弾みを実ブラウザで確認する。"""
from __future__ import annotations

import pytest
import re
from playwright.sync_api import Page

from scripts.render_overlay_design import snapshot
from tests.phase_j.test_mascot_expressions import face_page
from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_distribution_app import distribution


def motions(page: Page, selector: str = '#mascot img') -> list[dict]:
    """実際に適用されたアニメーションの時間と変形を取得する。"""
    return page.locator(selector).evaluate(
        '(e)=>e.getAnimations().map(a=>({duration:a.effect.getTiming().duration,'
        'frames:a.effect.getKeyframes().map(f=>f.transform)}))')


@pytest.mark.parametrize('level', ['low', 'normal', 'high'])
@pytest.mark.parametrize('probability,winner,excited', [
    (.1, None, False), (.5, None, False), (.7, None, True), (.9, None, True),
    (.5, '1P', True), (.9, '2P', False)])
def test_tension_conditions(face_page: Page, level: str, probability: float,
                            winner: str | None, excited: bool) -> None:
    face_page.evaluate('(level)=>{overlayOptions.mascot_dopamine=level;'
                       'overlayOptions.mascot_expressions=false;}', level)
    data = snapshot(probability)
    if winner:
        data['terminal'] = dict(state='confirmed', winner=winner)
    face_page.evaluate('renderSnapshot', data)
    assert bool(motions(face_page)) == (excited and level != 'low')
    assert face_page.locator('#mascot').get_attribute('data-expression') == 'off'


@pytest.mark.parametrize('setting', ['motion', 'reduced', 'hidden'])
def test_disable_cancels_active_motion(face_page: Page, setting: str) -> None:
    face_page.evaluate('renderSnapshot', snapshot(.9))
    assert motions(face_page)
    if setting == 'reduced':
        face_page.emulate_media(reduced_motion='reduce')
        face_page.wait_for_function('document.querySelector("#mascot img").getAnimations().length===0')
    else:
        name = 'mascot_motion' if setting == 'motion' else 'mascot'
        face_page.evaluate('(name)=>{overlayOptions[name]=false;PuyoMascot.update({});}', name)
    assert not motions(face_page)
    assert not motions(face_page, '#mascot')


@pytest.mark.parametrize('state', ['between_matches', 'offline', 'unavailable', 'new_match', 'rewind'])
def test_boundary_cancels_previous_motion(face_page: Page, state: str) -> None:
    data = snapshot(.9)
    face_page.evaluate('renderSnapshot', data)
    assert motions(face_page)
    if state == 'offline':
        face_page.evaluate("unavailable('接続待ち','offline')")
    else:
        if state == 'between_matches':
            data['display']['match_phase'] = state
        elif state == 'unavailable':
            data['evaluations']['practical']['availability'] = 'unavailable'
        else:
            data['evaluations']['practical']['p1_win_probability'] = .5
            if state == 'new_match':
                data['identity']['match_id'] = 'next-match'
            else:
                data['timing']['source_available_ms'] -= 1
        face_page.evaluate('renderSnapshot', data)
    assert not motions(face_page)
    assert not motions(face_page, '#mascot')


def test_amplitude_hop_and_layout(face_page: Page) -> None:
    face_page.evaluate('renderSnapshot', snapshot(.5))
    box = face_page.locator('#mascot').bounding_box()
    face_page.evaluate('renderSnapshot', snapshot(.9))
    assert not motions(face_page, '#mascot')
    face_page.clock.run_for(600)
    # 仮想時計とは独立のWeb Animations時計を確定させ、配置だけを比較する。
    face_page.locator('#mascot').evaluate('(e)=>e.getAnimations().forEach(a=>a.finish())')
    normal = motions(face_page)[0]
    assert normal['duration'] == 720
    assert normal['frames'][1] == 'translateY(-8%) rotate(-6deg) scale(1.08)'
    assert face_page.locator('#mascot').bounding_box() == box
    face_page.evaluate("overlayOptions.mascot_dopamine='high';PuyoMascot.update({})")
    high = motions(face_page)[0]
    assert high['duration'] == 480 < normal['duration']
    assert high['frames'][1] == 'translateY(-12%) rotate(-9deg) scale(1.12)'
    assert face_page.locator('#mascot').bounding_box() == box


def test_display_band_and_custom_edges(face_page: Page) -> None:
    face_page.evaluate('renderSnapshot', snapshot(.5))
    face_page.evaluate('renderSnapshot', snapshot(.9))
    assert motions(face_page)
    face_page.evaluate('cancelAnimationFrame(animation);paintProbability(.69999)')
    assert not motions(face_page)
    face_page.evaluate('paintProbability(.7)')
    assert motions(face_page)
    face_page.evaluate('overlayOptions.mascot_band_edges=[20,40,80,95];PuyoMascot.update({})')
    assert not motions(face_page)
    face_page.evaluate('paintProbability(.8)')
    assert motions(face_page)


@pytest.mark.parametrize('level,p,lift,tilt,scale,duration', [
    ('normal', .75, 4, 3, 1.04, 900), ('normal', .95, 8, 6, 1.08, 720),
    ('high', .75, 7, 5, 1.07, 620), ('high', .95, 12, 9, 1.12, 480)])
def test_band_amplitudes(face_page: Page, level: str, p: float, lift: int,
                         tilt: int, scale: float, duration: int) -> None:
    """70帯を維持し、90帯と1P決着に強い動きを適用する。"""
    face_page.evaluate('(level)=>overlayOptions.mascot_dopamine=level', level)
    face_page.evaluate('renderSnapshot', snapshot(p))
    motion = motions(face_page)[0]
    assert motion['duration'] == duration
    assert motion['frames'][1] == f'translateY(-{lift}%) rotate(-{tilt}deg) scale({scale})'
    if p == .95:
        data = snapshot(.5)
        data['terminal'] = dict(state='confirmed', winner='1P')
        face_page.evaluate('renderSnapshot', data)
        assert motions(face_page) == [motion]


def test_band_and_level_order_and_return(face_page: Page) -> None:
    """全軸で90>70・high>normalを確認し、70帯へ戻ると弱くなる。"""
    result = {}
    for level in ('normal', 'high'):
        face_page.evaluate('(level)=>overlayOptions.mascot_dopamine=level', level)
        for band, p in [('70', .75), ('90', .95), ('70', .75)]:
            face_page.evaluate('renderSnapshot', snapshot(p))
            face_page.evaluate('(p)=>{cancelAnimationFrame(animation);paintProbability(p);}', p)
            motion = motions(face_page)[0]
            values = [abs(float(v)) for v in re.findall(r'-?\d+(?:\.\d+)?', motion['frames'][1])]
            current = (*values, motion['duration'])
            if (level, band) in result:
                assert result[level, band] == current
            result[level, band] = current
    for smaller, larger in [(('normal', '70'), ('normal', '90')), (('high', '70'), ('high', '90')),
                            (('normal', '70'), ('high', '70')), (('normal', '90'), ('high', '90'))]:
        assert all(a < b for a, b in zip(result[smaller][:3], result[larger][:3]))
        assert result[smaller][3] > result[larger][3]
