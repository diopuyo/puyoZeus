"""出来事だけ吹き出しを出し、ゼウスくんを各枠の中央に保つ。"""
from playwright.sync_api import Browser, expect
import pytest

from scripts.render_overlay_design import new_page
from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_overlay_battle_browser import ROOT, selected


@pytest.mark.parametrize('layout', ['ribbon', 'l'])
def test_event_speech_and_center(browser: Browser, layout: str) -> None:
    page = new_page(browser, layout+'&theme=battle-a&mascot=true&scale=80')
    page.add_script_tag(path=str(ROOT/'overlay_winrate_switch.js'))
    page.evaluate('document.fonts.ready')
    data = selected(.8)
    for event, words in [('idle', ''), ('exchange', ''),
                         ('waiting', ''), ('result', ''), ('between', '次の試合まだかな')]:
        data['evaluations']['winrate_switch']['v49f_active'] = event == 'exchange'
        data['display']['status'] = 'hold' if event == 'waiting' else 'live'
        if event == 'result':
            data['evaluations']['winrate_switch']['line1'] = 0.
            data['evaluations']['practical']['source'] = 'confirmed_death'
        # PROMPT 5・7: 保留や決着には出さず、試合待ちだけ表示する。
        if event == 'between':
            data['display']['input_status'] = 'no_puyo_screen'
        page.evaluate('renderSnapshot', data)
        speech = page.locator('#battle-speech')
        expect(speech).to_have_text(words)
        mascot = page.locator('#mascot').bounding_box()
        center = 960 if layout == 'ribbon' else 110
        assert mascot['x']+mascot['width']/2 == center
        if layout == 'l':
            assert mascot['y']+mascot['height']/2 == 972
        if not words:
            expect(speech).to_be_hidden()
            continue
        box = speech.bounding_box()
        if layout == 'ribbon':
            assert box['x']+box['width']/2 == center
            assert box['x'] >= 830 and box['x']+box['width'] <= 1090
            assert box['y'] >= mascot['y']+mascot['height']
            assert box['y']+box['height'] <= 935
        else:
            assert box['x'] >= mascot['x']+mascot['width']
            assert box['x']+box['width'] <= 1920
            assert box['y'] >= 864 and box['y']+box['height'] <= 1080
    page.close()
