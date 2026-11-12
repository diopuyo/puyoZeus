"""グラフの幅だけを試合内で広げる表示と設定互換を確認する。"""
import pytest
import re
from playwright.sync_api import Browser, Page, expect

from scripts.render_overlay_design import new_page, snapshot
from src.phase_j.overlay_config import validate_overlay
from src.phase_j.distribution_config import load_settings
from tests.phase_j.test_overlay_browser import browser, page
from tests.phase_j.test_distribution_app import distribution


def test_expansion_preserves_time_coordinates(browser: Browser) -> None:
    instance = new_page(browser, 'center')
    instance.clock.install(time=0)
    instance.clock.pause_at(1000)
    instance.evaluate('renderSnapshot', snapshot(.55, 1000))
    instance.evaluate('renderSnapshot', snapshot(.8, 2000))
    before = list(map(float, re.findall(r'-?\d+\.?\d*', instance.locator('#trace').get_attribute('d'))))
    instance.clock.run_for(150)
    assert .1 < instance.evaluate('PuyoGraphScale.half()') < .345
    during = list(map(float, re.findall(r'-?\d+\.?\d*', instance.locator('#trace').get_attribute('d'))))
    assert before[1::2] == during[1::2]
    assert before[0] > during[0]
    instance.clock.run_for(200)
    assert instance.evaluate('PuyoGraphScale.half()') == pytest.approx(.345)
    after = list(map(float, re.findall(r'-?\d+\.?\d*', instance.locator('#trace').get_attribute('d'))))
    assert before[1::2] == after[1::2]
    assert during[0] > after[0]
    assert instance.evaluate('shown') == .8
    instance.close()


def test_scale_lifecycle(page: Page) -> None:
    page.evaluate('clearFullHistory()')
    assert page.evaluate('PuyoGraphScale.half()') == .1
    page.evaluate('renderSnapshot', snapshot(.55))
    page.wait_for_timeout(350)
    assert page.evaluate('PuyoGraphScale.half()') == .1
    page.evaluate('renderSnapshot', snapshot(.8, 31000))
    page.wait_for_timeout(350)
    width = page.evaluate('PuyoGraphScale.half()')
    assert width == pytest.approx(.345)
    page.evaluate('renderSnapshot', snapshot(.51, 32000))
    assert page.evaluate('PuyoGraphScale.half()') == width
    data = snapshot(.5, 33000)
    data['identity']['match_id'] = 'next-match'
    page.evaluate('renderSnapshot', data)
    assert page.evaluate('PuyoGraphScale.half()') == .1
    page.evaluate('renderSnapshot', snapshot(1., 34000))
    page.wait_for_timeout(350)
    assert page.evaluate('PuyoGraphScale.half()') == .5


def test_settings_and_evaluation(page: Page) -> None:
    assert page.evaluate('OverlayOptions.resolve({}).graph_autoscale') is True
    for mode in [True, False]:
        assert validate_overlay({'graph_autoscale': mode}) == {'graph_autoscale': mode}
    page.evaluate("Object.assign(overlayOptions,{graph_autoscale:false,value_display:'probability'})")
    assert page.evaluate('PuyoEvaluation.graph(.6)') == .6
    page.evaluate("overlayOptions.value_display='evaluation'")
    assert page.evaluate('PuyoEvaluation.graph(.6)') == pytest.approx(.5+243/6000)
    page.evaluate("overlayOptions.graph_autoscale=true;clearFullHistory()")
    assert page.evaluate('PuyoEvaluation.graph(.6)') == 1
    assert page.evaluate('PuyoEvaluation.graph(.4)') == 0


def test_layouts(browser: Browser) -> None:
    for layout in ['center', 'minimal', 'vertical', 'card', 'l-top-left',
                   'l-top-right', 'l-bottom-left', 'l-bottom-right']:
        for mode in ['probability', 'evaluation']:
            instance = new_page(browser, f'{layout}&value_display={mode}')
            errors: list[str] = []
            instance.on('pageerror', lambda error: errors.append(str(error)))
            for index, probability in enumerate([.5, .55, .8, .51, 1.]):
                instance.evaluate('renderSnapshot', snapshot(probability, 30000+index*1000))
                instance.wait_for_timeout(350)
                assert instance.evaluate('Number.isFinite(PuyoEvaluation.graph(shown))')
            assert not errors, (layout, mode, errors)
            assert 'NaN' not in instance.locator('#trace').get_attribute('d')
            instance.close()


def test_checkbox_save_reload_and_language(browser: Browser, distribution: tuple) -> None:
    runtime, url = distribution
    settings = browser.new_page(viewport={'width': 390, 'height': 844})
    settings.goto(url+'/settings')
    settings.locator('#tab-appearance').click()
    choice = settings.get_by_label('グラフの縦の幅を試合に合わせる', exact=True)
    expect(choice).to_be_checked()
    expect(choice).to_have_attribute('type', 'checkbox')
    with settings.expect_response(lambda response: response.url.endswith('/api/settings') and response.request.method == 'POST'):
        choice.uncheck()
    settings.reload()
    settings.locator('#tab-appearance').click()
    expect(choice).not_to_be_checked()
    assert load_settings(runtime.root)['overlay']['graph_autoscale'] is False
    with settings.expect_response(lambda response: response.url.endswith('/api/settings') and response.request.method == 'POST'):
        choice.check()
    settings.reload()
    settings.locator('#tab-appearance').click()
    expect(choice).to_be_checked()
    assert load_settings(runtime.root)['overlay']['graph_autoscale'] is True
    settings.evaluate("PuyoI18n.setLanguage('en')")
    expect(settings.get_by_label('Fit graph range to the match', exact=True)).to_be_checked()
    assert settings.evaluate('document.documentElement.scrollWidth<=390')
    settings.close()
