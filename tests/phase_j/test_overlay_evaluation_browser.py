"""1P 視点の評価値・旧設定・全配置の読みやすさを製品HTMLで検証する。"""
from itertools import product
import json
from pathlib import Path

from playwright.sync_api import Browser, Page, expect

from scripts.render_overlay_design import new_page, snapshot
from src.phase_j.overlay_config import validate_overlay
from src.phase_j.distribution_config import load_settings
from tests.phase_j.test_overlay_browser import browser, page
from tests.phase_j.test_distribution_app import distribution

MIN_EVALUATION_FONT_PX = 54


def test_conversion_and_default(page: Page) -> None:
    result = page.evaluate("""() => {
      const f=PuyoEvaluation.value;
      return {mid:f(.5),edges:[f(0),f(1),f(Number.MIN_VALUE),f(1-Number.EPSILON)],
        pairs:[.001,.01,.1,.3,.49,.5,.8,.999].map(p=>[f(p),f(1-p)]),
        default:OverlayOptions.resolve({}).value_display,
        invalid:OverlayOptions.resolve({value_display:'bad'}).value_display};
    }""")
    assert result['mid'] == 0
    assert result['edges'] == [-9999, 9999, -9999, 9999]
    assert all(a == -b for a, b in result['pairs'])
    assert result['default'] == result['invalid'] == 'probability'
    for mode in ['probability', 'evaluation']:
        assert validate_overlay({'value_display': mode}) == {'value_display': mode}
    assert page.evaluate('OverlayOptions.DEFAULT_VALUE_DISPLAY') == 'probability'


def test_legacy_settings(page: Page, tmp_path: Path) -> None:
    for old, new in [('evaluation_main', 'evaluation'), ('probability_main', 'probability'),
                     ('evaluation', 'evaluation'), ('probability', 'probability')]:
        original = {'value_display': old}
        assert validate_overlay(original) == {'value_display': new}
        assert original == {'value_display': old}
        assert page.evaluate('mode=>OverlayOptions.resolve({value_display:mode}).value_display', old) == new
        assert page.evaluate("mode=>OverlayOptions.resolve({},'?value_display='+mode).value_display", old) == new
        (tmp_path / 'distribution.json').write_text(json.dumps({'overlay': original}), 'utf-8')
        assert load_settings(tmp_path)['overlay']['value_display'] == new


def assert_numbers_fit(page: Page, mode: str) -> None:
    selectors = ['#evaluation-value'] if mode == 'evaluation' else ['#p1', '#p2']
    for selector in selectors:
        assert page.locator(selector).evaluate("""e=>{
          const main=e.closest('.num')||e,box=e.closest('.plate')||e.parentElement;
          const r=main.getBoundingClientRect(),p=box.getBoundingClientRect();
          return r.left>=p.left&&r.right<=p.right&&r.top>=p.top&&r.bottom<=p.bottom;
        }"""), (mode, selector)
    if mode == 'evaluation':
        expect(page.locator('.scores')).to_be_hidden()
        expect(page.locator('#evaluation-value')).to_be_visible()
        assert page.locator('#evaluation-value').evaluate('e=>parseFloat(getComputedStyle(e).fontSize)') >= MIN_EVALUATION_FONT_PX
    else:
        expect(page.locator('#evaluation')).to_be_hidden()


def test_all_layouts_and_modes(browser: Browser) -> None:
    layouts = ['center', 'minimal', 'vertical', 'card', 'l-top-left',
               'l-top-right', 'l-bottom-left', 'l-bottom-right']
    for layout, theme, mode in product(layouts, ['pop', 'pop-cream'], ['probability', 'evaluation']):
        instance = new_page(browser, f'{layout}&theme={theme}&value_display={mode}')
        errors: list[str] = []
        instance.on('pageerror', lambda error: errors.append(str(error)))
        for probability in [0., .01, .5, .99, 1.]:
            instance.evaluate('renderSnapshot', snapshot(probability))
            instance.evaluate('document.fonts.ready')
            assert_numbers_fit(instance, mode)
            if mode == 'evaluation':
                expected = instance.evaluate('p=>PuyoEvaluation.value(p)', probability)
                assert int(instance.locator('#evaluation-value').inner_text()) == expected
        assert not errors, (layout, theme, mode, errors)
        instance.close()


def test_live_refresh_preserves_probability(page: Page) -> None:
    page.evaluate('renderSnapshot', snapshot(.75))
    page.evaluate("overlayOptions.value_display='evaluation';applyOverlayOptions(overlayOptions);PuyoEvaluation.refresh()")
    expect(page.locator('#evaluation-value')).to_have_text('+659')
    expect(page.locator('.scores')).to_be_hidden()
    assert page.evaluate('shown') == .75
    page.evaluate("overlayOptions.value_display='probability';applyOverlayOptions(overlayOptions);PuyoEvaluation.refresh()")
    expect(page.locator('#p1')).to_have_text('75')
    expect(page.locator('#p2')).to_have_text('25')
    expect(page.locator('#evaluation')).to_be_hidden()


def test_graph_limits_and_result(page: Page) -> None:
    page.evaluate("overlayOptions.value_display='evaluation';applyOverlayOptions(overlayOptions)")
    assert page.evaluate('[0,.5,1].map(PuyoEvaluation.graph)') == [0, .5, 1]
    assert page.evaluate('PuyoEvaluation.graph(.75)') == .5 + 659 / 6000
    data = snapshot(1.)
    data['terminal'] = {'state': 'confirmed', 'winner': '1P'}
    page.evaluate('renderSnapshot', data)
    assert 'WIN!' in page.locator('#result-text').inner_text()
    expect(page.locator('#evaluation-value')).to_have_text('+9999')


def test_advantage_colors_and_waiting(page: Page) -> None:
    page.evaluate("overlayOptions.value_display='evaluation';applyOverlayOptions(overlayOptions)")
    assert page.evaluate('PuyoEvaluation.NEUTRAL_LIMIT') == 100
    colors = {}
    for score, advantage in [(-101, 'two'), (-100, 'neutral'), (0, 'neutral'), (100, 'neutral'), (101, 'one')]:
        probability = page.evaluate('score=>1/(1+Math.exp(-score/PuyoEvaluation.SCALE))', score)
        page.evaluate('renderSnapshot', snapshot(probability))
        expect(page.locator('#evaluation')).to_have_attribute('data-advantage', advantage)
        assert int(page.locator('#evaluation-value').inner_text()) == score
        colors[advantage] = page.locator('#evaluation').evaluate('e=>getComputedStyle(e).backgroundColor')
    assert len(set(colors.values())) == 3
    page.evaluate("unavailable('waiting')")
    expect(page.locator('#evaluation-value')).to_have_text('—')
    expect(page.locator('#evaluation')).to_have_attribute('data-advantage', 'neutral')


def test_settings_save_and_open_overlay(browser: Browser, distribution: tuple) -> None:
    runtime, url = distribution
    settings = browser.new_page(viewport={'width': 390, 'height': 844})
    overlay = browser.new_page()
    # 未接続カメラの待機通知だけ止め、設定通知は実際のSSEで受ける。
    overlay.add_init_script("""const NativeEvents=EventSource;
      window.EventSource=class {constructor(url){
        return url==='/events'?new EventTarget():new NativeEvents(url);
      }};""")
    settings.goto(url + '/settings')
    settings.locator('#tab-appearance').click()
    choice = settings.locator('[name=value_display]')
    expect(choice).to_have_value('probability')
    expect(choice.locator('option')).to_have_text(['勝率のみ', '評価値のみ'])
    overlay.goto(url + '/')
    overlay.evaluate('renderSnapshot', snapshot(.75))
    overlay.evaluate('window.evaluationReloadProbe=123')
    choice.select_option('evaluation')
    expect(overlay.locator('#stage')).to_have_attribute('data-value-display', 'evaluation')
    expect(overlay.locator('#evaluation-value')).to_have_text('+659')
    expect(overlay.locator('.scores')).to_be_hidden()
    assert load_settings(runtime.root)['overlay']['value_display'] == 'evaluation'
    assert overlay.evaluate('window.evaluationReloadProbe') == 123
    assert settings.evaluate('document.documentElement.scrollWidth<=390')
    settings.reload()
    settings.locator('#tab-appearance').click()
    expect(choice).to_have_value('evaluation')
    choice.select_option('probability')
    expect(overlay.locator('#p1')).to_have_text('75')
    expect(overlay.locator('#evaluation')).to_be_hidden()
    settings.evaluate("PuyoI18n.setLanguage('en')")
    expect(choice.locator('option')).to_have_text(['Win probability only', 'Evaluation only'])
    settings.close()
    overlay.close()
