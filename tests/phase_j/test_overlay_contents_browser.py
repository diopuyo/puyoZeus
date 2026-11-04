"""実値の表示、項目選択、古い値の消去とHTTP資産を検証する。"""
from __future__ import annotations

from playwright.sync_api import Browser
import pytest

from scripts.render_overlay_design import new_page, snapshot
from src.b3_contents import contents_from_row
from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_overlay_options_browser import server
from tests.test_b3_contents import feature_row


@pytest.mark.parametrize('theme', ['classic', 'neon', 'arena', 'pop', 'noir'])
@pytest.mark.parametrize('layout', ['card', 'ribbon', 'minimal', 'data', 'l-right', 'l-left',
    'l&corner=bottom-left', 'l&corner=bottom-right', 'l&band=false'])
def test_real_contents_and_missing_values(browser: Browser, layout: str, theme: str) -> None:
    page = new_page(browser, f'{layout}&theme={theme}&details=true&content_hands=true&content_model=true')
    data = snapshot()
    data['evaluations']['b3_contents'] = contents_from_row(feature_row())
    page.evaluate('renderSnapshot', data)
    assert '12個' in page.locator('[data-field=pending]').inner_text()
    assert '5手' in page.locator('[data-field=hands]').inner_text()
    assert '120個' in page.locator('[data-field=return]').inner_text()
    assert '＋108個' in page.locator('[data-field=margin]').inner_text()
    assert page.locator('[data-field=length] span').last.inner_text() == '—'
    assert '撃ち合い' in page.locator('#contents-model').inner_text()
    visible = page.locator('body').inner_text()
    for old in ('段', '来る', '台帳', 'mask', 'episode', 'static', 'exchange', '評価',
                '計算中', '推定勝率', '予測不能', '上限補完', ' Q', ' M', ' H', 'k1'):
        assert old not in visible
    for selector in ('#panel', '#band', '.content-row', '.content-row span'):
        for node in page.locator(selector).all():
            if node.is_visible():
                assert node.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1'), (layout, selector)
    page.evaluate("unavailable('更新待ち')")
    assert '12個' not in page.locator('[data-field=pending]').inner_text()
    page.close()


def test_units_signs_and_chain_wording(browser: Browser) -> None:
    page = new_page(browser, 'data')
    cases = [(3, 'step', '3連鎖目'), (8, 'prediction', '予想 8連鎖'),
             (5, 'remaining', '残り 5連鎖'), (-12, 'margin', '−12個'),
             (0, 'margin', '0個'), (5, 'hands', '5手'), (None, 'ojama', '—')]
    for value, unit, expected in cases:
        assert page.evaluate('([v,u])=>contentValue(v,u)', [value, unit]) == expected
    assert '＋なら返せる・−なら足りない' in page.locator('#margin-note').inner_text()
    page.close()


def test_screen_step_is_separate_from_b3_count(browser: Browser) -> None:
    """古いDTO・欠測・終了を件数にフォールバックせず、予想は維持する。"""
    page = new_page(browser, 'data')
    contents = contents_from_row(feature_row())
    contents['p1'].update(chain_n_steps=1, chain_pred_l=8, chain_display_step=14)
    page.evaluate('renderContents', contents)
    assert page.locator('[data-field=step] span').nth(1).inner_text() == '14連鎖目'
    assert page.locator('[data-field=length] span').nth(1).inner_text() == '予想 8連鎖'
    contents['p1']['chain_display_step'] = None
    page.evaluate('renderContents', contents)
    assert page.locator('[data-field=step] span').nth(1).inner_text() == '—'
    del contents['p1']['chain_display_step']
    page.evaluate('renderContents', contents)
    assert page.locator('[data-field=step] span').nth(1).inner_text() == '—'
    page.close()


def test_each_content_switch(browser: Browser) -> None:
    page = new_page(browser, 'data')
    fields = page.evaluate('CONTENT_GROUPS.flatMap(g=>g[1].map(f=>f[0])).concat("model")')
    page.close()
    for field in fields:
        page = new_page(browser, f'l-right&content_{field}=true')
        selector = '#contents-model' if field == 'model' else f'[data-field={field}]'
        assert page.locator(selector).is_hidden()  # 旧URLを読めても説明は配信に出さない。
        page.close()
        page = new_page(browser, f'l-right&content_{field}=false')
        if field == 'model':
            assert not page.locator('#contents-model').is_visible()
        else:
            assert page.locator(f'[data-field={field}]').count() == 0
        page.close()


def test_column_only_keeps_player_headers_when_main_fields_are_off(browser: Browser) -> None:
    """下帯の主要項目を消しても、横列の1P/2P見出しを失わない。"""
    page = new_page(browser, 'l&band=false&content_pending=false&content_return=false&content_margin=false')
    assert page.locator('#contents-main .content-group').count() == 0
    heading = page.locator('#contents-extra .content-head').first
    assert heading.is_hidden()
    assert heading.locator('span').all_text_contents()[-2:] == ['1P', '2P']
    page.close()


def test_settings_copies_individual_choice_and_serves_contents(browser: Browser, server: object) -> None:
    page = browser.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    origin = f'http://127.0.0.1:{server.address()[1]}'
    assert page.request.get(origin+'/overlay_contents.js').status == 200
    page.goto(origin+'/settings')
    controls = page.locator('#controls fieldset').filter(has=page.locator('[name=content_pending]'))
    assert controls.is_hidden()  # 表示内容の欄は廃止。内部の既存設定は互換のため維持する。
    assert page.locator('[name=content_pending]').is_checked()
    assert not page.locator('[name=content_hands]').is_checked()
    assert page.locator('[name=content_hands]').is_hidden()
    page.evaluate("()=>{form.elements.content_hands.checked=true;updatePreview()}")
    assert 'content_hands=true' in page.locator('#url').input_value()
    page.evaluate("()=>{form.elements.content_margin.checked=false;updatePreview()}")
    assert 'content_margin=false' in page.locator('#url').input_value()
    assert '"content_margin": false' in page.locator('#json').input_value()
    assert not errors
    page.close()


@pytest.mark.parametrize('theme', ['classic', 'neon', 'arena', 'pop', 'noir'])
@pytest.mark.parametrize('layout', ['l-right', 'l-left', 'l&scale=85', 'l&band=false', 'data'])
def test_default_contents_are_readable_and_fit(browser: Browser, layout: str, theme: str) -> None:
    """旧設定のデータを保持しつつ、採用表示には説明を出さない。"""
    page = new_page(browser, f'{layout}&theme={theme}')
    page.set_viewport_size(dict(width=1080, height=608))
    page.evaluate('resizeStage()')
    data = snapshot()
    data['evaluations']['b3_contents'] = contents_from_row(feature_row())
    page.evaluate('renderSnapshot', data)
    assert set(page.locator('[data-field]').evaluate_all('(es)=>es.map(e=>e.dataset.field)')) == {
        'pending', 'return', 'margin', 'step', 'length', 'k1'}
    assert not page.locator('#contents-model').is_visible()
    for selector in ('#panel', '#band', '#details', '#contents-extra'):
        for node in page.locator(selector).all():
            if node.is_visible():
                assert node.evaluate('(e)=>e.scrollHeight<=e.clientHeight+1'), (layout, theme, selector)
    for node in page.locator('.content-row span, .contents-title, #margin-note').all():
        if node.is_visible():
            assert node.evaluate('(e)=>parseFloat(getComputedStyle(e).fontSize)>=22')
            assert node.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1')
    for row in page.locator('[data-field]').all():
        assert row.is_hidden()
    assert '3手以内' in page.locator('[data-field=k1]').get_attribute('title')
    page.close()
