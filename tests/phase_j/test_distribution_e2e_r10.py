"""言語切替で操作結果に語単位の翻訳を当てないことを確認する。"""
import pytest
from playwright.sync_api import Browser, expect

from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_overlay_browser import browser

TRANSIENT_IDS = ('obs-result', 'save-status', 'log-status', 'app-save-status', 'obs-copied')


@pytest.mark.parametrize('language', ['ja', 'en'])
@pytest.mark.parametrize('kind', ['added', 'already_added', 'url_updated'])
def test_obs_result_language_switch(distribution: tuple, browser: Browser,
                                    language: str, kind: str) -> None:
    _, url = distribution
    page = browser.new_page()
    page.route('**/api/obs', lambda route: route.fulfill(json=dict(
        scene='配信', already_added=kind == 'already_added',
        url_updated=kind == 'url_updated', url='http://127.0.0.1:8766/')))
    try:
        page.goto(url + '/settings?language=' + language)
        page.locator('#tab-other').click()
        page.locator('[name=language]').select_option(language)
        page.evaluate("setLocalMessage('obs-result','以前のエラー / Old error')")
        page.evaluate('async()=>{await obsAction(true)}')
        result = page.locator('#obs-result')
        expect(result).to_contain_text('OBS')
        before = result.inner_text()
        page.evaluate('PuyoI18n.translate()')
        expect(result).to_have_text(before)
        page.locator('[name=language]').select_option('en' if language == 'ja' else 'ja')
        page.locator('#tab-stream').click()
        expect(result).to_be_empty()
        page.evaluate('renderAppNotice();PuyoI18n.translate()')
        expect(result).to_be_empty()
    finally:
        page.close()


@pytest.mark.parametrize('language', ['ja', 'en'])
@pytest.mark.parametrize('ident', TRANSIENT_IDS)
def test_transient_text_and_cached_results(distribution: tuple, browser: Browser,
                                          language: str, ident: str) -> None:
    _, url = distribution
    page = browser.new_page()
    try:
        page.goto(url + '/settings?language=' + language)
        page.evaluate('(lang)=>PuyoI18n.setLanguage(lang)', language)
        page.evaluate("id=>setLocalMessage(id,'配信の操作結果')", ident)
        page.evaluate('PuyoI18n.translate()')
        expect(page.locator('#' + ident)).to_have_text('配信の操作結果')
        other = 'en' if language == 'ja' else 'ja'
        page.evaluate('(lang)=>PuyoI18n.setLanguage(lang)', other)
        page.evaluate('renderLocalMessages();PuyoI18n.translate()')
        expect(page.locator('#' + ident)).to_be_empty()
        page.evaluate("id=>setLocalMessage(id,'保存しました / Saved')", ident)
        page.evaluate('(lang)=>PuyoI18n.setLanguage(lang)', language)
        expect(page.locator('#' + ident)).to_have_text('Saved' if language == 'en' else '保存しました')
    finally:
        page.close()
