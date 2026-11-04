"""設定画面の追加操作と、日英の短い受信結果を実ブラウザで確認する。"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from playwright.sync_api import Browser, expect
import pytest

from src import production_config
from src.phase_j.distribution_config import validate_settings
from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_overlay_browser import browser


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('language', ['ja', 'en'])
def test_feedback_settings(browser: Browser, distribution: tuple, monkeypatch: pytest.MonkeyPatch,
                           language: str, enabled: bool) -> None:
    monkeypatch.setattr(production_config, 'DISTRIBUTION_SITE_FEEDBACK', enabled)
    runtime, url = distribution
    runtime.settings = validate_settings(runtime.settings)
    page = browser.new_page(viewport={'width': 1000, 'height': 900})
    errors: list[str] = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(url + '/settings?language=' + language)
    page.locator('#tab-other').click()
    page.locator('[name=language]').select_option(language)
    english = language == 'en'
    expect(page.locator('#open-reports')).to_have_text('Open report folder' if english else '報告のフォルダを開く')
    expect(page.locator('#open-reports')).to_be_visible()
    if enabled:
        expect(page.locator('#report-local-notice')).to_have_text('')
    expect(page.locator('#open-feedback')).to_be_visible() if enabled else expect(page.locator('#open-feedback')).to_be_hidden()
    expect(page.locator('#uninstall-help, #uninstall-app')).to_have_count(0)
    with patch('src.phase_j.distribution_http.os.startfile') as folder:
        page.locator('#open-reports').click()
        expect(page.locator('#save-status')).to_have_text('Opened' if english else '開きました')
        folder.assert_called_once_with(str(runtime.store.directory.resolve()))
    if enabled:
        with patch('src.phase_j.distribution_http.webbrowser.open', return_value=True) as opened:
            with page.expect_response(url + '/api/feedback/open'):
                page.locator('#open-feedback').click()
            expect(page.locator('#save-status')).to_have_text('Opened' if english else '開きました')
            assert opened.call_count == 1
    screenshot_directory = os.environ.get('FEEDBACK_SCREENSHOT_DIR')
    if screenshot_directory:
        directory = Path(screenshot_directory)
        directory.mkdir(parents=True, exist_ok=True)
        page.locator('#open-reports').locator('..').screenshot(path=str(directory / f'buttons-{language}-{enabled}.png'))
    assert not errors
    page.close()


@pytest.mark.parametrize('language', ['ja', 'en'])
@pytest.mark.parametrize('response,expected_ja,expected_en', [
    ({'sent': True, 'accepted': 1, 'duplicates': 2}, '受理 1 件、重複 2 件', 'Accepted: 1; duplicates: 2'),
    ({'sent': False, 'status': 429}, '送信回数の上限', 'Rate limit reached'),
    ({'sent': False, 'status': 413}, '報告が大きすぎます', 'Reports are too large'),
])
def test_receipt_language(browser: Browser, distribution: tuple, language: str,
                          response: dict, expected_ja: str, expected_en: str) -> None:
    _, url = distribution
    page = browser.new_page()
    page.route('**/api/logs/preview', lambda route: route.fulfill(json={
        'token': 'local-test', 'local_only': False, 'payload': {'reports': []}}))
    page.route('**/api/logs/send', lambda route: route.fulfill(json=response))
    page.goto(url + '/settings?language=' + language)
    page.locator('#tab-other').click()
    page.locator('[name=language]').select_option(language)
    page.locator('#send-logs').click()
    expect(page.locator('#confirm-logs')).to_be_enabled()
    page.locator('#confirm-logs').click()
    expect(page.locator('#log-status')).to_contain_text(expected_en if language == 'en' else expected_ja)
    page.locator('#cancel-logs').click()
    other = 'ja' if language == 'en' else 'en'
    page.locator('[name=language]').select_option(other)
    expect(page.locator('#log-status')).to_contain_text(expected_ja if other == 'ja' else expected_en)
    page.close()
