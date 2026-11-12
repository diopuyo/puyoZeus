"""日英の更新操作・既定ON・明示OFF・確認前の無通信を実ブラウザで検査する。"""
import json
from pathlib import Path
import threading

import pytest
from playwright.sync_api import Browser, expect

from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_distribution_update import local_http
from src.phase_j import update_download as fetch


@pytest.mark.parametrize('language', ['ja', 'en'])
def test_check_progress_cancel(browser: Browser, distribution: tuple, local_http: tuple,
                               monkeypatch: pytest.MonkeyPatch, language: str) -> None:
    runtime, url = distribution
    origin, bodies = local_http
    monkeypatch.setenv('PUYO_UPDATE_BUTTON', '1')
    runtime.version = dict(app='0.6.4')
    runtime.settings.update(language=language, releases_url=origin+'/releases.json')
    bodies['/releases.json'] = json.dumps(dict(version='0.6.5', date='2026-11-10',
                                              summary='<script>literal text</script>')).encode()
    def download(info: dict, destination: Path, cancel: threading.Event, progress: object,
                 local_origin: str | None = None) -> None:
        progress(256, 1024)
        assert cancel.wait(15)
        raise fetch.Cancelled()
    monkeypatch.setattr(fetch, 'download', download)
    page = browser.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(url+'/settings')
    page.locator('#tab-other').click()
    page.locator('#check-update').evaluate("node => {let p=node.closest('details');while(p){p.open=true;p=p.parentElement.closest('details')}}")
    expect(page.locator('#check-update')).to_be_visible()
    assert runtime.update_checked is False
    page.locator('#check-update').click()
    expect(page.locator('#update-release')).to_contain_text('0.6.5')
    expect(page.locator('#update-release')).to_contain_text('2026-11-10')
    expect(page.locator('#update-summary')).to_have_text('<script>literal text</script>')
    expect(page.locator('#update-summary script')).to_have_count(0)
    page.locator('#install-update').click()
    expect(page.locator('#update-progress')).to_contain_text('256 / 1024')
    expect(page.locator('#update-download-progress')).to_be_visible()
    page.locator('#cancel-update').click()
    expect(page.locator('#update-progress')).to_contain_text('cancelled' if language == 'en' else '中断')
    assert not errors
    runtime.updater.thread.join(timeout=10)
    page.close()


def test_update_visible_and_api_enabled_by_default(browser: Browser, distribution: tuple,
                                                   local_http: tuple,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, url = distribution
    origin, bodies = local_http
    runtime.settings['releases_url'] = origin+'/releases.json'
    bodies['/releases.json'] = json.dumps(dict(version='0.6.6')).encode()
    monkeypatch.delenv('PUYO_UPDATE_BUTTON', raising=False)
    page = browser.new_page()
    page.goto(url+'/settings')
    page.locator('#tab-other').click()
    page.locator('#check-update').evaluate("node => {let p=node.closest('details');while(p){p.open=true;p=p.parentElement.closest('details')}}")
    expect(page.locator('#check-update')).to_be_visible()
    assert runtime.update_checked is False
    response = page.request.post(url+'/api/update/check', data={})
    assert response.status == 200
    page.close()


def test_update_hidden_and_api_disabled_when_explicit_off(browser: Browser, distribution: tuple,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    _, url = distribution
    monkeypatch.setenv('PUYO_UPDATE_BUTTON', '0')
    page = browser.new_page()
    page.goto(url+'/settings')
    page.locator('#tab-other').click()
    expect(page.locator('#app-update')).to_be_hidden()
    response = page.request.post(url+'/api/update/check', data={})
    assert response.status == 400
    page.close()
