"""実ブラウザの埋め込み拒否と資格情報の消去を確かめる。"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from playwright.sync_api import Browser, expect
import pytest

from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_overlay_options_browser import server
from tests.phase_j.test_distribution_app import distribution


def test_other_origin_iframe_is_blocked(browser: Browser, distribution: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, url = distribution
    monkeypatch.setattr(runtime, 'scan_inputs', lambda: {})
    class Parent(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = f'<iframe src="{url}/settings"></iframe>'.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, format: str, *args: Any) -> None:
            pass
    parent = ThreadingHTTPServer(('127.0.0.1', 0), Parent)
    threading.Thread(target=parent.serve_forever, daemon=True).start()
    page = browser.new_page()
    messages = []
    page.on('console', lambda message: messages.append(message.text))
    try:
        page.goto(f'http://127.0.0.1:{parent.server_port}')
        page.wait_for_timeout(300)
        assert any('frame-ancestors' in message or 'X-Frame-Options' in message for message in messages)
        assert all(frame.locator('#save-settings').count() == 0 for frame in page.frames)
        page.goto(url + '/settings')
        page.evaluate("const frame=document.createElement('iframe');frame.id='same-origin-setup';frame.src='/setup';document.body.append(frame)")
        expect(page.frame_locator('#same-origin-setup').locator('body')).to_contain_text('OBS')
    finally:
        page.close()
        parent.shutdown()
        parent.server_close()


def test_password_clear_button(browser: Browser, distribution: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, url = distribution
    monkeypatch.setattr('src.phase_j.distribution_runtime.shortcuts', lambda *args: None)
    runtime.configure({'obs_password': 'private-secret'})
    page = browser.new_page()
    try:
        page.goto(url + '/settings')
        with page.expect_response(lambda response: response.url.endswith('/api/settings')):
            page.locator('#obs-password-clear').evaluate('(button) => button.click()')
        expect(page.locator('#obs-result')).to_contain_text('消去')
        assert runtime.settings['obs_password'] == ''
    finally:
        page.close()


@pytest.mark.parametrize('route', ['/', '/settings', '/latest', '/events', '/overlay_i18n.js'])
def test_inner_server_checks_host_and_fetch_site(server: Any, route: str) -> None:
    url = f'http://127.0.0.1:{server.address()[1]}'
    for headers in ({'Host': 'untrusted.invalid'}, {'Sec-Fetch-Site': 'cross-site'}):
        with pytest.raises(HTTPError) as denied:
            urlopen(Request(url + route, headers=headers), timeout=3)
        assert denied.value.code == 403
        assert denied.value.headers['X-Frame-Options'] == 'SAMEORIGIN'
