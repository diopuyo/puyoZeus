"""第四回の退行を偽の通知・WebSocketと隔離ブラウザで確かめる。"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from playwright.sync_api import Browser

from scripts.render_overlay_design import new_page, snapshot, ROOT
from tests.phase_j.test_overlay_browser import browser
from tests.phase_j.test_distribution_app import distribution
from src.phase_j import distribution_obs as obs

FAKE_EVENTS = """window.sources=[];
window.EventSource=class extends EventTarget {
 static CLOSED=2; constructor(url){super();this.url=url;this.readyState=0;sources.push(this);}
 close(){this.readyState=2;}
};"""
LAYOUTS = ['center', 'minimal', 'vertical', 'l-top-left', 'l-top-right',
           'l-bottom-left', 'l-bottom-right']


@pytest.mark.parametrize('url', ['/events', '/api/settings/events'])
def test_closed_stream_retries(browser: Browser, url: str) -> None:
    page = browser.new_page()
    page.add_init_script(FAKE_EVENTS)
    page.route('**/overlay.html', lambda route: route.fulfill(
        body=(ROOT / 'src/phase_j/overlay.html').read_text('utf-8').replace(
            '/*OVERLAY_DEFAULTS*/{}', '{"distribution_mode":true,"app_started_at":123}'),
        content_type='text/html'))
    page.goto((ROOT / 'src/phase_j/overlay.html').as_uri())
    page.clock.install()
    for state, expected in [(0, 1), (2, 2), (2, 3)]:
        page.evaluate('''([url,state])=>{const s=sources.filter(s=>s.url===url).at(-1);
          s.readyState=state;s.dispatchEvent(new Event('error'));s.dispatchEvent(new Event('error'));}''', [url, state])
        page.clock.fast_forward(3100)
        assert page.evaluate('(url)=>sources.filter(s=>s.url===url).length', url) == expected
    kind = 'analysis' if url == '/events' else 'appearance'
    data = snapshot(.68) if kind == 'analysis' else dict(app_started_at=123, theme='pop-cream')
    page.evaluate('''([url,kind,data])=>sources.filter(s=>s.url===url).at(-1).dispatchEvent(
        new MessageEvent(kind,{data:JSON.stringify(data)}))''', [url, kind, data])
    page.clock.fast_forward(1000)
    if kind == 'analysis':
        assert page.locator('#p1').inner_text() == '68'
    else:
        assert page.locator('#stage').get_attribute('data-pop') == 'cream'
    page.close()


def test_new_application_reloads_page(browser: Browser) -> None:
    page = browser.new_page()
    page.add_init_script(FAKE_EVENTS)
    loads: list[bool] = []
    def serve(route: Any) -> None:
        loads.append(True)
        route.fulfill(body=(ROOT / 'src/phase_j/overlay.html').read_text('utf-8').replace(
            '/*OVERLAY_DEFAULTS*/{}', '{"distribution_mode":true,"app_started_at":123}'), content_type='text/html')
    page.route('**/overlay.html', serve)
    page.goto((ROOT / 'src/phase_j/overlay.html').as_uri())
    dispatch = """stamp=>sources.find(s=>s.url==='/api/settings/events').dispatchEvent(
        new MessageEvent('appearance',{data:JSON.stringify({app_started_at:stamp,language:'en'})}))"""
    page.evaluate(dispatch, 123)
    assert len(loads) == 1
    with page.expect_navigation():
        page.evaluate(dispatch, 124)
    assert len(loads) == 2
    page.close()


def test_settings_notification_has_application_stamp(distribution: tuple) -> None:
    from urllib.request import urlopen
    runtime, url = distribution
    with urlopen(url + '/api/settings/events', timeout=5) as response:
        assert response.readline().decode().strip() == 'event: appearance'
        data = json.loads(response.readline().decode().removeprefix('data: '))
    assert data['app_started_at'] == runtime.started_at


@pytest.mark.parametrize('ready', [False, True])
def test_stopped_frame_during_verification(ready: bool) -> None:
    from src.phase_j.distribution_input import stopped_reference
    from src.phase_j.live_device import DirectShowSource
    source = object.__new__(DirectShowSource)
    source.config = SimpleNamespace(verify_fail_streak=2)
    source._split_diagnostics = False
    statuses: list[str] = []
    source.on_status = statuses.append
    source.sleep = lambda *args: None
    continuity = SimpleNamespace(diagnostics={})
    frame = stopped_reference()
    assert frame is not None
    source._screen_hold(ready, frame, continuity, 2, SimpleNamespace())
    source._screen_hold(ready, np.zeros_like(frame), continuity, 2, SimpleNamespace())
    assert statuses == (['obs_stopped', 'no_puyo_screen'] if ready else ['verifying', 'verifying'])


class Socket:
    """要求と応答のJSONを往復させる、ネットワークを使わないWebSocket。"""
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.closed = False

    def send(self, value: str) -> None:
        self.calls.append(json.loads(value)['d'])

    def recv(self) -> str:
        call = self.calls[-1]
        inputs = [dict(inputName=name, inputKind=kind) for name, kind in [
            ('puyoZeus Live 1234abcd', 'browser_source'), ('other', 'browser_source'),
            ('puyoZeus Live deadbeef', 'image_source')]]
        responses = dict(GetInputList=dict(inputs=inputs),
            GetInputSettings=dict(inputSettings=dict(url='http://127.0.0.1:8765/overlay')),
            GetCurrentProgramScene=dict(currentProgramSceneName='Game'),
            GetSceneItemList=dict(sceneItems=[dict(sourceName='puyoZeus Live 1234abcd', inputKind='browser_source')]),
            GetVirtualCamStatus=dict(outputActive=True))
        return json.dumps(dict(op=7, d=dict(requestId=call['requestId'],
            requestStatus=dict(result=True), responseData=responses.get(call['requestType'], {}))))

    def close(self) -> None:
        self.closed = True


def test_startup_refresh_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
    client = object.__new__(obs.ObsClient)
    client.socket = Socket()
    monkeypatch.setattr(obs, 'ObsClient', lambda *args: client)
    obs.refresh_saved_sources(dict(obs_port=4455, obs_password='saved'))
    presses = [c for c in client.socket.calls if c['requestType'] == 'PressInputPropertiesButton']
    assert [c['requestData'] for c in presses] == [dict(
        inputName='puyoZeus Live 1234abcd', propertyName='refreshnocache')]
    assert client.socket.closed


@pytest.mark.parametrize('apply', [False, True])
def test_manual_setup_refreshes_existing_source(monkeypatch: pytest.MonkeyPatch, apply: bool) -> None:
    client = object.__new__(obs.ObsClient)
    client.socket = Socket()
    monkeypatch.setattr(obs, 'ObsClient', lambda *args: client)
    obs.integrate(dict(obs_enabled=True, obs_port=4455, obs_password='saved'),
                  'http://127.0.0.1:8765/overlay', apply=apply)
    presses = [c['requestData'] for c in client.socket.calls if c['requestType'] == 'PressInputPropertiesButton']
    assert presses == ([dict(inputName='puyoZeus Live 1234abcd', propertyName='refreshnocache')] if apply else [])
    assert client.socket.closed


@pytest.mark.parametrize('enabled,password,count', [(False, 'saved', 0), (True, '', 0), (True, 'saved', 1)])
def test_startup_refresh_is_background(monkeypatch: pytest.MonkeyPatch,
                                      enabled: bool, password: str, count: int) -> None:
    entered, release = threading.Event(), threading.Event()
    threads: list[threading.Thread] = []
    original = threading.Thread
    def blocked(*args: Any) -> None:
        entered.set()
        release.wait(5)
        raise OSError('connection failed')
    def worker(**kwargs: Any) -> threading.Thread:
        thread = original(**kwargs)
        threads.append(thread)
        return thread
    monkeypatch.setattr(obs, 'ObsClient', blocked)
    monkeypatch.setattr(obs.threading, 'Thread', worker)
    try:
        obs.refresh_on_startup(dict(obs_enabled=enabled, obs_password=password, obs_port=4455))
        assert len(threads) == count
        if count:
            assert entered.wait(1) and threads[0].is_alive() and threads[0].daemon
    finally:
        release.set()
        for thread in threads:
            thread.join(2)
            assert not thread.is_alive()


@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('theme', ['pop', 'pop-cream'])
@pytest.mark.parametrize('winner', ['1P', '2P'])
def test_winner_badge_does_not_cover_number(browser: Browser, layout: str, theme: str, winner: str) -> None:
    page = new_page(browser, layout + '&theme=' + theme)
    assert page.locator('#stage').get_attribute('data-pop') == ('cream' if theme == 'pop-cream' else 'black')
    data = snapshot(1.0 if winner == '1P' else 0.0)
    data['terminal'] = dict(state='confirmed', winner=winner)
    page.evaluate('renderSnapshot', data)
    page.wait_for_timeout(400)
    badge = page.locator('#result-tab')
    assert badge.is_visible()
    box = badge.bounding_box()
    for number in page.locator('.num').all():
        rect = number.bounding_box()
        if rect:
            assert (box['x'] + box['width'] <= rect['x'] or rect['x'] + rect['width'] <= box['x']
                    or box['y'] + box['height'] <= rect['y'] or rect['y'] + rect['height'] <= box['y'])
    page.close()
