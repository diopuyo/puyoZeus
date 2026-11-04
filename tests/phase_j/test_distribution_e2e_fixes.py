"""実物の窓・OBSを使わず、E2Eで見つかった不具合を再現する。"""
from __future__ import annotations

import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pytest

from src.phase_j.distribution_http import DistributionHandler, make_server
from src.phase_j.distribution_ports import inspect_conflict, switch_port
from src.phase_j.distribution_runtime import DistributionRuntime
from src.phase_j.local_http import LocalHTTPHandler, LocalHTTPServer


@pytest.mark.parametrize('error', [ConnectionAbortedError, ConnectionResetError, BrokenPipeError])
@pytest.mark.parametrize('headers', [False, True])
def test_disconnected_response_is_quiet(error: type[OSError], headers: bool,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    handler = object.__new__(DistributionHandler)
    handler.send_response = lambda *a: None
    handler.send_header = lambda *a: None
    def disconnected(*args: Any) -> None:
        raise error('切断済み')
    handler.end_headers = disconnected if headers else lambda: None
    handler.wfile = SimpleNamespace(write=disconnected)
    handler.respond(b'{}')
    assert handler.close_connection
    monkeypatch.setattr('http.server.BaseHTTPRequestHandler.handle', disconnected)
    handler.close_connection = False
    handler.handle()
    assert handler.close_connection


def test_python_runtime_configs(tmp_path: Path) -> None:
    from build_release import write_python_runtime_configs
    (tmp_path / 'python').mkdir()
    write_python_runtime_configs(tmp_path)
    for name in ('python.exe.config', 'pythonw.exe.config'):
        text = (tmp_path / 'python' / name).read_bytes().decode('ascii')
        assert ET.fromstring(text).find('runtime/loadFromRemoteSources').attrib == {'enabled': 'true'}


@pytest.fixture
def old_app(tmp_path: Path) -> Any:
    root = tmp_path / 'old'
    requests: list[str] = []
    class OldApp(LocalHTTPHandler):
        # 公開済み0.6.0はBaseHTTPRequestHandlerの標準見出しを使う。
        server_version = 'BaseHTTP/0.6'
        sys_version = 'Python/3.12.3'

        def local_request(self) -> bool:
            host = f'127.0.0.1:{self.server.server_port}'
            return self.headers.get('Host') == host and self.headers.get('Origin', f'http://{host}') == f'http://{host}'

        def log_message(self, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            assert self.local_request()
            body = json.dumps(dict(root=str(root), settings={}, running=True, diagnostics={},
                                   os_language='ja', url=f'http://127.0.0.1:{self.server.server_port}/',
                                   error=None, update={})).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            assert self.local_request()
            assert self.headers.get('Origin') is None
            assert self.headers.get('Content-Type') == 'application/json'
            requests.append(self.path)
            self.send_response(200)
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'{}')
            threading.Thread(target=self.server.shutdown, daemon=True).start()
    server = LocalHTTPServer(('127.0.0.1', 0), OldApp)
    def serve() -> None:
        server.serve_forever()
        server.server_close()
    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    yield root, server, requests
    server.shutdown()
    worker.join()
    server.server_close()


def test_port_handover_to_requested_port(old_app: tuple, tmp_path: Path) -> None:
    root, old, requests = old_app
    runtime = DistributionRuntime(tmp_path / 'new')
    current = make_server(runtime, old.server_port)
    worker = threading.Thread(target=current.serve_forever, daemon=True)
    worker.start()
    replacement = None
    try:
        assert runtime.port_conflict == dict(kind='different', root=str(root), port=old.server_port)
        assert runtime.port != old.server_port
        request = Request(f'http://127.0.0.1:{current.server_port}/api/port/switch', data=b'{}',
                          headers={'Content-Type': 'application/json'})
        with urlopen(request) as response:
            result = json.load(response)
        replacement = current.next_server
        assert runtime.port == old.server_port == replacement.server_port
        assert result['url'] == f'http://127.0.0.1:{old.server_port}/settings'
        assert requests == ['/api/stop'] and runtime.port_conflict is None
        thread = threading.Thread(target=replacement.serve_forever, daemon=True)
        thread.start()
        with urlopen(f'http://127.0.0.1:{old.server_port}/api/state') as response:
            assert json.load(response)['root'] == str(runtime.root)
        replacement.shutdown()
        thread.join()
    finally:
        current.shutdown()
        worker.join()
        current.server_close()
        if replacement:
            replacement.server_close()
        runtime.stop()


def test_same_root_is_not_offered_for_takeover(old_app: tuple) -> None:
    root, old, requests = old_app
    runtime = DistributionRuntime(root)
    current = make_server(runtime, old.server_port)
    try:
        assert runtime.port_conflict['kind'] == 'same'
        with pytest.raises(ValueError, match='No other app'):
            switch_port(current, runtime)
        assert requests == []
    finally:
        current.server_close()
        runtime.stop()


def test_application_keeps_window_alive_during_port_switch(old_app: tuple, tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from argparse import Namespace
    from src.phase_j import distribution_app as app
    _, old, requests = old_app
    runtime = DistributionRuntime(tmp_path / 'new')
    runtime.settings.update(check_updates=False, port=old.server_port)
    visited: list[str] = []
    class FakeWindow:
        def __init__(self, url: str, *args: Any) -> None:
            self.url = self.default_url = url
            self.finished = threading.Event()

        def open(self) -> None:
            pass

        def run(self) -> None:
            base = self.url.removesuffix('/settings')
            request = Request(base + '/api/port/switch', data=b'{}', headers={'Content-Type': 'application/json'})
            with urlopen(request) as response:
                target = json.load(response)['url']
            with urlopen(target.removesuffix('/settings') + '/api/state') as response:
                assert json.load(response)['root'] == str(runtime.root)
            assert not self.finished.is_set()
            assert self.default_url == target
            visited.append(target)
            stop = Request(target.removesuffix('/settings') + '/api/stop', data=b'{}',
                           headers={'Content-Type': 'application/json'})
            with urlopen(stop) as response:
                assert json.load(response)['stopped']
            assert self.finished.wait(5)
    monkeypatch.setattr(app, 'AppWindow', FakeWindow)
    monkeypatch.setattr(app, 'migrate_shortcuts', lambda *args: None)
    monkeypatch.setattr(app.signal, 'signal', lambda *args: None)
    monkeypatch.setattr('src.phase_j.distribution_tray.start_tray', lambda *args: None)
    app.run_application(runtime, Namespace(port=None, setup=False, no_browser=True, autostart=False))
    assert visited == [f'http://127.0.0.1:{old.server_port}/settings']
    assert requests == ['/api/stop']


def test_non_app_port_is_not_stopped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    response = io.BytesIO(b'{"root":"C:/foreign","settings":{}}')
    response.headers = {'Server': 'Other software'}
    monkeypatch.setattr('src.phase_j.distribution_ports.urlopen', lambda *a, **k: response)
    assert inspect_conflict(tmp_path, 8765) == dict(kind='other', port=8765)


def test_obs_updates_existing_source_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import distribution_obs as obs
    calls: list[tuple[str, dict]] = []
    name = 'puyoZeus Live abcdef01'
    class FakeOBS:
        def __init__(self, *args: Any) -> None:
            pass

        def close(self) -> None:
            pass

        def request(self, kind: str, **data: Any) -> dict:
            calls.append((kind, data))
            return {'GetCurrentProgramScene': dict(currentProgramSceneName='Game'),
                    'GetSceneItemList': dict(sceneItems=[dict(sourceName=name, inputKind='browser_source')]),
                    'GetInputSettings': dict(inputSettings=dict(url='http://127.0.0.1:8765/')),
                    'GetInputList': dict(inputs=[dict(inputName=name)]),
                    'GetVirtualCamStatus': dict(outputActive=True)}.get(kind, {})
    monkeypatch.setattr(obs, 'ObsClient', FakeOBS)
    result = obs.integrate(dict(obs_enabled=True, obs_port=4455, obs_password=''), 'http://127.0.0.1:8766/', True)
    assert result['already_added']
    assert not any(kind in ('CreateInput', 'CreateSceneItem') for kind, _ in calls)
    assert ('SetInputSettings', dict(inputName=name, inputSettings=dict(url='http://127.0.0.1:8766/'), overlay=True)) in calls
    def unavailable(*args: Any, **kwargs: Any) -> dict:
        raise ValueError('Target scene unavailable')
    monkeypatch.setattr(FakeOBS, 'request', unavailable)
    with pytest.raises(ValueError, match='Target scene unavailable'):
        obs.integrate(dict(obs_enabled=True, obs_port=4455, obs_password=''), 'http://127.0.0.1:8766/', True)


def test_common_shortcut_ownership(tmp_path: Path) -> None:
    from src.phase_j import distribution_windows as windows
    directories = windows.shortcut_directories()
    first, second = tmp_path / 'first', tmp_path / 'second'
    for root in (first, second):
        root.mkdir()
        (root / 'puyoZeus Live.exe').touch()
    settings = dict.fromkeys(directories, True)
    windows.shortcuts(first, settings)
    windows.shortcuts(second, settings)
    windows.migrate_shortcuts(first, settings)
    DistributionRuntime(first).configure({'device_name': 'Fake camera'})
    windows.shortcuts(first, settings, remove=True)
    assert all((p / 'puyoZeus Live.lnk').exists() for p in directories.values())
    windows.shortcuts(second, settings, remove=True)
    assert all(not (p / 'puyoZeus Live.lnk').exists() for p in directories.values())


def test_legacy_shortcuts_are_removed_only_when_owned(tmp_path: Path) -> None:
    from src.phase_j import distribution_windows as windows
    settings = dict.fromkeys(windows.shortcut_directories(), True)
    windows.shortcuts(tmp_path, settings)
    for directory in windows.shortcut_directories().values():
        link = directory / windows.shortcut_name(tmp_path)
        for name in windows.legacy_names(tmp_path):
            (directory / name).write_bytes(link.read_bytes())
    windows.migrate_shortcuts(tmp_path, settings)
    for directory in windows.shortcut_directories().values():
        assert (directory / 'puyoZeus Live.lnk').exists()
        assert not any((directory / name).exists() for name in windows.legacy_names(tmp_path))


def test_shortcut_guard_rejects_real_directory() -> None:
    import os
    import subprocess
    real_link = Path(os.environ['APPDATA']) / 'Microsoft/Windows/Start Menu/Programs/puyoZeus-guard.lnk'
    with pytest.raises(AssertionError, match='本物のショートカット'):
        real_link.write_text('これは書かれない', encoding='utf-8')
    with pytest.raises(AssertionError, match='本物のショートカット'):
        subprocess.run(['cmd.exe', '/c', 'exit', '0'], env={**os.environ, 'PUYO_LINK': str(real_link)}, check=True)
