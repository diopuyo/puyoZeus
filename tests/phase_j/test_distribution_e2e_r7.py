"""実物の窓やOBSを使わず、早期終了とポートの持ち主を確認する。"""
from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.phase_j import distribution_app as app
from src.phase_j import distribution_instance as instance
from src.phase_j import distribution_port_owner as owner
from src.phase_j import distribution_ports as ports
from src.phase_j.distribution_window import AppWindow
from src.phase_j.local_http import LocalHTTPServer, LocalHTTPHandler
from playwright.sync_api import Browser
from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_overlay_browser import browser


@pytest.mark.parametrize('phase', ['before', 'initialize', 'create', 'shown'])
def test_stop_during_gui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    window = AppWindow('http://localhost/settings', tmp_path, True)
    events = SimpleNamespace(closing=MockEvent(), before_show=MockEvent(), shown=MockEvent())
    native = SimpleNamespace(events=events, destroy=Mock())
    def initialize(renderer: str) -> SimpleNamespace:
        if phase == 'initialize':
            window.finished.set()
        return SimpleNamespace(renderer=renderer)
    def create(*args: object, **kwargs: object) -> object:
        if phase == 'create':
            window.finished.set()
        return native
    def start(**kwargs: object) -> None:
        window.finished.set()
        window.shown()
    view = SimpleNamespace(initialize=Mock(side_effect=initialize), create_window=Mock(side_effect=create),
                           start=Mock(side_effect=start))
    monkeypatch.setitem(sys.modules, 'webview', view)
    if phase == 'before':
        window.finished.set()
    window.run()
    assert window.finished.is_set()
    assert view.start.call_count == int(phase == 'shown')
    if phase == 'shown':
        assert native.destroy.called and window.exiting


class MockEvent:
    def __iadd__(self, callback: object) -> MockEvent:
        return self


def test_shutdown_closes_socket_before_gui_finishes(tmp_path: Path) -> None:
    server = LocalHTTPServer(('127.0.0.1', 0), LocalHTTPHandler)
    address = server.server_address
    window = AppWindow('http://localhost', tmp_path, False)
    deadline = Mock()
    worker = threading.Thread(target=app.serve_application,
                              args=(server, window, SimpleNamespace(port=address[1]), deadline))
    worker.start()
    server.shutdown()
    worker.join(timeout=2)
    assert not worker.is_alive() and window.finished.is_set()
    deadline.start.assert_called_once()
    with socket.socket() as probe:
        probe.settimeout(1)
        assert probe.connect_ex(address) != 0


def test_record_actual_port(tmp_path: Path) -> None:
    instance.record_port(tmp_path, 8766)
    assert instance.recorded_port(tmp_path, 8765) == 8766
    instance.record_port(tmp_path, 8765)
    assert instance.recorded_port(tmp_path, 8766) == 8765


@pytest.mark.parametrize('content', ['', '{}', '{', '{"port": true}', '{"port": 70000}'])
def test_invalid_record_falls_back(tmp_path: Path, content: str) -> None:
    path = tmp_path / instance.INSTANCE_FILE
    path.parent.mkdir()
    path.write_text(content, encoding='utf-8')
    assert instance.recorded_port(tmp_path, 8765) == 8765


@pytest.mark.parametrize('marked,executable', [(True, 'pythonw.exe'), (False, 'pythonw.exe'), (True, 'other.exe')])
def test_owner_requires_product(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                marked: bool, executable: str) -> None:
    marker = tmp_path / owner.MARKER
    if marked:
        marker.parent.mkdir(parents=True)
        marker.touch()
    identity = dict(pid=123, executable=str(tmp_path / 'python' / executable), created='1234')
    monkeypatch.setattr(owner, 'query_owner', lambda *args: identity)
    found = owner.inspect_owner(8765)
    assert bool(found) == (marked and executable == 'pythonw.exe')


def test_unresponsive_probe_and_verified_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / owner.MARKER
    marker.parent.mkdir(parents=True)
    marker.touch()
    identity = dict(pid=123, executable=str(tmp_path / 'python/pythonw.exe'), created='1234')
    query = Mock(return_value=identity)
    monkeypatch.setattr(owner, 'query_owner', query)
    monkeypatch.setattr(ports, 'urlopen', Mock(side_effect=TimeoutError))
    conflict = ports.inspect_conflict(tmp_path / 'other', 8765)
    assert conflict['kind'] == 'unresponsive'
    ports.stop_conflict(conflict)
    query.assert_called_with(8765, identity)


def test_changed_owner_never_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    query = Mock()
    monkeypatch.setattr(owner, 'query_owner', query)
    monkeypatch.setattr(owner, 'inspect_owner', lambda port: None)
    with pytest.raises(ValueError):
        owner.stop_owner(dict(port=8765, kind='unresponsive'))
    query.assert_not_called()


def test_foreign_user_or_missing_owner_is_other(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(owner, 'query_owner', lambda *args: None)
    monkeypatch.setattr(ports, 'urlopen', Mock(side_effect=TimeoutError))
    assert ports.inspect_conflict(Path('C:/unused'), 8765) == dict(kind='other', port=8765)


def test_uninstaller_uses_actual_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import distribution_uninstall as uninstall
    instance.record_port(tmp_path, 8766)
    monkeypatch.setattr(uninstall, 'ensure_stopped', Mock(side_effect=[OSError(), None]))
    monkeypatch.setattr(uninstall, 'checked_path', lambda root, name: root / name)
    monkeypatch.setattr('builtins.input', lambda prompt: 'y')
    inspect = Mock(return_value=dict(kind='same'))
    monkeypatch.setattr(ports, 'inspect_conflict', inspect)
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    request = Mock(return_value=response)
    monkeypatch.setattr(uninstall, 'urlopen', request)
    assert uninstall.stop_interactively(tmp_path)
    inspect.assert_called_once_with(tmp_path, 8766)
    assert request.call_args.args[0].full_url == 'http://127.0.0.1:8766/api/stop'


@pytest.mark.parametrize('phase', ['before', 'during', 'after'])
def test_hung_gui_process_exits_by_deadline(tmp_path: Path, phase: str) -> None:
    """GUI相当の無限待機が残っても、自分の試験プロセスだけが期限内に終わる。"""
    script = f'''
import sys
sys.path[:] = {sys.path!r}
import os, threading
from pathlib import Path
from types import SimpleNamespace
from src.phase_j.distribution_app import serve_application
from src.phase_j.distribution_window import AppWindow
from src.phase_j.local_http import LocalHTTPServer, LocalHTTPHandler
server = LocalHTTPServer(('127.0.0.1', 0), LocalHTTPHandler)
window = AppWindow('http://localhost', Path({str(tmp_path)!r}), False)
if {phase!r} == 'after': window.ready.set()
deadline = threading.Timer(0.3, lambda: os._exit(0))
threading.Thread(target=serve_application, args=(server, window, SimpleNamespace(), deadline), daemon=True).start()
server.shutdown()
assert window.finished.wait(2)
assert server.socket.fileno() == -1
threading.Event().wait(20)
'''
    result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, timeout=8)
    assert result.returncode == 0, result.stderr.decode(errors='replace')


def test_switch_unresponsive_updates_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conflict = dict(kind='unresponsive', port=8765, root=str(tmp_path / 'old'))
    runtime = SimpleNamespace(control_lock=threading.Lock(), root=tmp_path, port=8766,
                              port_conflict=conflict)
    server = SimpleNamespace(RequestHandlerClass=LocalHTTPHandler, shutdown=Mock())
    replacement = SimpleNamespace(server_port=8765)
    monkeypatch.setattr(ports, 'inspect_conflict', lambda *args: conflict)
    stopped = Mock()
    monkeypatch.setattr(ports, 'stop_conflict', stopped)
    monkeypatch.setattr('src.phase_j.local_http.LocalHTTPServer', lambda *args: replacement)
    assert ports.switch_port(server, runtime)['url'] == 'http://127.0.0.1:8765/settings'
    assert app.active_server(server) is replacement
    assert instance.recorded_port(tmp_path, 9999) == 8765
    stopped.assert_called_once_with(conflict)


def test_query_refuses_foreign_user(monkeypatch: pytest.MonkeyPatch) -> None:
    result = SimpleNamespace(returncode=1, stdout='')
    run = Mock(return_value=result)
    monkeypatch.setattr(owner.subprocess, 'run', run)
    assert owner.query_owner(8765) is None
    assert json.loads(run.call_args.kwargs['input']) == dict(port=8765, stop=False)
    assert 'GetCurrent().User.Value' in run.call_args.args[0][-1]


@pytest.mark.parametrize('language', ['ja', 'en'])
def test_unresponsive_guidance(distribution: tuple, browser: Browser, language: str) -> None:
    _, url = distribution
    page = browser.new_page()
    page.goto(url + '/settings')
    page.locator('#tab-other').click()
    page.locator('[name=language]').select_option(language)
    page.locator('#tab-stream').click()
    state = dict(running=False, url=url, settings={}, diagnostics={},
                 port_conflict=dict(kind='unresponsive', root='C:/synthetic', port=8765))
    page.evaluate('state=>window.renderDistributionUX(state)', state)
    panel = page.locator('#port-conflict')
    assert ('Unresponsive puyoZeus Live' if language == 'en' else '応答しない puyoZeus Live') in panel.inner_text()
    assert 'C:/synthetic' in panel.inner_text()
    assert panel.locator('button').is_visible()
    page.close()
