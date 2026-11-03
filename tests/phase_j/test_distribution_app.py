"""配布版の確認境界・保存・匿名化・独立ポートを実通信で検証する。"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
from typing import Any, Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from src.phase_j.distribution_config import available_port, load_settings, validate_settings
from src.phase_j.distribution_http import make_server
from src.phase_j.distribution_logs import MAX_REPORTS, ReportStore, redact
from src.phase_j.distribution_runtime import DistributionRuntime
from src.phase_j.overlay_config import validate_overlay


@pytest.mark.parametrize('path', [r'C:\Users\Alice\logs', '/home/Alice/logs', 'D:/Users/Bob/app'])
def test_redacts_nested_values_keys_and_other_profiles(path: str) -> None:
    result = redact({'Alice': [path, {'error': 'PC-ONE Alice'}]}, ('Alice', 'PC-ONE'))
    encoded = json.dumps(result)
    assert all(identity not in encoded for identity in ('Alice', 'Bob', 'PC-ONE'))
    assert '[redacted]' in encoded


@pytest.mark.parametrize('settings', [{'mascot': 'yes'}, {'language': 'fr'},
    {'mascot_position': 'top'}, {'mascot_size': 512}, {'mascot_motion': 1}])
def test_invalid_overlay_options_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        validate_overlay(settings)


@pytest.mark.parametrize('settings', [{'port': True}, {'port': 65536}, {'device_index': -1},
    {'language': 'fr'}, {'autostart': 1}, {'relay_url': 'http://evil.invalid'},
    {'relay_url': 'https://secret@evil.invalid'}, {'automatic_logs': True}])
def test_invalid_distribution_settings_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        validate_settings(settings)


@pytest.mark.parametrize('saved, expected', [(None, False), ({}, False),
    ({'check_updates': False}, False), ({'check_updates': True}, True)])
def test_update_check_default_and_saved_choice(tmp_path: Path, saved: dict | None,
                                             expected: bool) -> None:
    """初回と未指定は更新確認OFF、保存済みの明示ON/OFFは維持する。"""
    if saved is not None:
        (tmp_path / 'distribution.json').write_text(json.dumps(saved), encoding='utf-8')
    assert load_settings(tmp_path)['check_updates'] is expected


def test_busy_port_is_not_stolen() -> None:
    with socket.socket() as owner:
        owner.bind(('127.0.0.1', 0))
        port = owner.getsockname()[1]
        assert available_port(port) != port
        assert owner.getsockname()[1] == port


def example_report(store: ReportStore) -> dict[str, Any]:
    return store.report(dict(app='test', model='v6', code_sha='abc'), {'port': 8765},
        dict(type='startup_failure', location='launcher', stack_trace='C:/Users/Alice/app.py'),
        dict(fps=None), ['sample'], [])


def test_preview_and_cancel_never_send(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('src.phase_j.distribution_logs.build_opener', lambda *args: pytest.fail('通信禁止'))
    store = ReportStore(tmp_path)
    preview = store.preview(example_report(store), 'https://example.invalid')
    assert preview['payload']['reports'][0]['error']['type'] == 'startup_failure'
    with pytest.raises(ValueError):
        store.send(preview['token'], False)
    assert not list(tmp_path.glob('pending-*.json'))


def test_failed_send_survives_restart_and_is_in_next_preview(tmp_path: Path) -> None:
    store = ReportStore(tmp_path)
    preview = store.preview(example_report(store), 'invalid')
    assert store.send(preview['token'], True) is False
    with pytest.raises(ValueError):
        store.send(preview['token'], True)
    restarted = ReportStore(tmp_path)
    assert restarted.installation_id == store.installation_id
    next_preview = restarted.preview(example_report(restarted), 'invalid')
    assert len(next_preview['payload']['reports']) == 2
    assert next_preview['payload']['reports'][0] == preview['payload']['reports'][0]


def test_report_retention_is_bounded(tmp_path: Path) -> None:
    store = ReportStore(tmp_path)
    store.save([example_report(store) for _ in range(MAX_REPORTS + 3)])
    assert len(list(tmp_path.glob('pending-*.json'))) == MAX_REPORTS


def test_confirmed_payload_exactly_matches_preview(tmp_path: Path) -> None:
    received: list[dict] = []
    class Receiver(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass
        def do_POST(self) -> None:
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(204)
            self.end_headers()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        store = ReportStore(tmp_path)
        store.save([example_report(store)])
        preview = store.preview(example_report(store), f'http://127.0.0.1:{server.server_port}')
        assert received == []
        assert store.send(preview['token'], True, allow_local=True)
        assert received == [preview['payload']]
        assert not list(tmp_path.glob('pending-*.json'))
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def distribution(tmp_path: Path) -> Iterator[tuple[DistributionRuntime, str]]:
    runtime = DistributionRuntime(tmp_path)
    server = make_server(runtime, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield runtime, f'http://127.0.0.1:{server.server_port}'
    runtime.stop()
    server.shutdown()
    server.server_close()


def post(url: str, body: dict, origin: str | None = None) -> dict:
    headers = {'Content-Type': 'application/json'}
    if origin:
        headers['Origin'] = origin
    request = Request(url, json.dumps(body).encode(), headers)
    with urlopen(request) as response:
        return json.load(response)


def test_cross_origin_send_is_rejected(distribution: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, url = distribution
    with pytest.raises(HTTPError) as caught:
        post(url + '/api/logs/preview', {}, 'https://untrusted.invalid')
    assert caught.value.code == 403
    assert runtime.store.previews == {}


def test_http_save_preview_and_confirmation(distribution: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, url = distribution
    monkeypatch.setattr('src.phase_j.distribution_runtime.shortcuts', lambda *args: None)
    options = dict(language='en', mascot=True, mascot_position='left', mascot_size='large')
    assert post(url + '/api/settings', {'overlay': options})['saved']
    saved = json.loads((runtime.root / 'distribution.json').read_text(encoding='utf-8'))
    assert saved['overlay'] == options
    assert saved['language'] == 'en'
    preview = post(url + '/api/logs/preview', {})
    with pytest.raises(HTTPError):
        post(url + '/api/logs/send', {'token': preview['token']})
    assert not list(runtime.store.directory.glob('pending-*.json'))


def test_http_uninstall_requires_confirmation_and_stops(distribution: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, url = distribution
    launched = []
    monkeypatch.setattr('src.phase_j.distribution_http.launch_uninstaller', launched.append)
    with pytest.raises(HTTPError):
        post(url + '/api/uninstall', {})
    with pytest.raises(HTTPError):
        post(url + '/api/uninstall', {'confirmed': True}, 'https://untrusted.invalid')
    assert not launched and not runtime.stopping
    assert post(url + '/api/uninstall', {'confirmed': True})['uninstalling']
    assert launched == [runtime.root] and runtime.stopping


def test_crash_is_saved_locally_without_network(distribution: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, url = distribution
    monkeypatch.setattr('src.phase_j.distribution_logs.build_opener', lambda *args: pytest.fail('通信禁止'))
    runtime.record_failure('abnormal_exit', 'test stack', 42)
    assert len(list(runtime.store.directory.glob('pending-*.json'))) == 1
    with urlopen(url + '/api/state') as response:
        assert json.load(response)['error'] == 'abnormal_exit'


def test_setup_and_overlay_assets_are_served(distribution: tuple) -> None:
    _, url = distribution
    for path in ('/setup', '/settings', '/', '/overlay_i18n.js', '/overlay_mascot.js', '/setup.js',
                 '/brand/app_logo.png', '/favicon.ico', '/mascot/overlay.webp'):
        with urlopen(url + path) as response:
            assert response.status == 200
            assert len(response.read()) > 0
    with pytest.raises(HTTPError):
        urlopen(url + '/mascot/../../distribution.json')


def test_expired_preview_cannot_send(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ReportStore(tmp_path)
    preview = store.preview(example_report(store), 'https://example.invalid')
    created = store.previews[preview['token']][0]
    monkeypatch.setattr('src.phase_j.distribution_logs.time.time', lambda: created + 901)
    with pytest.raises(ValueError, match='expired'):
        store.send(preview['token'], True)


def test_unclean_shutdown_is_local_and_clean_stop_clears_marker(distribution: tuple) -> None:
    runtime, _ = distribution
    runtime.begin_session()
    assert runtime.error == {}
    runtime.begin_session()
    assert runtime.error['type'] == 'abnormal_exit'
    assert len(list(runtime.store.directory.glob('pending-*.json'))) == 1
    runtime.stop()
    assert not (runtime.logs / 'session.running').exists()


def test_installer_pending_report_is_in_manual_preview(tmp_path: Path) -> None:
    fallback = tmp_path / 'installer'
    fallback.mkdir()
    store = ReportStore(tmp_path / 'runtime', fallback)
    report = example_report(store)
    (fallback / 'pending-install.json').write_text(json.dumps(report), encoding='utf-8-sig')
    preview = store.preview(example_report(store), 'https://example.invalid')
    assert len(preview['payload']['reports']) == 2


@pytest.mark.skipif(os.name != 'nt', reason='Windowsの回復コンソール')
def test_recovery_console_cancel_and_redaction(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    log = tmp_path / 'installer.log'
    computer = os.environ['COMPUTERNAME']
    log.write_text(f'Alice {computer} C:/Users/Bob/app error C:\\Users\\Bob\\app', encoding='utf-8')
    env = {**os.environ, 'LOCALAPPDATA': str(tmp_path), 'USERNAME': 'Alice'}
    result = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        str(root / 'packaging/SendLogs.ps1'), '-LogPath', str(log)], env=env,
        input='CANCEL\n', capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
    assert result.returncode == 0, result.stderr
    assert all(identity not in result.stdout for identity in ('Alice', 'Bob', computer))
    assert '[redacted]' in result.stdout
    assert not list(tmp_path.rglob('pending-*.json'))


def test_stream_relays_original_event_without_changing_payload(distribution: tuple) -> None:
    runtime, url = distribution
    packet = b'event: analysis\ndata: {"display":{"input_status":"no_puyo_screen"},"timing":{"source_available_ms":17}}\n\n'
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            self.wfile.write(packet)
            self.wfile.flush()
    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    runtime.inner_port = upstream.server_port
    try:
        with urlopen(url + '/events', timeout=5) as response:
            assert response.readline() + response.readline() + response.readline() == packet
        assert runtime.latest['timing']['source_available_ms'] == 17
    finally:
        runtime.stopping = True
        upstream.shutdown()
        upstream.server_close()


@pytest.mark.parametrize('extension,mime', [('webp','image/webp'), ('png','image/png'), ('gif','image/gif')])
def test_single_mascot_override(distribution: tuple, extension: str, mime: str) -> None:
    runtime, url = distribution
    directory = runtime.root / 'mascot'
    directory.mkdir()
    target = directory / ('overlay.' + extension)
    target.write_bytes(b'custom mascot')
    with urlopen(url + '/mascot/overlay.webp') as response:
        assert response.headers['Content-Type'] == mime
        assert response.read() == b'custom mascot'
    target.unlink()
    with urlopen(url + '/mascot/overlay.webp') as response:
        assert response.headers['Content-Type'] == 'image/webp'
        assert response.read().startswith(b'RIFF')
