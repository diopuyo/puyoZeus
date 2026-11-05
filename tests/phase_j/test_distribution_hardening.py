"""再起動・ローカルHTTP・保存・資格情報の最終チェック回帰試験。"""
from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import threading
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from src.phase_j import distribution_runtime as runtime_module
from src.phase_j.distribution_config import load_settings, save_settings, validate_settings
from src.phase_j.distribution_http import DistributionHandler
from src.phase_j.distribution_logs import ReportStore, redact
from src.phase_j import distribution_output as output_module
from src.phase_j.distribution_runtime import DistributionRuntime
from tests.phase_j.test_distribution_app import distribution, post, example_report


@pytest.fixture(autouse=True)
def packaging_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'packaging'))


@pytest.mark.parametrize('legacy', [False, True])
def test_output_restart_preserves_previous_and_allows_exclusive_files(tmp_path: Path, legacy: bool) -> None:
    output = tmp_path / 'output'
    output.mkdir()
    names = ['live_24h.jsonl', 'b3_time_gate.jsonl', 'b3_resources.jsonl', 'ama_updates.jsonl', 'v49f_updates.jsonl']
    if legacy:
        names += ['runtime.jsonl', 'status.json', 'summary.json']
    for name in names:
        (output / name).write_text('previous', encoding='utf-8')
    for _ in range(2):
        output_module.prepare_output(tmp_path)
        assert not list(output.iterdir())
        for name in names:
            with (output / name).open('x', encoding='utf-8') as stream:
                stream.write('next')
    histories = sorted((tmp_path / 'output_history').iterdir())
    assert len(histories) == 2
    assert all((histories[0] / name).read_text() == 'previous' for name in names)
    assert all((histories[1] / name).read_text() == 'next' for name in names)


def test_history_count_and_byte_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(output_module, 'MAX_HISTORY_RUNS', 2)
    monkeypatch.setattr(output_module, 'MAX_HISTORY_BYTES', 5)
    for _ in range(5):
        output_module.prepare_output(tmp_path)
        (tmp_path / 'output/value').write_bytes(b'123')
    histories = list((tmp_path / 'output_history').iterdir())
    assert len(histories) == 1
    assert sum(p.stat().st_size for p in histories[0].iterdir()) == 3


def test_reports_read_current_output_after_rotation(tmp_path: Path) -> None:
    runtime = DistributionRuntime(tmp_path)
    for label in ('previous', 'current'):
        output_module.prepare_output(tmp_path)
        for name in ('runtime.jsonl', 'status.json', 'summary.json'):
            (tmp_path / 'output' / name).write_text(json.dumps({'run': label}) + '\n', encoding='utf-8')
        report = runtime.report()['self_diagnostics']
        assert all(report[name] == {'run': label} for name in ('runtime', 'status.json', 'summary.json'))
    history = next((tmp_path / 'output_history').iterdir())
    assert json.loads((history / 'summary.json').read_text()) == {'run': 'previous'}


@pytest.mark.parametrize('route', ['/settings', '/setup', '/api/state', '/latest', '/brand/app_logo.png', '/absent'])
def test_response_headers_and_cross_site_rejection(distribution: tuple, route: str) -> None:
    _, url = distribution
    try:
        response = urlopen(url + route)
    except HTTPError as error:
        response = error
    with response:
        assert response.headers['Content-Security-Policy'] == "frame-ancestors 'self'"
        assert response.headers['X-Frame-Options'] == 'SAMEORIGIN'
        assert response.headers['Cross-Origin-Resource-Policy'] == 'same-origin'
        assert 'Python' not in response.headers['Server']
    with pytest.raises(HTTPError) as denied:
        urlopen(Request(url + route, headers={'Sec-Fetch-Site': 'cross-site'}))
    assert denied.value.code == 403


@pytest.mark.parametrize('payload', [[], None, True, 'text', 1])
def test_json_must_be_object(distribution: tuple, payload: Any) -> None:
    _, url = distribution
    with pytest.raises(HTTPError) as denied:
        post(url + '/api/obs', payload)
    assert denied.value.code == 400
    assert 'JSON object' in json.loads(denied.value.read())['error']


def test_unexpected_post_error_is_json(distribution: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    _, url = distribution
    def fail(*args: Any) -> None:
        raise RuntimeError('private details')
    monkeypatch.setattr(DistributionHandler, 'action', fail)
    with pytest.raises(HTTPError) as failed:
        post(url + '/api/settings', {})
    assert failed.value.code == 500
    assert 'private details' not in json.loads(failed.value.read())['error']


def test_shortcut_failure_keeps_saved_settings(distribution: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, url = distribution
    def fail(*args: Any) -> None:
        raise subprocess.CalledProcessError(1, 'powershell')
    monkeypatch.setattr(runtime_module, 'shortcuts', fail)
    result = post(url + '/api/settings', {'language': 'en', 'autostart': True})
    assert result['saved'] and result['warning']
    assert load_settings(runtime.root)['autostart'] is True
    assert runtime.settings['language'] == 'en'


def test_saving_settings_does_not_block_snapshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = DistributionRuntime(tmp_path)
    entered, release, accepted = threading.Event(), threading.Event(), threading.Event()
    def slow(*args: Any) -> None:
        entered.set()
        assert release.wait(5)
    monkeypatch.setattr(runtime_module, 'shortcuts', slow)
    worker = threading.Thread(target=runtime.configure, args=({'language': 'en', 'autostart': True},))
    worker.start()
    try:
        assert entered.wait(2)
        snapshot = threading.Thread(target=lambda: (runtime.accept_snapshot({'display': {'status': 'ready'}}), accepted.set()))
        snapshot.start()
        assert accepted.wait(1)
        snapshot.join()
    finally:
        release.set()
        worker.join(5)


def test_heartbeat_does_not_replace_latest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = DistributionRuntime(tmp_path)
    runtime.inner_port = 12345
    handler = DistributionHandler.__new__(DistributionHandler)
    handler.runtime, handler.wfile = runtime, io.BytesIO()
    data = {'display': {'status': 'ready'}}
    packet = ('event: analysis\ndata: ' + json.dumps(data) + '\n\nevent: heartbeat\ndata: {"ts": 123}\n\n').encode()
    monkeypatch.setattr('src.phase_j.distribution_http.urlopen', lambda *args, **kwargs: io.BytesIO(packet))
    handler.relay_events()
    assert runtime.latest == data
    assert handler.wfile.getvalue() == packet


@pytest.mark.parametrize('password', ['ordinary-secret', '日本語🔒パスワード'])
def test_dpapi_roundtrip_migration_and_clear(tmp_path: Path, password: str) -> None:
    path = tmp_path / 'distribution.json'
    path.write_text(json.dumps({'obs_password': password}), encoding='utf-8')
    settings = load_settings(tmp_path)
    assert settings['obs_password'] == password
    protected = json.loads(path.read_text(encoding='utf-8'))['obs_password']
    assert protected['protection'] == 'dpapi' and password not in path.read_text(encoding='utf-8')
    assert load_settings(tmp_path)['obs_password'] == password
    save_settings(tmp_path, {**settings, 'obs_password': ''})
    assert load_settings(tmp_path)['obs_password'] == ''
    assert json.loads(path.read_text(encoding='utf-8'))['obs_password'] == ''


@pytest.mark.parametrize('raw', ['broken', '[]', '{"port": false}'])
def test_invalid_settings_are_preserved_before_defaults(tmp_path: Path, raw: str) -> None:
    path = tmp_path / 'distribution.json'
    path.write_text(raw, encoding='utf-8')
    runtime = DistributionRuntime(tmp_path)
    backups = list(tmp_path.glob('distribution.invalid-*.json'))
    assert len(backups) == 1 and backups[0].read_text(encoding='utf-8') == raw
    assert not path.exists()
    assert runtime.error['type'] == 'configuration_error'


@pytest.mark.parametrize('raw', ['broken', '{}', '[]', '{"report_id": null}'])
def test_corrupt_pending_is_quarantined_and_valid_reports_remain(tmp_path: Path, raw: str) -> None:
    store = ReportStore(tmp_path)
    (tmp_path / 'pending-broken.json').write_text(raw, encoding='utf-8')
    valid = example_report(store)
    store.save([valid])
    preview = store.preview(example_report(store), 'https://example.invalid')
    assert len(preview['payload']['reports']) == 2
    assert len(list(tmp_path.glob('invalid-*.json'))) == 1
    assert len(list(tmp_path.glob('pending-*.json'))) == 1
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('text', [r'C:\\Users\\Alice\\OneDrive - Example Corp\\log',
    r'\\NAS-SERVER\PRIVATE-SHARE\log', 'alice@example.com password="private secret"',
    r'D:\Owner\App\log', r'D:\\Owner\\App\\log'])
def test_redaction_handles_escaped_paths_and_inline_secrets(text: str) -> None:
    safe = redact(text, ('Alice', 'a'), root=Path('D:/Owner/App'))
    assert all(word not in safe for word in ('Alice', 'Example Corp', 'NAS-SERVER', 'PRIVATE-SHARE',
                                             'alice@example.com', 'private secret', 'Owner'))
    assert redact('abnormal_exit cat concatenate', ('a', 'cat')) == 'abnormal_exit [redacted] concatenate'


@pytest.mark.parametrize('version,available', [('0.6.2', True), ('0.6.1', False), ('0.5.9', False),
    ('0.6.2rc1', True), (None, False), ('', False), ('broken', False)])
def test_update_version_order_missing_and_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             version: str | None, available: bool) -> None:
    runtime = DistributionRuntime(tmp_path)
    runtime.version = {'app': '0.6.1'}
    requests = []
    class Opener:
        def open(self, *args: Any, **kwargs: Any) -> io.BytesIO:
            requests.append(args)
            return io.BytesIO(json.dumps({'version': version}).encode())
    monkeypatch.setattr(runtime_module, 'build_opener', lambda *args: Opener())
    runtime.check_update()
    runtime.check_update()
    assert len(requests) == 1 and runtime.update['available'] is available


def test_autostart_retries_and_can_be_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = DistributionRuntime(tmp_path)
    calls = []
    def start() -> None:
        calls.append(True)
        if len(calls) == 2:
            runtime.child = object()
    monkeypatch.setattr(runtime, 'start', start)
    monkeypatch.setattr(runtime_module, 'AUTOSTART_RETRY_SECONDS', 0)
    runtime.start_autostart()
    assert len(calls) == 2 and not runtime.input_waiting
    runtime.child = None
    monkeypatch.setattr(runtime, 'start', lambda: runtime.stop(final=False))
    runtime.start_autostart()
    assert runtime.cancel_start.is_set() and not runtime.input_waiting


def test_shortcuts_only_rewrite_changed_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import distribution_windows as windows
    locations = {name: tmp_path / name for name in ('desktop_shortcut', 'start_menu_shortcut', 'autostart')}
    monkeypatch.setattr(windows, 'shortcut_directories', lambda: locations)
    settings = validate_settings({'autostart': True})
    calls, real_run = [], windows.subprocess.run
    def run(command: list[str], **kwargs: Any) -> Any:
        calls.append(command)
        assert Path(command[0]).is_absolute()
        return real_run(command, **kwargs)
    monkeypatch.setattr(windows.subprocess, 'run', run)
    windows.shortcuts(tmp_path, settings)
    link = locations['autostart'] / windows.shortcut_name(tmp_path)
    stamp = link.stat().st_mtime_ns
    windows.shortcuts(tmp_path, settings)
    assert len(calls) == 1 and link.stat().st_mtime_ns == stamp
    (tmp_path / 'logs/shortcuts.json').unlink()
    windows.shortcuts(tmp_path, settings)
    assert len(calls) == 2 and link.stat().st_mtime_ns == stamp
    windows.shortcuts(tmp_path, {**settings, 'autostart': True})
    assert len(calls) == 2 and (locations['autostart'] / windows.shortcut_name(tmp_path)).exists()


def test_send_logs_window_returns_to_settings(tmp_path: Path) -> None:
    from unittest.mock import Mock
    from src.phase_j.distribution_window import AppWindow
    app = AppWindow('http://127.0.0.1:8765/settings', tmp_path, True)
    app.window = Mock()
    app.ready.set()
    app.open('/settings#logs')
    app.window.load_url.assert_called_with('http://127.0.0.1:8765/settings#logs')
    app.open()
    app.window.load_url.assert_called_with('http://127.0.0.1:8765/settings')


def test_output_symlink_rejected_without_touching_target(tmp_path: Path) -> None:
    import os
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'keep').write_text('unchanged')
    output = tmp_path / 'install/output'
    output.parent.mkdir()
    # Windowsで管理者権限を要しないジャンクションを使う。
    result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(output), str(outside)], capture_output=True)
    assert result.returncode == 0
    try:
        with pytest.raises(OSError, match='Output links'):
            output_module.prepare_output(output.parent)
        assert (outside / 'keep').read_text() == 'unchanged'
    finally:
        os.rmdir(output)


def test_release_prunes_only_offline_recognition_script(tmp_path: Path) -> None:
    from build_release import prune_release_development_files
    scripts = tmp_path / 'app/scripts'
    (scripts / '__pycache__').mkdir(parents=True)
    for name in ('visualize_recognition.py', '__pycache__/visualize_recognition.cpython-312.pyc',
                 'run_live_pipeline_20260928.py', 'visualize_advantage_overlay.py'):
        (scripts / name).write_text('test')
    prune_release_development_files(tmp_path)
    assert not (scripts / 'visualize_recognition.py').exists()
    assert not list((scripts / '__pycache__').iterdir())
    assert (scripts / 'run_live_pipeline_20260928.py').exists()
    assert (scripts / 'visualize_advantage_overlay.py').exists()


@pytest.mark.parametrize('path', ['C:/Users/Example/.gemini/antigravity/scratch/puyo_analyzer/data/rows.json',
    'C:/Users/Example/.codex/worktrees/abc/puyo_analyzer/data/rows.json', '/mnt/d/puyo_analyzer/data/rows.json',
    '<build>/.gemini/antigravity/scratch/puyo_analyzer/data/rows.json'])
def test_provenance_hides_workspace_layout(path: str) -> None:
    from release_privacy import portable_path
    assert portable_path(path) == '<build>/data/rows.json'
