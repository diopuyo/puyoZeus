"""手元HTTPだけを使う更新と、電源断相当の入替中断・復旧。"""
from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Iterator
import zipfile

import pytest

from src.phase_j import distribution_update as update
from src.phase_j import update_download as fetch
from src.phase_j import update_transaction as tx


@pytest.fixture
def local_http() -> Iterator[tuple[str, dict[str, bytes]]]:
    bodies: dict[str, bytes] = {}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            data = bodies.get(self.path)
            if data is None:
                self.send_error(404)
                return
            assert self.headers['User-Agent'] == fetch.SITE_USER_AGENT
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: object) -> None:
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', bodies
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def archive_bytes(extra: dict[str, bytes] | None = None) -> bytes:
    stream = io.BytesIO()
    files = {'PuyoLive/app/RELEASE.json': b'{"app":"0.6.5"}',
             'PuyoLive/puyoZeus Live.exe': b'new exe',
             'PuyoLive/app/component.py': b'new component'}
    files.update(extra or {})
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, data in files.items():
            info = zipfile.ZipInfo('placeholder')
            info.filename = name
            archive.writestr(info, data)
    return stream.getvalue()


@pytest.mark.parametrize('sums', [False, True])
def test_local_download_extract_and_progress(tmp_path: Path, local_http: tuple, sums: bool) -> None:
    origin, bodies = local_http
    raw = archive_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    bodies['/release.zip'] = raw
    bodies['/SHA256SUMS'] = f'{digest} *release.zip\n'.encode()
    info = dict(url=origin+'/release.zip', name='release.zip', sums_url=origin+'/SHA256SUMS',
                sha256=None if sums else digest)
    progress = []
    cancel = threading.Event()
    fetch.download(info, tmp_path/'download.zip', cancel, lambda a, b: progress.append((a, b)), origin)
    payload = fetch.extract(tmp_path/'download.zip', tmp_path, '0.6.5', cancel)
    assert (payload/'app/component.py').read_bytes() == b'new component'
    assert progress[-1] == (len(raw), len(raw))


@pytest.mark.parametrize('cancelled', [False, True])
def test_bad_sha_and_cancel_leave_install_unchanged(tmp_path: Path, local_http: tuple, cancelled: bool) -> None:
    origin, bodies = local_http
    bodies['/release.zip'] = archive_bytes()
    installed = tmp_path/'installed.exe'
    installed.write_bytes(b'old exe')
    cancel = threading.Event()
    if cancelled:
        cancel.set()
    info = dict(url=origin+'/release.zip', name='release.zip', sha256='0'*64)
    with pytest.raises(fetch.Cancelled if cancelled else ValueError):
        fetch.download(info, tmp_path/'download.zip', cancel, lambda a, b: None, origin)
    assert installed.read_bytes() == b'old exe'
    assert not (tmp_path/'download.zip').exists()


def test_cancel_after_first_chunk(tmp_path: Path, local_http: tuple) -> None:
    origin, bodies = local_http
    raw = b'x' * (fetch.CHUNK_BYTES * 3)
    bodies['/release.zip'] = raw
    event, calls = threading.Event(), []
    def progress(received: int, total: int) -> None:
        calls.append(received)
        event.set()
    with pytest.raises(fetch.Cancelled):
        fetch.download(dict(url=origin+'/release.zip', sha256=hashlib.sha256(raw).hexdigest()),
                       tmp_path/'download.zip', event, progress, origin)
    assert calls == [fetch.CHUNK_BYTES]
    assert not (tmp_path/'download.zip').exists()


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'C:/escape', 'file:stream',
                                  'PuyoLive/../escape', 'PuyoLive\\escape', 'CON.txt', 'trailing.'])
def test_unsafe_zip_rejected(tmp_path: Path, name: str) -> None:
    path = tmp_path/'download.zip'
    path.write_bytes(archive_bytes({name: b'bad'}))
    with pytest.raises(ValueError):
        fetch.extract(path, tmp_path, '0.6.5', threading.Event())
    assert not (tmp_path/'payload').exists()


@pytest.mark.parametrize('url', ['http://github.com/diopuyo/puyoZeus/releases/download/v1/a',
                                'https://evil.test/a', 'https://github.com/other/repo/a',
                                'file:///tmp/update.zip', 'https://user@github.com/a'])
def test_untrusted_download_source(url: str) -> None:
    with pytest.raises(ValueError):
        fetch.validate_url(url)


def test_github_network_is_blocked_by_test_guard() -> None:
    with pytest.raises(AssertionError, match='試験から本番'):
        fetch.open_download(fetch.RELEASE_BASE+'/v0.6.5/puyoZeus-Live-0.6.5.zip')


def test_release_published_latest_metadata() -> None:
    data = dict(releases=[dict(version='0.7.0', status='draft'),
                         dict(version='0.6.5', date='2026-11-10', changes=['one', 'two'])])
    info = fetch.release_info(data, '0.6.4')
    assert info['available'] and info['version'] == '0.6.5'
    assert info['date'] == '2026-11-10' and info['summary'] == 'one\ntwo'
    assert info['url'] == fetch.RELEASE_BASE+'/v0.6.5/puyoZeus-Live-0.6.5.zip'


def installation(tmp_path: Path) -> tuple[Path, Path, Path]:
    root, stage = tmp_path/'installed', tmp_path/'stage'
    root.mkdir()
    payload = stage/'payload'
    payload.mkdir(parents=True)
    for relative, old, new in [('app/code.py', b'old', b'new'),
                                ('app/new.py', None, b'added'),
                                ('distribution.json', b'OBS secret', b'bad defaults'),
                                ('logs/runtime.log', b'log', b'bad logs'),
                                ('mascot/10.webp', b'custom', b'bundled')]:
        if old:
            target = root/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(old)
        source = payload/relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(new)
    return root, stage, payload


def test_transaction_preserves_settings_and_restores(tmp_path: Path) -> None:
    root, stage, payload = installation(tmp_path)
    journal = tx.prepare(root, payload, stage, '0.6.5', 'nonce')
    tx.apply(journal)
    assert (root/'app/code.py').read_bytes() == b'new'
    assert (root/'distribution.json').read_bytes() == b'OBS secret'
    assert (root/'logs/runtime.log').read_bytes() == b'log'
    assert (root/'mascot/10.webp').read_bytes() == b'custom'
    tx.rollback(tx.load_journal(stage/'transaction.json'))
    assert (root/'app/code.py').read_bytes() == b'old'
    assert not (root/'app/new.py').exists()
    assert not (root/tx.MARKER).exists()


@pytest.mark.parametrize('position', [1, 2])
def test_power_loss_at_each_replace_can_restore(tmp_path: Path, position: int) -> None:
    root, stage, payload = installation(tmp_path)
    journal = tx.prepare(root, payload, stage, '0.6.5', 'nonce')
    calls = []
    def crash(relative: str) -> None:
        calls.append(relative)
        if len(calls) == position:
            raise SystemExit('simulated power loss')
    with pytest.raises(SystemExit):
        tx.apply(journal, crash)
    assert (root/tx.MARKER).exists()
    tx.rollback(tx.load_journal(stage/'transaction.json'))
    assert (root/'app/code.py').read_bytes() == b'old'
    assert not (root/'app/new.py').exists()
    tx.rollback(tx.load_journal(stage/'transaction.json'))
    assert (root/'distribution.json').read_bytes() == b'OBS secret'


@pytest.mark.parametrize('relative', ['python312.dll', 'python/_ctypes.pyd', 'python/_socket.pyd'])
def test_python_bootstrap_change_rejected_before_modification(tmp_path: Path, relative: str) -> None:
    root, stage, payload = installation(tmp_path)
    (root/relative).parent.mkdir(parents=True, exist_ok=True)
    (payload/relative).parent.mkdir(parents=True, exist_ok=True)
    (root/relative).write_bytes(b'old python')
    (payload/relative).write_bytes(b'new python')
    with pytest.raises(ValueError, match='Python'):
        tx.prepare(root, payload, stage, '0.6.5', 'nonce')
    assert (root/relative).read_bytes() == b'old python'
    assert not (root/tx.MARKER).exists()


def test_staged_changes_and_corrupt_backup_detected(tmp_path: Path) -> None:
    root, stage, payload = installation(tmp_path)
    journal = tx.prepare(root, payload, stage, '0.6.5', 'nonce')
    (payload/'app/code.py').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Staged'):
        tx.apply(journal)
    assert (root/'app/code.py').read_bytes() == b'old'
    (payload/'app/code.py').write_bytes(b'new')
    tx.apply(journal)
    (stage/'backup/app/code.py').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Backup'):
        tx.rollback(journal)
    assert (root/tx.MARKER).exists()


def test_controller_initialization_never_fetches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object) -> None:
        raise AssertionError('Unexpected network request')
    monkeypatch.setattr(fetch, 'open_download', forbidden)
    controller = update.UpdateController(SimpleNamespace(root=tmp_path))
    assert controller.snapshot()['status'] == 'idle'
    monkeypatch.delenv('PUYO_UPDATE_BUTTON', raising=False)
    assert update.enabled() is True
    monkeypatch.setenv('PUYO_UPDATE_BUTTON', '0')
    assert update.enabled() is False
    monkeypatch.setenv('PUYO_UPDATE_BUTTON', '1')
    assert update.enabled() is True


def test_recovery_bootstrap_runs_before_cleanup() -> None:
    source = Path(__file__).parents[2]/'packaging/launcher/signed_launcher.py'
    text = source.read_text(encoding='utf8')
    assert text.index('recover_before_launch(root)') < text.index('cleanup(root)')


def test_completed_child_result_reaches_open_window(tmp_path: Path) -> None:
    controller = update.UpdateController(SimpleNamespace(root=tmp_path))
    assert controller.snapshot()['status'] == 'idle'
    tx.write_json(tmp_path/tx.RESULT, dict(status='complete', version='0.6.5'))
    assert controller.snapshot() == dict(status='complete', version='0.6.5')


def test_recovery_retries_after_interrupted_rollback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, stage, payload = installation(tmp_path)
    journal = tx.prepare(root, payload, stage, '0.6.5', 'nonce')
    tx.apply(journal)
    original = tx.atomic_copy
    def crash(source: Path, target: Path) -> None:
        original(source, target)
        raise SystemExit('power loss during recovery')
    monkeypatch.setattr(tx, 'atomic_copy', crash)
    with pytest.raises(SystemExit):
        tx.rollback(journal)
    monkeypatch.setattr(tx, 'atomic_copy', original)
    tx.rollback(tx.load_journal(stage/'transaction.json'))
    assert (root/'app/code.py').read_bytes() == b'old'
    assert not (root/'app/new.py').exists()
    assert not (root/tx.MARKER).exists()


def test_deep_windows_paths_extract_replace_and_restore(tmp_path: Path) -> None:
    relative = '/'.join(['deep_directory'] * 20) + '/component.py'
    archive = tmp_path/'download.zip'
    archive.write_bytes(archive_bytes({'PuyoLive/'+relative: b'new'}))
    stage = tmp_path/'stage'
    stage.mkdir()
    payload = fetch.extract(archive, stage, '0.6.5', threading.Event())
    root = tmp_path/'installed'
    root.mkdir()
    old = tx.filesystem_path(root/relative)
    old.parent.mkdir(parents=True)
    old.write_bytes(b'old')
    journal = tx.prepare(root, payload, stage, '0.6.5', 'deep-path')
    tx.apply(journal)
    assert old.read_bytes() == b'new'
    tx.rollback(journal)
    assert old.read_bytes() == b'old'


def test_backup_alone_is_not_enough_space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, stage, payload = installation(tmp_path)
    monkeypatch.setattr(tx.shutil, 'disk_usage', lambda path: SimpleNamespace(free=len(b'old')))
    with pytest.raises(OSError, match='空き容量'):
        tx.prepare(root, payload, stage, '0.6.5', 'space')
    assert (root/'app/code.py').read_bytes() == b'old'
    assert not (root/tx.MARKER).exists()
