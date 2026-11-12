"""別プロセスで動く更新台本。標準ライブラリだけで入替と復旧を行う。"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Callable, Any

MARKER = 'logs/update_pending.json'
RESULT = 'logs/update_result.json'
HEALTH = 'logs/update_health.json'
PRESERVED = {'distribution.json', 'logs', 'output', 'mascot', 'user_config.json', 'install.log'}
BOOTSTRAP = ('tools/launcher/update_recovery.py', 'tools/launcher/signed_launcher.py')
WAIT_SECONDS = 60
POLL_SECONDS = 0.2
BUFFER_BYTES = 256 * 1024
SYNCHRONIZE = 0x00100000
ERROR_INVALID_PARAMETER = 87
MILLISECONDS_PER_SECOND = 1000
HTTP_TIMEOUT_SECONDS = 5


def filesystem_path(path: Path) -> Path:
    """深い展開先もWin32の260文字制限に掛けずに扱う。"""
    if os.name != 'nt':
        return path
    value = str(path.absolute())
    if value.startswith('\\\\?\\'):
        return path
    if value.startswith('\\\\'):
        return Path('\\\\?\\UNC\\' + value[2:])
    return Path('\\\\?\\' + value)


def write_json(path: Path, value: Any) -> None:
    path = filesystem_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest(path: Path) -> str:
    with filesystem_path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_target(root: Path, relative: str) -> Path:
    root = filesystem_path(root)
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError('Unsafe update target')
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink() or (hasattr(parent, 'is_junction') and parent.is_junction()):
            raise ValueError('Update target contains a link')
    return path


def atomic_copy(source: Path, target: Path) -> None:
    source, target = filesystem_path(source), filesystem_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.puyo-update-tmp')
    with source.open('rb') as incoming, temporary.open('wb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, BUFFER_BYTES)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    os.replace(temporary, target)


def pinned(relative: str) -> bool:
    """復旧入口が使うPython本体の変更は手動更新に案内する。"""
    path = Path(relative)
    if relative == 'tools/launcher/sitecustomize.py':
        return True
    return (len(path.parts) == 1 or path.parent == Path('python')) and path.suffix.lower() in {
        '.dll', '.pyd', '.zip', '._pth'} or relative in ('python/python.exe', 'python/pythonw.exe')


def required_space(root: Path, payload: Path, files: list[dict]) -> int:
    """退避だけでなく、入替用一時ファイルと復旧にも空きを残す。"""
    backup, growth, temporary = 0, 0, 0
    for item in files:
        old = filesystem_path(root / item['path']).stat().st_size if item['old'] else 0
        new = filesystem_path(payload / item['path']).stat().st_size
        backup += old
        growth += max(0, new - old)
        temporary = max(temporary, old, new)
    return backup + growth + temporary


def prepare(root: Path, payload: Path, stage: Path, version: str, token: str) -> dict:
    """アプリがまだ動いている間に退避を完成させ、入替の予定を固定する。"""
    if stage.parent.resolve() != root.parent.resolve() or (root / MARKER).exists():
        raise ValueError('更新が保留中です / An update is already pending')
    files = []
    for source in sorted(filesystem_path(payload).rglob('*')):
        if not source.is_file():
            continue
        relative = source.relative_to(filesystem_path(payload)).as_posix()
        if relative.split('/')[0].casefold() in PRESERVED:
            continue
        target = safe_target(root, relative)
        old_hash = digest(target) if target.is_file() else None
        new_hash = digest(source)
        if old_hash == new_hash:
            continue
        if pinned(relative):
            raise ValueError('Python本体の変更はZIPの手動展開で更新してください / Extract the ZIP manually to update Python')
        files.append(dict(path=relative, old=old_hash, new=new_hash))
    size = required_space(root, payload, files)
    if shutil.disk_usage(stage).free < size:
        raise OSError('退避用の空き容量がありません / Not enough backup space')
    for item in files:
        if item['old']:
            atomic_copy(root / item['path'], stage / 'backup' / item['path'])
            if digest(stage / 'backup' / item['path']) != item['old']:
                raise OSError('Backup verification failed')
    files.sort(key=lambda f: (0 if f['path'] in BOOTSTRAP else 1,
                             BOOTSTRAP.index(f['path']) if f['path'] in BOOTSTRAP else f['path']))
    journal = dict(root=str(root), stage=str(stage), payload=str(payload), version=version,
                   token=token, state='prepared', files=files, touched=[])
    write_json(stage / 'transaction.json', journal)
    return journal


def load_journal(path: Path) -> dict:
    journal = json.loads(path.read_text(encoding='utf8'))
    stage, root, payload = (Path(journal[k]).resolve() for k in ('stage', 'root', 'payload'))
    if path.resolve() != stage / 'transaction.json' or stage.parent != root.parent or not payload.is_relative_to(stage):
        raise ValueError('Invalid update journal location')
    for item in journal['files']:
        safe_target(root, item['path'])
        safe_target(payload, item['path'])
    return journal


def apply(journal: dict, checkpoint: Callable[[str], None] | None = None) -> None:
    root, stage, payload = (Path(journal[k]) for k in ('root', 'stage', 'payload'))
    journal['state'] = 'applying'
    write_json(stage / 'transaction.json', journal)
    write_json(root / MARKER, dict(stage=str(stage), token=journal['token']))
    for item in journal['files']:
        relative = item['path']
        if digest(payload / relative) != item['new']:
            raise ValueError('Staged file changed')
        journal['touched'].append(relative)
        write_json(stage / 'transaction.json', journal)
        atomic_copy(payload / relative, safe_target(root, relative))
        if checkpoint:
            checkpoint(relative)
    journal['state'] = 'verifying'
    write_json(stage / 'transaction.json', journal)


def rollback(journal: dict) -> None:
    root, stage = Path(journal['root']), Path(journal['stage'])
    journal['state'] = 'rolling_back'
    write_json(stage / 'transaction.json', journal)
    touched = set(journal['touched'])
    for item in reversed(journal['files']):
        if item['path'] not in touched:
            continue
        target = safe_target(root, item['path'])
        if item['old']:
            backup = stage / 'backup' / item['path']
            if digest(backup) != item['old']:
                raise ValueError('Backup changed; manual recovery required')
            atomic_copy(backup, target)
            if digest(target) != item['old']:
                raise OSError('Rollback verification failed')
        else:
            target.unlink(missing_ok=True)
        target.with_name(target.name + '.puyo-update-tmp').unlink(missing_ok=True)
    journal['state'] = 'rolled_back'
    write_json(stage / 'transaction.json', journal)
    write_json(root / RESULT, dict(status='rolled_back', version=journal['version']))
    (root / MARKER).unlink(missing_ok=True)


def wait_parent(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == 'nt':
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(SYNCHRONIZE, False, pid)
        if not handle:
            if ctypes.get_last_error() == ERROR_INVALID_PARAMETER:
                return
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if kernel.WaitForSingleObject(handle, WAIT_SECONDS * MILLISECONDS_PER_SECOND) != 0:
                raise TimeoutError('アプリの終了を待てませんでした / App did not exit')
        finally:
            kernel.CloseHandle(handle)
    else:
        deadline = time.monotonic() + WAIT_SECONDS
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(POLL_SECONDS)
        raise TimeoutError('App did not exit')


def start_app(journal: dict, token: bool = False) -> subprocess.Popen:
    root = Path(journal['root'])
    environment = dict(os.environ)
    environment.pop('PUYO_UPDATE_TOKEN', None)
    if token:
        environment['PUYO_UPDATE_TOKEN'] = journal['token']
    return subprocess.Popen([str(root / 'puyoZeus Live.exe')], cwd=root, env=environment)


def verify_start(journal: dict) -> None:
    health = Path(journal['root']) / HEALTH
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            record = json.loads(health.read_text(encoding='utf8'))
            if record.get('token') == journal['token'] and record.get('version') == journal['version']:
                return
        except (OSError, ValueError):
            pass
        time.sleep(POLL_SECONDS)
    raise TimeoutError('新版の起動を確認できませんでした / New version did not start')


def lock_stage(stage: Path) -> Any:
    stream = (stage / 'worker.lock').open('a+b')
    stream.write(b'0')
    stream.flush()
    stream.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return stream


def main() -> int:
    path, pid, recover = Path(sys.argv[1]), int(sys.argv[2]), '--recover' in sys.argv[3:]
    journal = load_journal(path)
    stage, root = Path(journal['stage']), Path(journal['root'])
    with lock_stage(stage):
        if recover:
            rollback(journal)
            return 0
        try:
            wait_parent(pid)
            apply(journal)
            start_app(journal, token=True)
            verify_start(journal)
            journal['state'] = 'complete'
            write_json(path, journal)
            write_json(root / RESULT, dict(status='complete', version=journal['version']))
            (root / MARKER).unlink(missing_ok=True)
        except Exception as error:
            # 新版が生きていれば専用HTTPの終了要求で閉じ、ファイルのロックを解放する。
            stop_new_app(journal)
            try:
                rollback(journal)
                start_app(journal)
            except Exception as rollback_error:
                write_json(root / RESULT, dict(status='recovery_failed', message=str(rollback_error)))
            write_json(stage / 'failure.json', dict(message=str(error)))
            return 1
    return 0


def stop_new_app(journal: dict) -> None:
    from urllib.request import Request, urlopen
    try:
        health = json.loads((Path(journal['root']) / HEALTH).read_text(encoding='utf8'))
        if health.get('token') != journal['token']:
            return
        port = int(health['port'])
        with urlopen(Request(f'http://127.0.0.1:{port}/api/stop', data=b'{}',
                             headers={'Content-Type': 'application/json'}), timeout=HTTP_TIMEOUT_SECONDS):
            pass
        wait_parent(int(health['pid']))
    except (OSError, ValueError, KeyError):
        pass


if __name__ == '__main__':
    raise SystemExit(main())
