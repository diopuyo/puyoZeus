"""配布先だけを対象に、停止確認と利用者が選んだデータの片付けを行う。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.request import Request, urlopen

from .distribution_windows import shortcuts, system_tool

WAIT_SECONDS = 15
POLL_SECONDS = 0.25
USER_DATA = ('output', 'output_history', 'logs', 'distribution.json', 'puyo_live.json', 'install.log',
             'app/config/device_calibration')
TEMPORARY = ('distribution.json.tmp', 'logs/instance.lock', 'logs/session.running')
PRODUCT_MARKERS = ('app/RELEASE.json', 'app/MANIFEST.json', 'app/src/phase_j/distribution_app.py')


def validate_root(root: Path) -> Path:
    """削除前に製品の配置を確かめ、ホームやドライブ全体を拒否する。"""
    if any(part.is_symlink() or part.is_junction() for part in (root, *root.parents)):
        raise ValueError('配布先のリンクは使用できません / Linked distribution root')
    root = root.resolve()
    if root == Path(root.anchor) or root == Path.home().resolve():
        raise ValueError('削除できない場所です / Unsafe distribution root')
    if not all(checked_path(root, name).is_file() for name in PRODUCT_MARKERS):
        raise ValueError('配布の目印がありません / Missing distribution markers')
    return root


def checked_path(root: Path, relative: str) -> Path:
    """リンク・ジャンクションを経由した配布先外への削除を拒否する。"""
    path = root / relative
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink() or part.is_junction():
            raise ValueError(f'リンク先は削除できません / Linked path: {part}')
    if not path.resolve().is_relative_to(root):
        raise ValueError('配布フォルダ外です / Outside distribution folder')
    return path


def ensure_stopped(root: Path, wait_seconds: float = WAIT_SECONDS) -> None:
    """ロックを実際に取得できるか確認し、動作中なら変更せず終了する。"""
    root = validate_root(root)
    import msvcrt
    path = checked_path(root, 'logs/instance.lock')
    deadline = time.monotonic() + wait_seconds
    while path.exists():
        try:
            with path.open('r+b') as stream:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise OSError('トレイの「終了」で停止して再実行してください。 / Exit puyoZeus Live from the tray, then retry.')
            time.sleep(POLL_SECONDS)
    ensure_runtime_stopped(root)


def ensure_runtime_stopped(root: Path) -> None:
    """直接起動された同梱Pythonも検出し、別配布先のプロセスは無視する。"""
    executable = root / 'python/python.exe'
    if not executable.exists():
        return
    script = ('Get-CimInstance Win32_Process | Where-Object { '
              '$_.ExecutablePath -eq $env:PUYO_PYTHON -and '
              '$_.ProcessId -ne [int]$env:PUYO_CLEANUP_PID } | '
              'ForEach-Object { $_.ProcessId }')
    result = subprocess.run([system_tool('powershell'), '-NoProfile', '-NonInteractive', '-Command', script],
        env={**os.environ, 'PUYO_PYTHON': str(executable), 'PUYO_CLEANUP_PID': str(os.getpid())},
        capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.stdout.strip():
        raise OSError('puyoZeus Live を終了して再実行してください。 / Exit puyoZeus Live and retry.')


def cleanup(root: Path, delete_data: bool = False) -> None:
    """記録は既定で残す。一時ファイルとこの配布先のリンクは必ず消す。"""
    root = validate_root(root)
    ensure_stopped(root)
    names = list(TEMPORARY) + (list(USER_DATA) if delete_data else [])
    names += [str(p.relative_to(root)) for p in (root / 'app/config/device_calibration').glob('*.tmp')]
    paths = [checked_path(root, name) for name in names]
    shortcuts(root, {}, remove=True)
    for path in paths:
        if path.is_dir():
            # 子のジャンクションも事前検査し、他の配布先を巻き込まない。
            for child in path.rglob('*'):
                checked_path(root, str(child.relative_to(root)))
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def launch_uninstaller(root: Path) -> None:
    """ブラウザから指定パスを受け取らず、同梱の起動口だけを開く。"""
    installer = root / 'unins000.exe'
    if installer.is_file():
        subprocess.Popen([str(installer)], cwd=root)
    elif (root / 'アンインストール.exe').is_file():
        os.startfile(str(root / 'アンインストール.exe'))
    elif (root / 'アンインストール.bat').is_file():
        os.startfile(str(root / 'アンインストール.bat'))
    elif (root / 'Uninstall.bat').is_file():
        os.startfile(str(root / 'Uninstall.bat'))
    else:
        raise OSError('アンインストーラーがありません / Uninstaller not found')


def stop_interactively(root: Path) -> bool:
    """この展開先だけを照合して終了を頼み、ロックの解放を待つ。"""
    from .distribution_config import DEFAULT_PORT
    from .distribution_ports import inspect_conflict, PROBE_TIMEOUT
    try:
        ensure_stopped(root, wait_seconds=0)
        return True
    except OSError:
        answer = input('puyoZeus Live が動いています。終了して続けますか？ [Y/n]\n'
                       'puyoZeus Live is running. Exit and continue? [Y/n]: ')
        if answer.strip().lower() not in ('', 'y'):
            return False
    path = checked_path(root, 'distribution.json')
    settings = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    from .distribution_instance import recorded_port
    port = recorded_port(root, settings.get('port', DEFAULT_PORT))
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('設定のポートが不正です / Invalid settings port')
    if inspect_conflict(root, port).get('kind') != 'same':
        raise ValueError('この展開先のアプリを確認できません。トレイの「終了」で停止してください。 / Exit this app from the tray.')
    request = Request(f'http://127.0.0.1:{port}/api/stop', data=b'{}',
                      headers={'Content-Type': 'application/json', 'Host': f'127.0.0.1:{port}'})
    with urlopen(request, timeout=PROBE_TIMEOUT) as response:
        response.read(1024)
    deadline = time.monotonic() + WAIT_SECONDS
    while True:
        try:
            ensure_stopped(root, wait_seconds=0)
            return True
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(POLL_SECONDS)


def validate_tree(root: Path) -> Path:
    """全体削除では利用者が追加した子も含めてリンクを拒否する。"""
    root = validate_root(root)
    for parent in root.parents:
        if parent.is_symlink() or parent.is_junction():
            raise ValueError('配布先の親がリンクです / Linked parent')
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            checked_path(root, str((Path(directory) / name).relative_to(root)))
    return root


def schedule_folder_removal(root: Path) -> None:
    """外部のPowerShellで両起動元の終了を待ち、再検査して削除する。"""
    root = validate_tree(root)
    from .distribution_remove import REMOVAL_SCRIPT
    subprocess.Popen([system_tool('powershell'), '-NoProfile', '-NonInteractive',
        '-Command', REMOVAL_SCRIPT],
        env={**os.environ, 'PUYO_REMOVE_ROOT': str(root), 'PUYO_REMOVE_PID': str(os.getpid()),
             'PUYO_REMOVE_PARENT': str(os.getppid())},
        cwd=tempfile.gettempdir(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--delete-data', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    options = parser.parse_args()
    try:
        root = validate_root(options.root)
        if options.interactive:
            if not stop_interactively(root):
                return
        else:
            ensure_stopped(root)
        if options.check:
            return
        if options.interactive:
            answer = input(f'設定・ログ・記録も含めて、このフォルダ ({root}) ごと削除しますか？ [y/N]\n'
                           f'Delete this entire folder ({root}), including settings, logs and recordings? [y/N]: ')
            options.delete_data = answer.strip().lower() == 'y'
        if options.delete_data:
            validate_tree(root)
        cleanup(root)
        if options.delete_data:
            schedule_folder_removal(root)
            print('登録を解除しました。この画面を閉じるとフォルダを削除します。\n'
                  'Registration removed. The folder will be deleted after this window closes.', flush=True)
        else:
            print('設定と記録はフォルダに残しました。不要になったらフォルダごと削除してください。\n'
                  'Settings and recordings remain in the folder. Delete the folder when no longer needed.', flush=True)
    except (OSError, ValueError) as error:
        print(str(error), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
