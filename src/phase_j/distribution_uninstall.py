"""配布先だけを対象に、停止確認と利用者が選んだデータの片付けを行う。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time

from .distribution_windows import shortcuts

WAIT_SECONDS = 15
POLL_SECONDS = 0.25
USER_DATA = ('output', 'logs', 'distribution.json', 'puyo_live.json', 'install.log',
             'app/config/device_calibration')
TEMPORARY = ('distribution.json.tmp', 'logs/instance.lock', 'logs/session.running')


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
                raise OSError('設定の「アプリを終了」で停止して再実行してください。 / Exit puyoZeus Live in Settings, then retry.')
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
    result = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
        env={**os.environ, 'PUYO_PYTHON': str(executable), 'PUYO_CLEANUP_PID': str(os.getpid())},
        capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.stdout.strip():
        raise OSError('puyoZeus Live を終了して再実行してください。 / Exit puyoZeus Live and retry.')


def cleanup(root: Path, delete_data: bool = False) -> None:
    """記録は既定で残す。一時ファイルとこの配布先のリンクは必ず消す。"""
    root = root.resolve()
    ensure_stopped(root)
    names = list(TEMPORARY) + (list(USER_DATA) if delete_data else [])
    names += [str(p.relative_to(root)) for folder in ('app', 'python')
              for p in (root / folder).rglob('__pycache__')]
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
    elif (root / 'Uninstall.bat').is_file():
        os.startfile(str(root / 'Uninstall.bat'))
    else:
        raise OSError('アンインストーラーがありません / Uninstaller not found')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--delete-data', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    options = parser.parse_args()
    try:
        ensure_stopped(options.root.resolve())
        if options.check:
            return
        if options.interactive:
            answer = input('設定・ログ・記録も削除しますか？ [y/N]\nDelete settings, logs and recordings too? [y/N]: ')
            options.delete_data = answer.strip().lower() == 'y'
        cleanup(options.root, options.delete_data)
        print('登録を解除しました。この画面を閉じ、必要な記録を退避してから展開フォルダを削除してください。\n'
              'Shortcuts and startup removed. Close this window, back up any records you need, then delete the extracted folder.\n'
              + str(options.root.resolve()), flush=True)
    except (OSError, ValueError) as error:
        print(str(error), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
