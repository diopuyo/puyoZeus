"""先行2レーンの終了を待ち、内部m/候補の55分対を既存試験台で実行する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import ctypes
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'logs/live_parallel'
QUEUE = Path(_public_dev_text('<work>/queue'))
SOAK = Path('C:/puyo1_tools/soak')
POLL_SEC = 15
QUIET_SEC = 60
QUERY = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | "
         "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")


def status(state: str, **values: Any) -> None:
    record = dict(state=state, wall_time=time.time(), **values)
    temporary = OUT/'camera_status.tmp'
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(OUT/'camera_status.json')
    print(json.dumps(record, ensure_ascii=False), flush=True)


def camera_users() -> list[dict]:
    reply = subprocess.run(['powershell.exe', '-NoProfile', '-Command', QUERY],
        capture_output=True, text=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    rows = json.loads(reply.stdout) if reply.stdout.strip() else []
    rows = rows if isinstance(rows, list) else [rows]
    return [row for row in rows if 'vcam_feed.py' in (row.get('CommandLine') or '')]


def prior_status() -> dict:
    """先行レーンの状態ファイル書換中を、完了や失敗と誤認しない。"""
    try:
        return json.loads((QUEUE/'camera_pair_status.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(state='updating')


def wait_camera() -> None:
    """既存の予約待ちとの同時起動を避け、queue完了と連続60秒の解放を待つ。"""
    from live_parallel_probe import free_bytes, MIN_FREE_BYTES
    quiet = None
    while True:
        prior = prior_status()
        if prior['state'] == 'failed':
            raise RuntimeError('先行queue試験が失敗。予約を追い越さず停止')
        held = Path(_public_dev_text('<work>/HOLD_FOR_OBS')).exists()
        users = camera_users()
        free = free_bytes()
        if prior['state'] != 'completed' or users or held or free < MIN_FREE_BYTES:
            quiet = None
        else:
            quiet = quiet or time.monotonic()
        status('waiting_camera', prior=prior['state'], users=users, hold=held, free_bytes=free)
        if quiet is not None and time.monotonic()-quiet >= QUIET_SEC:
            return
        time.sleep(POLL_SEC)


def settings(mode: str, root: Path) -> dict:
    config = json.loads((OUT/f'soak_{mode}'/'PuyoLive/puyo_live.json').read_text())
    config['output_dir'] = str(root/'output')
    (root/'puyo_live.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    return config


def main() -> None:
    os.chdir(ROOT)
    os.environ['PATH'] = 'C:/Windows/System32;C:/Windows;C:/Windows/System32/WindowsPowerShell/v1.0'
    sys.path[:0] = [str(ROOT/'scripts'), str(QUEUE), str(SOAK)]
    import soak_pair as base
    import soak_dshow
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    base.ROOT = OUT
    base.BUNDLES = {mode: OUT/f'soak_{mode}'/'PuyoLive' for mode in ('off', 'on')}
    base.settings = settings
    soak_dshow.CTRL_C = soak_dshow.CTRL_C.replace('GenerateConsoleCtrlEvent(0,0)',
                                                 'GenerateConsoleCtrlEvent(1,0)')
    try:
        for mode in ('off', 'on'):
            wait_camera()
            status('running', mode=mode, seconds=base.SECONDS)
            base.run(mode)
            run = OUT/f'camera_{mode}'
            shutil.copyfile(run/'live.log', run/'soak_stdout.log')
            shutil.copyfile(run/'stopped.json', run/'soak_status.json')
            with (OUT/f'camera_{mode}'/'analysis.txt').open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, str(SOAK/'analyze_soak.py'),
                    str(OUT/f'camera_{mode}')], stdout=log, stderr=log, check=True, creationflags=0x4000)
        status('completed', deployment='品質門未達のため配布pは作成しない')
        subprocess.run([sys.executable, str(ROOT/'scripts/report_live_parallel_soak.py')],
                       check=True, creationflags=0x4000)
    except BaseException as error:
        status('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
