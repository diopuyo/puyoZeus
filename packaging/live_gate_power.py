"""元の電源プランを改変せず、複製だけを使う。独立監視と次回ログオンでも復元する。"""
from __future__ import annotations
import argparse
import ctypes as ct
from ctypes import wintypes as wt
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import winreg

RUNONCE = r'Software\Microsoft\Windows\CurrentVersion\RunOnce'
REG_NAME = '!PuyoLiveGateRestore20261022'
BOOST = 'be337238-0d82-4146-a960-4f3749d470c7'
SUB_PROCESSOR = '54533251-82be-4824-96c1-47b60b740d00'
GUID = re.compile(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')


def write(path: Path, data: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def power(*args: str) -> str:
    result = subprocess.run(['powercfg', *args], capture_output=True, text=True, errors='replace')
    if result.returncode:
        raise RuntimeError(f'powercfg {args}: {result.stderr or result.stdout}')
    return result.stdout


def active() -> str:
    match = GUID.search(power('/getactivescheme'))
    if match is None:
        raise RuntimeError('現在の電源プランを確認できない')
    return match.group().lower()


def register_recovery(path: Path) -> None:
    command = subprocess.list2cmdline([sys.executable, '-B', str(Path(__file__).resolve()),
                                     '--restore', str(path.resolve())])
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUNONCE) as key:
        winreg.SetValueEx(key, REG_NAME, 0, winreg.REG_SZ, command)


def restore(path: Path) -> None:
    """監視プロセスだけが復元する。失敗時は復元票とRunOnceを残す。"""
    state = json.loads(path.read_text(encoding='utf-8'))
    if state.get('restored'):
        return
    try:
        (path.parent/'monitor.stop').touch()
        power('/setactive', state['original'])
        restored = active()
        unchanged = power('/qh', state['original']) == state['original_query']
        if restored != state['original'] or not unchanged:
            raise RuntimeError('元プランの再選択・設定一致を確認できない')
        # 作成前に死んだ場合の「存在しない」だけを許す。
        schemes = power('/list').lower()
        if state['temporary'] in schemes:
            power('/delete', state['temporary'])
        state.update(restored=True, restored_at=time.time(), active_after=restored,
                     original_settings_unchanged=unchanged)
        write(path, state)
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUNONCE) as key:
            try:
                winreg.DeleteValue(key, REG_NAME)
            except FileNotFoundError:
                pass
    except BaseException as error:
        state.update(restore_error=repr(error), restored=False)
        write(path, state)
        raise


def watch(path: Path, pid: int) -> None:
    api = ct.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = (wt.DWORD, wt.BOOL, wt.DWORD)
    api.OpenProcess.restype = wt.HANDLE
    api.WaitForSingleObject.argtypes = (wt.HANDLE, wt.DWORD)
    api.CloseHandle.argtypes = (wt.HANDLE,)
    handle = api.OpenProcess(0x100000, False, pid)
    if not handle:
        restore(path)
        return
    path.with_suffix('.ready').write_text(str(pid), encoding='ascii')
    try:
        while api.WaitForSingleObject(handle, 1000) == 258:
            if path.with_suffix('.stop').exists():
                break
    finally:
        api.CloseHandle(handle)
        restore(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', type=Path)
    parser.add_argument('--watch', type=int)
    parser.add_argument('--state', type=Path)
    options = parser.parse_args()
    if options.restore:
        restore(options.restore)
    elif options.watch and options.state:
        watch(options.state, options.watch)
    else:
        parser.error('復元票が必須')
