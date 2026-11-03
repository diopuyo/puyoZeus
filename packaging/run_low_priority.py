"""Windows検査を低優先度で直列実行し、共有PCの空きメモリを守る。"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

GIB = 1024 ** 3
START_FREE_GIB = 10.25
WAIT_FREE_GIB = 8.0
MIN_FREE_GIB = 6.0
POLL_SECONDS = 10
IDLE_PRIORITY_CLASS = 0x40


class MemoryStatus(ctypes.Structure):
    """GlobalMemoryStatusExのWindows ABI。"""
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in
        ('total', 'available', 'page_total', 'page_available', 'virtual_total',
         'virtual_available', 'extended')]


def available_gib() -> float:
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return status.available / GIB


def run(plan: dict) -> None:
    """余裕を確保して開始し、下限割れ時は自分の検査processだけを止める。"""
    start_free = max(WAIT_FREE_GIB, float(plan.get('start_free_gib', START_FREE_GIB)))
    while available_gib() < start_free:
        print(f'メモリ待機: {available_gib():.2f} GiB', flush=True)
        time.sleep(POLL_SECONDS)
    path = Path(plan['log'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as log:
        child = subprocess.Popen(plan['command'], cwd=plan['cwd'], stdout=log,
                                 stderr=subprocess.STDOUT, creationflags=IDLE_PRIORITY_CLASS,
                                 env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1',
                                      **plan.get('env', {})})
        minimum = available_gib()
        while child.poll() is None:
            minimum = min(minimum, available_gib())
            if minimum < MIN_FREE_GIB:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                               capture_output=True, check=False)
                raise RuntimeError('空き6GiBを割ったため、この検査だけを停止しました')
            time.sleep(POLL_SECONDS)
    result = dict(exit_code=child.returncode, min_free_gib=minimum, command=plan['command'])
    path.with_suffix('.result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    if child.returncode:
        raise SystemExit(child.returncode)


if __name__ == '__main__':
    run(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig')))
