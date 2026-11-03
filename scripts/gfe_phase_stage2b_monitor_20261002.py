"""段2bの実行中に空きメモリとnice値を読み取り記録する。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

OUT = Path('logs/gfe_phase_stage2b_20261002')
RESERVE_BYTES, INTERVAL = 6*1024**3, 10


def sample() -> dict:
    """他worktreeのプロセスには触れず、指定プロセスの状態だけを読む。"""
    entries = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    processes = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = path.read_bytes().replace(b'\x00', b' ').decode()
            if 'python' not in command or '-m scripts.gfe_phase_stage2b_' not in command:
                continue
            pid = int(path.parent.name)
            processes.append(dict(pid=pid, nice=os.getpriority(os.PRIO_PROCESS, pid), command=command))
        except (OSError, UnicodeDecodeError):
            continue
    return dict(time=time.time(), available_bytes=int(entries['MemAvailable'].split()[0])*1024,
                processes=processes)


def main() -> None:
    """10秒間隔で保存し、6GiBを下回ったら本タスクの実験だけを停止する。"""
    os.nice(max(0, 19-os.nice(0)))
    with (OUT/'resources.jsonl').open('a') as stream:
        while True:
            value = sample()
            stream.write(json.dumps(value)+'\n')
            stream.flush()
            if value['available_bytes'] < RESERVE_BYTES:
                import signal
                for process in value['processes']:
                    if process['pid'] != os.getpid():
                        os.kill(process['pid'], signal.SIGTERM)
                raise RuntimeError('空き6GiB未満のため本タスクの実験を停止')
            if not any('stage2b_driver_' in p['command'] for p in value['processes']):
                break
            time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
