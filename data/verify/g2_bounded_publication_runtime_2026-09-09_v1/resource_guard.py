"""対象PIDの同一性を固定し、RAM圧迫時だけ対象実走を停止する。"""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import sys
import time

RSS_LIMIT_KIB = 8 * 1024 * 1024
AVAILABLE_MIN_KIB = 2 * 1024 * 1024
INTERVAL_SEC = 10


def identity(proc: Path, runner: str) -> tuple[str, str]:
    command = (proc / 'cmdline').read_bytes().split(b'\0')
    index = 2 if len(command) > 1 and command[1] == b'-u' else 1
    if len(command) <= index or command[index].decode() != runner:
        raise ValueError('resource_guard_process_identity')
    start = (proc / 'stat').read_text().split(') ', 1)[1].split()[19]
    return command[index].decode(), start


def numbers(path: Path) -> dict[str, int]:
    return {parts[0].rstrip(':'): int(parts[1]) for line in path.read_text().splitlines()
        if len(parts := line.split()) == 3 and parts[2] == 'kB'}


def main() -> int:
    pid, runner, output = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    proc = Path('/proc') / str(pid)
    with output.open('x') as stream:
        initial = identity(proc, runner)
        while proc.exists():
            try:
                if identity(proc, runner) != initial:
                    raise ValueError('resource_guard_pid_reused')
                rss = numbers(proc / 'status')['VmRSS']
                available = numbers(Path('/proc/meminfo'))['MemAvailable']
            except (FileNotFoundError, ProcessLookupError, KeyError, ValueError):
                break
            stop = rss > RSS_LIMIT_KIB or available < AVAILABLE_MIN_KIB
            stream.write(json.dumps(dict(time=time.time(), pid=pid, rss_kib=rss, available_kib=available,
                rss_limit_kib=RSS_LIMIT_KIB, minimum_available_kib=AVAILABLE_MIN_KIB, safety_stop=stop)) + '\n')
            stream.flush()
            if stop:
                if identity(proc, runner) == initial:
                    os.kill(pid, signal.SIGTERM)
                return 0
            time.sleep(INTERVAL_SEC)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
