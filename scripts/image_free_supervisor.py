"""WSL比較ジョブをPuyoLive稼働中はプロセス群ごと停止する監督器。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.obs_priority_guard import QUERY
from scripts.compare_image_free_all import save_json

POLL_SECONDS = 5
QUERY_TIMEOUT_SECONDS = 15


def live_running() -> tuple[bool, str]:
    """監視不能も停止側に倒し、重い処理を継続しない。"""
    try:
        result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', QUERY],
            check=True, capture_output=True, text=True, timeout=QUERY_TIMEOUT_SECONDS)
        raw = result.stdout.strip()
        pids = json.loads(raw) if raw else []
        return bool(pids), f'PuyoLive={pids}'
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        return True, f'監視不能: {error}'


def supervise(config: Path, output: Path) -> int:
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = None
        paused = False
        while child is None or child.poll() is None:
            active, reason = live_running()
            if child is None and not active:
                child = subprocess.Popen([sys.executable, '-m', 'scripts.compare_image_free_all',
                    '--config', str(config)], start_new_session=True)
            if child is not None and active != paused:
                os.killpg(child.pid, signal.SIGSTOP if active else signal.SIGCONT)
                paused = active
            save_json(output/'supervisor.json', dict(supervisor_pid=os.getpid(),
                worker_pid=child.pid if child else None, paused=active, reason=reason,
                updated_at=time.time(), status='paused' if active else 'running'))
            time.sleep(POLL_SECONDS)
        save_json(output/'supervisor.json', dict(supervisor_pid=os.getpid(), worker_pid=child.pid,
            status='finished' if child.returncode == 0 else 'failed', exit_code=child.returncode,
            updated_at=time.time()))
        return child.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(supervise(args.config, args.output))


if __name__ == '__main__':
    main()
