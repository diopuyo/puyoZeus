"""実走とguardの実waitを所有する。guard早期終了時も無監視実走を残さない。"""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
import resource_guard as G

POLL_SECONDS = .1
GUARD_EXIT_TIMEOUT = G.R.INTERVAL_SEC + G.QUERY_TIMEOUT + 5


def code(value: int) -> int:
    return value if value >= 0 else 128 - value


def terminate(child: Any, fd: int) -> None:
    if child.poll() is None:
        try:
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        except ProcessLookupError:
            pass


def reap(child: Any, fd: int | None) -> tuple[int, bool]:
    try:
        return code(child.wait(timeout=GUARD_EXIT_TIMEOUT)), False
    except subprocess.TimeoutExpired:
        if fd is None:
            child.kill()
        else:
            try: signal.pidfd_send_signal(fd, signal.SIGKILL)
            except ProcessLookupError: pass
        return code(child.wait()), True


def run_pair(child_command: list[str], guard_script: str, runner: str,
             resources: Path) -> dict:
    child = subprocess.Popen(child_command)
    fd, guard, error, early = None, None, None, False
    child_code, guard_code, forced = None, None, False
    try:
        fd = os.pidfd_open(child.pid)
        guard = subprocess.Popen([sys.executable, guard_script, str(child.pid), runner, str(resources)])
        print(f'ACTUAL_CHILD_PID={child.pid}', flush=True)
        print(f'ACTUAL_RESOURCE_PID={guard.pid}', flush=True)
        while child.poll() is None and guard.poll() is None:
            time.sleep(POLL_SECONDS)
        if child.poll() is None:
            early = True
            terminate(child, fd)
        child_code, forced = reap(child, fd)
        try:
            guard_code = code(guard.wait(timeout=GUARD_EXIT_TIMEOUT))
        except subprocess.TimeoutExpired:
            error = 'guard_did_not_exit_after_child'
            guard.terminate()  # このPopenで作った未回収のguardだけを終了する。
            guard_code, _ = reap(guard, None)
    except BaseException as failure:
        error = repr(failure)
        if fd is not None:
            terminate(child, fd)
        elif child.poll() is None:
            child.terminate()  # pidfd取得失敗時も自分の未回収子を残さない。
    finally:
        child_code, forced_final = reap(child, fd)
        forced = forced or forced_final
        if guard is not None and guard_code is None:
            if guard.poll() is None: guard.terminate()
            guard_code, _ = reap(guard, None)
        if fd is not None: os.close(fd)
    return dict(child_pid=child.pid, guard_pid=None if guard is None else guard.pid,
        child_exit_code=child_code, resource_guard_exit=guard_code, guard_exited_first=early,
        supervisor_error=error, forced_child_kill=forced, source='actual_Popen_wait', quality_gate_clear=False)


def main() -> int:
    runner, output, guard = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    resources = Path(str(output) + '.resources.jsonl')
    receipt = Path(str(output) + '.supervisor.json')
    if any(p.exists() for p in (output, resources, receipt)):
        raise FileExistsError('exclusive_supervised_target')
    result = run_pair([sys.executable, '-u', runner, '--mode', 'observe', '--output-root', str(output)],
                      guard, runner, resources)
    with receipt.open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)
    if not output.is_dir() or result['resource_guard_exit'] is None:
        return 1  # 起動前失敗も外側票へ保存。未実行waitを捏造しない。
    final = subprocess.run([sys.executable, '-u', runner, '--mode', 'finalize', '--output-root', str(output),
        '--child-exit', str(result['child_exit_code']), '--resource-exit', str(result['resource_guard_exit'])])
    print(f'ACTUAL_FINALIZE_EXIT={final.returncode}', flush=True)
    return final.returncode or int(result['supervisor_error'] is not None)


if __name__ == '__main__':
    raise SystemExit(main())
