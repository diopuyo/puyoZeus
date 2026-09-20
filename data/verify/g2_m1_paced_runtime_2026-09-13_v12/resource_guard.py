"""元RAM監視を再用し、GPU熱制限/取得失敗時に所有pidfdだけへTERMする。"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import signal
import select
import subprocess
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'g2_bounded_publication_runtime_2026-09-09_v1/resource_guard.py'
SPEC = importlib.util.spec_from_file_location('_g2_original_ram_guard', SOURCE)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
SCHEMA = 'g2_ram_thermal_guard_v1'
NVIDIA = '/usr/lib/wsl/lib/nvidia-smi'
QUERY_TIMEOUT = 3.0


def thermal(text: str) -> dict:
    rows = list(line for line in text.splitlines() if line.strip())
    if len(rows) != 1:
        raise ValueError('thermal_gpu_rows')
    parts = [v.strip() for v in rows[0].split(',')]
    if len(parts) != 4 or parts[0] != '0' or not parts[1].isdigit():
        raise ValueError('thermal_gpu_columns')
    if any(v not in ('Active', 'Not Active') for v in parts[2:]):
        raise ValueError('thermal_gpu_flags')
    return dict(gpu0_temperature_c=int(parts[1]), hw_thermal=parts[2], sw_thermal=parts[3],
                nvidia_smi_status='ok', nvidia_smi_error=None)


def gpu() -> dict:
    query = '--query-gpu=index,temperature.gpu,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown'
    result = subprocess.run([NVIDIA, query, '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=QUERY_TIMEOUT, check=True)
    return thermal(result.stdout)


def sample(proc: Path, read_gpu: Callable[[], dict] = gpu) -> dict:
    rss = R.numbers(proc / 'status')['VmRSS']
    available = R.numbers(Path('/proc/meminfo'))['MemAvailable']
    try:
        values = read_gpu()
    except Exception as error:
        values = dict(gpu0_temperature_c=None, hw_thermal=None, sw_thermal=None,
                      nvidia_smi_status='error', nvidia_smi_error=repr(error))
    reasons = []
    if rss > R.RSS_LIMIT_KIB: reasons.append('rss_limit')
    if available < R.AVAILABLE_MIN_KIB: reasons.append('available_memory')
    if values['nvidia_smi_status'] != 'ok': reasons.append('thermal_unavailable')
    if 'Active' in (values['hw_thermal'], values['sw_thermal']): reasons.append('thermal_slowdown')
    return dict(schema=SCHEMA, time=time.time(), pid=int(proc.name), rss_kib=rss,
        available_kib=available, rss_limit_kib=R.RSS_LIMIT_KIB, minimum_available_kib=R.AVAILABLE_MIN_KIB,
        safety_stop=bool(reasons), stop_reasons=reasons, cpu_package_temperature_c=None, **values)


def same_live(proc: Path, runner: str, initial: tuple, fd: int) -> bool:
    try:
        current = R.identity(proc, runner)
    except (FileNotFoundError, ProcessLookupError):
        return False
    except ValueError:
        if select.select([fd], [], [], 0)[0]:
            return False
        raise
    if current != initial:
        raise ValueError('thermal_guard_pid_changed')
    return True


def monitor(pid: int, runner: str, output: Path, *, interval: float = R.INTERVAL_SEC,
            read_gpu: Callable[[], dict] = gpu) -> int:
    proc = Path('/proc') / str(pid)
    initial = R.identity(proc, runner)
    fd = os.pidfd_open(pid)
    try:
        if not same_live(proc, runner, initial, fd):
            return 0
        with output.open('x') as stream:
            while proc.exists():
                try:
                    if not same_live(proc, runner, initial, fd):
                        return 0
                    row = sample(proc, read_gpu)
                except (FileNotFoundError, ProcessLookupError, KeyError):
                    return 0
                row['process_start_ticks'] = initial[1]
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                if row['safety_stop']:
                    try:
                        if same_live(proc, runner, initial, fd):
                            signal.pidfd_send_signal(fd, signal.SIGTERM)
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                    return 0  # 実停止/終了codeは親のwaitが検証する。
                time.sleep(interval)
        return 0
    finally:
        os.close(fd)


if __name__ == '__main__':
    raise SystemExit(monitor(int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])))
