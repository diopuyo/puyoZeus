"""ライブprocessのCPU配分。既定は従来設定を維持する。"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from . import live_win_cores as win_cores

THREAD_ENV = 'PUYO_LIVE_CPU_THREADS'
NICE_ENV = 'PUYO_LIVE_EVALUATION_NICE'
MAX_NICE = 19
WINDOWS_ABOVE_NORMAL = 0x8000
POOL_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def configure_environment(threads: int, evaluation_nice: int) -> None:
    """spawnより前にライブラリ設定を渡す。0は既存の環境設定を変えない。"""
    if type(threads) is not int or threads < 0:
        raise ValueError('cpu_threadsは0以上の整数が必要です')
    if type(evaluation_nice) is not int or not 0 <= evaluation_nice <= MAX_NICE:
        raise ValueError('evaluation_niceは0〜19の整数が必要です')
    os.environ[THREAD_ENV], os.environ[NICE_ENV] = str(threads), str(evaluation_nice)
    if threads:
        for key in POOL_ENV:
            os.environ[key] = str(threads)


def apply_runtime(role: str, lower_priority: bool = True) -> dict[str, Any]:
    """認識をspawnしてから評価のniceを下げ、認識への継承を避ける。"""
    import cv2
    import torch
    threads = int(os.environ.get(THREAD_ENV, '0'))
    nice = int(os.environ.get(NICE_ENV, '0')) if role != 'recognition' else 0
    if threads:
        cv2.setNumThreads(threads)
        torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != threads:
            torch.set_num_interop_threads(threads)
    if lower_priority and os.environ.get('PUYO_CPU_ISOLATION') == '1':
        isolate_cpu(role)
    if lower_priority and nice:
        if not hasattr(os, 'nice'):
            windows_below_normal()
        else:
            current = os.nice(0)
            if current < nice:
                os.nice(nice-current)
    snapshot = runtime_snapshot(role)
    if role == 'recognition' and lower_priority and win_cores.requested():
        snapshot['performance_cpu_sets'] = apply_performance_cores()
    return snapshot


def apply_performance_cores() -> list[int]:
    """認識 process を性能コアへ寄せ、適用した件数を必ず stderr へ残す (0 件を成功と読み替えない)。"""
    ids = win_cores.prefer_performance_cores()
    print(f'[performance_cores] 性能コアの CPU set {len(ids)} 個へ限定'
          if ids else '[performance_cores] 適用なし (非ハイブリッド CPU または Windows 以外)',
          file=sys.stderr, flush=True)
    return ids


def isolate_cpu(role: str) -> None:
    """認識用1論理CPUを評価から除外する。外部アプリのaffinityは変更しない。"""
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        reserved = int(os.environ.get('PUYO_LIVE_RESERVED_CPU', str(allowed[0])))
        siblings = cpu_siblings(reserved)
        selected = [reserved] if role == 'recognition' else [cpu for cpu in allowed if cpu not in siblings]
        if selected and set(selected).issubset(allowed):
            set_process_affinity(selected)
    elif os.name == 'nt' and role == 'recognition':
        windows_priority(WINDOWS_ABOVE_NORMAL)


def set_process_affinity(cpus: list[int], tasks: Path = Path('/proc/self/task')) -> None:
    """既に生成されたlibrary/IPC threadにも適用する。新threadは親から継承する。"""
    os.sched_setaffinity(0, cpus)
    for task in tasks.iterdir():
        try:
            os.sched_setaffinity(int(task.name), cpus)
        except ProcessLookupError:
            continue


def cpu_siblings(cpu: int, root: Path = Path('/sys/devices/system/cpu')) -> set[int]:
    """評価が認識CPUのSMT兄弟へ入り、同じ物理coreを奪うことも防ぐ。"""
    try:
        value = (root/f'cpu{cpu}/topology/thread_siblings_list').read_text().strip()
        result = set()
        for item in value.split(','):
            bounds = [int(v) for v in item.split('-')]
            result.update(range(bounds[0], bounds[-1]+1))
        return result | {cpu}
    except (OSError, ValueError):
        return {cpu}


def windows_below_normal() -> None:
    """Windows実機では相当する低優先度クラスを使う（数値niceとの同値ではない）。"""
    below_normal = 0x4000
    windows_priority(below_normal)


def windows_priority(priority: int) -> None:
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), priority):
        raise ctypes.WinError(ctypes.get_last_error())


def runtime_snapshot(role: str) -> dict[str, Any]:
    import cv2
    import torch
    return dict(role=role, pid=os.getpid(), nice=os.nice(0) if hasattr(os, 'nice') else None,
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        opencv_threads=cv2.getNumThreads(), pools={key: os.environ.get(key) for key in POOL_ENV},
        loadavg=os.getloadavg() if hasattr(os, 'getloadavg') else None)
