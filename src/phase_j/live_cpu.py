"""ライブprocessのCPU配分。既定は従来設定を維持する。"""
from __future__ import annotations

import os
from typing import Any

THREAD_ENV = 'PUYO_LIVE_CPU_THREADS'
NICE_ENV = 'PUYO_LIVE_EVALUATION_NICE'
MAX_NICE = 19
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
    if lower_priority and nice:
        if not hasattr(os, 'nice'):
            windows_below_normal()
        else:
            current = os.nice(0)
            if current < nice:
                os.nice(nice-current)
    return runtime_snapshot(role)


def windows_below_normal() -> None:
    """Windows実機では相当する低優先度クラスを使う（数値niceとの同値ではない）。"""
    import ctypes
    below_normal = 0x4000
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), below_normal):
        raise ctypes.WinError(ctypes.get_last_error())


def runtime_snapshot(role: str) -> dict[str, Any]:
    import cv2
    import torch
    return dict(role=role, pid=os.getpid(), nice=os.nice(0) if hasattr(os, 'nice') else None,
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        opencv_threads=cv2.getNumThreads(), pools={key: os.environ.get(key) for key in POOL_ENV},
        loadavg=os.getloadavg() if hasattr(os, 'getloadavg') else None)
