"""WSLの親強制終了とCtrl+Cでライブ子processを残さない。"""
from __future__ import annotations

import ctypes
import os
import signal
import sys
from typing import Any

PR_SET_PDEATHSIG = 1
STOP_TIMEOUT_SEC = 15.


def protect_parent(expected_pid: int | None = None) -> None:
    """spawnのimport中に親が消えた場合も、期待PIDとの比較で終了する。"""
    if expected_pid is None or not sys.platform.startswith('linux'):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'prctlに失敗しました')
    if os.getppid() != expected_pid:
        os._exit(1)


def join_evaluation(worker: Any) -> None:
    """Ctrl+Cは評価のfinallyへ渡し、応答しない場合だけ停止を強制する。"""
    try:
        worker.join()
    except KeyboardInterrupt:
        if worker.is_alive():
            os.kill(worker.pid, signal.SIGINT)
        worker.join(STOP_TIMEOUT_SEC)
        if worker.is_alive():
            worker.terminate()
            worker.join(STOP_TIMEOUT_SEC)
        raise
