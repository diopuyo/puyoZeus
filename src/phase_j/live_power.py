"""窓なしの配信プロセス自身の実行速度の省電力を外す。既定 OFF。"""
from __future__ import annotations

import ctypes
import json
import multiprocessing as mp
import os
import sys
from typing import Any

ENV_POWER = 'PUYO_LIVE_DISABLE_POWER_THROTTLING'
POWER_KEY = 'live_disable_power_throttling'
PROCESS_POWER_THROTTLING = 4
EXECUTION_SPEED = 1
POWER_VERSION = 1
_attempted_pid: int | None = None


class PowerState(ctypes.Structure):
    _fields_ = [('Version', ctypes.c_uint32), ('ControlMask', ctypes.c_uint32),
                ('StateMask', ctypes.c_uint32)]


def validate_power(value: bool) -> bool:
    if type(value) is not bool:
        raise ValueError(f'{POWER_KEY}はboolです')
    return value


def power_enabled() -> bool:
    return os.environ.get(ENV_POWER, '0') == '1'


def configure_power(value: bool) -> None:
    os.environ[ENV_POWER] = str(int(validate_power(value)))
    apply_power_policy()


def power_pipeline_values(user: Any) -> dict[str, bool]:
    """既定OFFは従来JSONを維持し、外部ONの明示OFFは配送する。"""
    value = validate_power(user.live_disable_power_throttling)
    return {POWER_KEY: value} if value or power_enabled() else {}


def _disable_execution_throttling() -> None:
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                           ctypes.c_void_p, ctypes.c_uint32]
    kernel.SetProcessInformation.restype = ctypes.c_int
    state = PowerState(POWER_VERSION, EXECUTION_SPEED, 0)
    if not kernel.SetProcessInformation(kernel.GetCurrentProcess(),
                                       PROCESS_POWER_THROTTLING, ctypes.byref(state),
                                       ctypes.sizeof(state)):
        raise ctypes.WinError(ctypes.get_last_error())


def apply_power_policy() -> None:
    """環境変数を spawn が継承する。各 PID で一度だけ試し、失敗も継続する。"""
    global _attempted_pid
    pid = os.getpid()
    if not power_enabled() or _attempted_pid == pid:
        return
    _attempted_pid = pid
    status, error = 'unsupported', None
    if sys.platform == 'win32':
        try:
            _disable_execution_throttling()
            status = 'applied'
        except Exception as exc:
            status, error = 'failed', repr(exc)
    record = dict(event=POWER_KEY, pid=pid, process=mp.current_process().name,
                  status=status, error=error)
    try:
        print(json.dumps(record, ensure_ascii=True), file=sys.stderr, flush=True)
    except (OSError, AttributeError):
        pass  # 出力先がない窓なし起動でも、配信は継続する。
