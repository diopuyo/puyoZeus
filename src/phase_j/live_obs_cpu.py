"""本人用OBS試験だけのCPU分離。起動元が許されたCPU集合の外へは出ない。"""
from __future__ import annotations

import ctypes
import os

MIN_SPLIT_CPUS = 4
MAX_AMA_CPUS = 2
BELOW_NORMAL = 0x4000
IDLE = 0x40


def split_mask(mask: int) -> tuple[int, int]:
    cpus = [1 << index for index in range(mask.bit_length()) if mask & (1 << index)]
    if len(cpus) < MIN_SPLIT_CPUS:
        return mask, mask
    count = min(MAX_AMA_CPUS, max(1, len(cpus)//MIN_SPLIT_CPUS))
    return sum(cpus[:-count]), sum(cpus[-count:])


def kernel_api() -> object:
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    return kernel


def configure_main() -> None:
    if os.name != 'nt' or os.environ.get('PUYO_OBS_CPU_ISOLATION') != '1':
        return
    kernel = kernel_api()
    handle = kernel.GetCurrentProcess()
    if 'PUYO_OBS_MAIN_MASK' not in os.environ:
        allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(allowed), ctypes.byref(system)):
            raise ctypes.WinError()
        main, ama = split_mask(allowed.value)
        os.environ.update(PUYO_OBS_MAIN_MASK=str(main), PUYO_OBS_AMA_MASK=str(ama))
    if not kernel.SetProcessAffinityMask(handle, int(os.environ['PUYO_OBS_MAIN_MASK'])):
        raise ctypes.WinError()
    if not kernel.SetPriorityClass(handle, BELOW_NORMAL):
        raise ctypes.WinError()


def configure_ama(handle: int | None = None) -> None:
    isolated = os.environ.get('PUYO_OBS_CPU_ISOLATION') == '1'
    idle = os.environ.get('PUYO_OBS_AMA_IDLE') == '1'
    if os.name != 'nt' or not (isolated or idle):
        return
    kernel = kernel_api()
    priority = IDLE if handle is not None and idle else BELOW_NORMAL
    handle = handle or kernel.GetCurrentProcess()
    if isolated and 'PUYO_OBS_AMA_MASK' in os.environ:
        if not kernel.SetProcessAffinityMask(handle, int(os.environ['PUYO_OBS_AMA_MASK'])):
            raise ctypes.WinError()
    if not kernel.SetPriorityClass(handle, priority):
        raise ctypes.WinError()
