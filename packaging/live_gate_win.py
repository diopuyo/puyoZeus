"""測定専用のCPU台帳と、子孫にも効く一時的なCPU制限。本番からは参照しない。"""
from __future__ import annotations
import ctypes as ct
from ctypes import wintypes as wt
import struct

PERFORMANCE_CORE_COUNT = 4
NORMAL_PRIORITY = 0x20
JOB_EXTENDED_LIMITS = 9
JOB_AFFINITY_AND_KILL = 0x10 | 0x2000
CPU_SET_TYPE = 0


def kernel() -> ct.WinDLL:
    value = ct.WinDLL('kernel32', use_last_error=True)
    for name in ('GetCurrentProcess', 'CreateJobObjectW', 'OpenProcess'):
        getattr(value, name).restype = wt.HANDLE
    value.CreateJobObjectW.argtypes = (ct.c_void_p, wt.LPCWSTR)
    value.OpenJobObjectW.argtypes = (wt.DWORD, wt.BOOL, wt.LPCWSTR)
    value.OpenJobObjectW.restype = wt.HANDLE
    value.AssignProcessToJobObject.argtypes = (wt.HANDLE, wt.HANDLE)
    value.SetInformationJobObject.argtypes = (wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD)
    value.SetProcessAffinityMask.argtypes = (wt.HANDLE, ct.c_size_t)
    value.GetProcessAffinityMask.argtypes = (wt.HANDLE, ct.c_void_p, ct.c_void_p)
    value.SetPriorityClass.argtypes = (wt.HANDLE, wt.DWORD)
    value.GetPriorityClass.argtypes = (wt.HANDLE,)
    value.CloseHandle.argtypes = (wt.HANDLE,)
    return value


def singleton() -> int:
    """準備・復元も含めて二重起動を拒否する。終了時はOSが解放する。"""
    api = kernel()
    api.CreateMutexW.argtypes = (ct.c_void_p, wt.BOOL, wt.LPCWSTR)
    api.CreateMutexW.restype = wt.HANDLE
    handle = api.CreateMutexW(None, True, 'Local\\PuyoLiveGateCoef20261022')
    if not handle or ct.get_last_error() == 183:
        raise RuntimeError('係数測定の入口が既に起動している')
    return handle


def process_birth(pid: int) -> int | None:
    api = kernel()
    api.OpenProcess.argtypes = (wt.DWORD, wt.BOOL, wt.DWORD)
    api.GetProcessTimes.argtypes = (wt.HANDLE, *([ct.POINTER(ct.c_ulonglong)]*4))
    handle = api.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    times = [ct.c_ulonglong() for _ in range(4)]
    try:
        return times[0].value if api.GetProcessTimes(handle, *[ct.byref(t) for t in times]) else None
    finally:
        api.CloseHandle(handle)


def topology() -> list[dict]:
    api, needed = kernel(), wt.DWORD()
    api.GetSystemCpuSetInformation.argtypes = (ct.c_void_p, wt.ULONG,
        ct.POINTER(wt.ULONG), wt.HANDLE, wt.ULONG)
    api.GetSystemCpuSetInformation(None, 0, ct.byref(needed), None, 0)
    buffer = ct.create_string_buffer(needed.value)
    if not api.GetSystemCpuSetInformation(buffer, len(buffer), ct.byref(needed), None, 0):
        raise ct.WinError(ct.get_last_error())
    rows, offset = [], 0
    while offset < needed.value:
        size, kind = struct.unpack_from('<II', buffer, offset)
        if size < 20 or offset + size > needed.value:
            raise ValueError('CPU topologyの長さが不正')
        if kind == CPU_SET_TYPE:
            ident, group, logical, core, cache, numa, efficiency, flags = struct.unpack_from(
                '<IHBBBBBB', buffer, offset + 8)
            rows.append(dict(id=ident, group=group, logical=logical, core=core,
                efficiency=efficiency, flags=flags))
        offset += size
    return rows


def masks(rows: list[dict]) -> tuple[int, int]:
    if not rows or {r['group'] for r in rows} != {0}:
        raise ValueError('単一processor groupのPCだけを対象とする')
    top = max(r['efficiency'] for r in rows)
    cores = sorted({r['core'] for r in rows if r['efficiency'] == top})
    if len(cores) < PERFORMANCE_CORE_COUNT:
        raise ValueError('性能側の物理コア4個を確認できない')
    selected = [r for r in rows if r['efficiency'] == top and
                r['core'] in cores[:PERFORMANCE_CORE_COUNT]]
    return sum(1 << r['logical'] for r in rows), sum(1 << r['logical'] for r in selected)


class BasicLimits(ct.Structure):
    _fields_ = [('process_time', ct.c_longlong), ('job_time', ct.c_longlong),
        ('flags', wt.DWORD), ('min_ws', ct.c_size_t), ('max_ws', ct.c_size_t),
        ('active', wt.DWORD), ('affinity', ct.c_size_t), ('priority', wt.DWORD),
        ('scheduling', wt.DWORD)]


class ExtendedLimits(ct.Structure):
    _fields_ = [('basic', BasicLimits), ('io', ct.c_ulonglong * 6),
        ('process_memory', ct.c_size_t), ('job_memory', ct.c_size_t),
        ('peak_process', ct.c_size_t), ('peak_job', ct.c_size_t)]


def create_job(name: str, mask: int) -> int:
    """controllerだけがハンドルを保持し、controller消失時に測定木を終了する。"""
    api, limits = kernel(), ExtendedLimits()
    limits.basic.flags = JOB_AFFINITY_AND_KILL
    limits.basic.affinity = mask
    handle = api.CreateJobObjectW(None, name)
    if not handle or not api.SetInformationJobObject(handle, JOB_EXTENDED_LIMITS,
            ct.byref(limits), ct.sizeof(limits)):
        raise ct.WinError(ct.get_last_error())
    return handle


def attach_job(mask: int, name: str) -> dict:
    """先に自身を制限し、その後起動する全子孫にhard affinityを継承させる。"""
    api = kernel()
    handle = api.OpenJobObjectW(1, False, name)
    process = api.GetCurrentProcess()
    if not handle:
        raise ct.WinError(ct.get_last_error())
    if not api.AssignProcessToJobObject(handle, process):
        raise ct.WinError(ct.get_last_error())
    api.CloseHandle(handle)
    if not api.SetProcessAffinityMask(process, mask) or not api.SetPriorityClass(process, NORMAL_PRIORITY):
        raise ct.WinError(ct.get_last_error())
    actual, system = ct.c_size_t(), ct.c_size_t()
    if not api.GetProcessAffinityMask(process, ct.byref(actual), ct.byref(system)):
        raise ct.WinError(ct.get_last_error())
    if actual.value != mask or api.GetPriorityClass(process) != NORMAL_PRIORITY:
        raise RuntimeError('要求したaffinity/Normal優先度と不一致')
    return dict(mask=actual.value, priority='Normal', job_kill_on_close=True)
