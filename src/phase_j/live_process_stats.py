"""本人試験のWindowsプロセス木・CPU累積時間・RSS・GPU原票。"""
from __future__ import annotations
import ctypes as c
from ctypes import wintypes as w
import os
import shutil
import subprocess


class Entry(c.Structure):
    _fields_ = [('size', w.DWORD), ('usage', w.DWORD), ('pid', w.DWORD),
        ('heap', c.c_size_t), ('module', w.DWORD), ('threads', w.DWORD),
        ('parent', w.DWORD), ('priority', w.LONG), ('flags', w.DWORD), ('name', c.c_wchar*260)]


class Memory(c.Structure):
    _fields_ = [('size', w.DWORD), ('faults', w.DWORD),
        *[(name, c.c_size_t) for name in ('peak_rss', 'rss', 'peak_page', 'page',
          'peak_nonpage', 'nonpage', 'pagefile', 'peak_pagefile', 'private')]]


def process_rows() -> list[dict]:
    kernel = c.windll.kernel32
    kernel.CreateToolhelp32Snapshot.restype = w.HANDLE
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    kernel.Process32FirstW.argtypes = kernel.Process32NextW.argtypes = [w.HANDLE, c.POINTER(Entry)]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    entry, rows = Entry(), []
    entry.size = c.sizeof(entry)
    try:
        exists = kernel.Process32FirstW(handle, c.byref(entry))
        while exists:
            rows.append(dict(pid=entry.pid, parent=entry.parent, name=entry.name, threads=entry.threads))
            exists = kernel.Process32NextW(handle, c.byref(entry))
    finally:
        kernel.CloseHandle(handle)
    return rows


def details(row: dict) -> dict:
    kernel = c.windll.kernel32
    kernel.CloseHandle.argtypes = [w.HANDLE]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.GetProcessAffinityMask.argtypes = [w.HANDLE,c.POINTER(c.c_size_t),c.POINTER(c.c_size_t)]
    kernel.GetPriorityClass.argtypes = [w.HANDLE]
    handle = kernel.OpenProcess(0x410, False, row['pid'])
    if not handle:
        return dict(row, unavailable=True)
    kernel.GetProcessTimes.argtypes = [w.HANDLE, *[c.POINTER(c.c_ulonglong)]*4]
    c.windll.psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, c.POINTER(Memory), w.DWORD]
    times = [c.c_ulonglong() for _ in range(4)]
    memory = Memory()
    memory.size = c.sizeof(memory)
    try:
        affinity, system = c.c_size_t(), c.c_size_t()
        if kernel.GetProcessAffinityMask(handle,c.byref(affinity),c.byref(system)):
            row['affinity_mask'] = affinity.value
        row['priority_class'] = kernel.GetPriorityClass(handle)
        if kernel.GetProcessTimes(handle, *[c.byref(t) for t in times]):
            row['cpu_sec'] = (times[2].value+times[3].value)/1e7
        if c.windll.psapi.GetProcessMemoryInfo(handle, c.byref(memory), memory.size):
            row.update(rss_bytes=memory.rss, peak_rss_bytes=memory.peak_rss, private_bytes=memory.private)
    finally:
        kernel.CloseHandle(handle)
    return row


def process_tree(include_obs: bool = True) -> list[dict]:
    if os.name != 'nt':
        return []
    rows, owned = process_rows(), {os.getpid()}
    while True:
        found = owned | {r['pid'] for r in rows if r['parent'] in owned}
        if found == owned:
            break
        owned = found
    return [details(r) for r in rows if r['pid'] in owned or (include_obs and r['name'].lower() == 'obs64.exe')]


def gpu_memory() -> dict:
    command = shutil.which('nvidia-smi')
    if command is None:
        return dict(available=False, reason='nvidia-smi not installed')
    try:
        result = subprocess.run([command, '--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return dict(available=result.returncode == 0, memory_mib_utilization_percent=result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(available=False, reason=str(error))
