"""CPU集合の種類を読取り専用で保存し、異なるCPU配置の速度比較を避ける。"""
from __future__ import annotations

import ctypes
from datetime import datetime
import json
from pathlib import Path
import struct
import sys

CPU_SET_TYPE = 0
HEADER = struct.Struct('<II')
CPU_SET = struct.Struct('<IIIHBBBBBBIQ')
FIELDS = ('size', 'type', 'id', 'group', 'logical_processor', 'core',
          'last_level_cache', 'numa_node', 'efficiency_class', 'flags',
          'scheduling_reserved', 'allocation_tag')
REFERENCE = 'https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information'


def capture() -> dict:
    """affinityや優先度を変更せず、可変長のCPU Setを公式定義で読む。"""
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    query = kernel.GetSystemCpuSetInformation
    query.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                      ctypes.c_void_p, ctypes.c_ulong]
    query.restype = ctypes.c_bool
    length = ctypes.c_ulong()
    query(None, 0, ctypes.byref(length), None, 0)
    if not length.value:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(length.value)
    if not query(buffer, length.value, ctypes.byref(length), None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    rows, offset = [], 0
    while offset < length.value:
        size, kind = HEADER.unpack_from(buffer, offset)
        if size < HEADER.size or offset+size > length.value:
            raise ValueError('CPU Setの長さが不正')
        if kind == CPU_SET_TYPE:
            if size < CPU_SET.size:
                raise ValueError('CPU Setの既知フィールドが不足')
            rows.append(dict(zip(FIELDS, CPU_SET.unpack_from(buffer, offset))))
        offset += size
    return dict(at=datetime.now().astimezone().isoformat(), reference=REFERENCE, cpu_sets=rows)


if __name__ == '__main__':
    value = capture()
    Path(sys.argv[1]).write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(value, indent=2))
