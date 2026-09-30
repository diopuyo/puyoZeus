"""Windows の性能コア (P コア) 優先。認識 process だけを高性能コアの論理 CPU 集合へ寄せる (既定 OFF)。

背景 (2026-09-30 実測、13th Gen i7-13620H = P 6 コア + E 4 コア): 認識 1 frame の P50 は
P コア固定 33.7 ms に対し E コア固定 75.6 ms (2.2 倍)。他プロセスが動く状況では OS が認識を
E コアや HT 兄弟へ寄せ、P50 が 63〜65 ms へ悪化した (同条件で P コア集合へ限定すると 49〜50 ms)。
認識の結果は計算しか変えないので、決定 (盤面・状態) には影響しない (packaging/perf の digest で確認)。

方式: `SetProcessDefaultCpuSets` (ソフトな集合。ハード affinity ではなく、利用者の設定を壊さない)。
効率クラスが 1 種類しか無い CPU (非ハイブリッド) では何もしない (戻り値 [] = 適用なし)。Windows 以外も何もしない。
"""
from __future__ import annotations

import os
import struct
import sys

PERFORMANCE_CORES_ENV = 'PUYO_PERFORMANCE_CORES'
# SYSTEM_CPU_SET_INFORMATION (winnt.h): Size(4) Type(4) | Id(4) Group(2) LogicalProcessorIndex(1) CoreIndex(1)
# LastLevelCacheIndex(1) NumaNodeIndex(1) EfficiencyClass(1) AllFlags(1) Reserved(4) AllocationTag(8)
CPU_SET_ENTRY_TYPE = 0
ENTRY_HEADER = struct.Struct('<II')
ENTRY_BODY = struct.Struct('<IHBBBBBB')
ENTRY_HEADER_SIZE = ENTRY_HEADER.size
INITIAL_BUFFER_BYTES = 4096
ERROR_INSUFFICIENT_BUFFER = 122


def parse_cpu_sets(buffer: bytes, length: int) -> list[tuple[int, int]]:
    """GetSystemCpuSetInformation の出力を [(CPU set の Id, 効率クラス)] へ。壊れた長さは例外にする。"""
    entries: list[tuple[int, int]] = []
    offset = 0
    while offset < length:
        size, kind = ENTRY_HEADER.unpack_from(buffer, offset)
        if size < ENTRY_HEADER_SIZE + ENTRY_BODY.size or offset + size > length:
            raise ValueError(f'CPU set 情報の長さが不正です (offset={offset}, size={size}, length={length})')
        if kind == CPU_SET_ENTRY_TYPE:
            cpu_id, _group, _logical, _core, _cache, _numa, efficiency, _flags = ENTRY_BODY.unpack_from(
                buffer, offset + ENTRY_HEADER_SIZE)
            entries.append((cpu_id, efficiency))
        offset += size
    return entries


def performance_cpu_set_ids(entries: list[tuple[int, int]]) -> list[int]:
    """最大の効率クラスの CPU set Id 一覧。効率クラスが 1 種類 (非ハイブリッド) なら空 (= 何もしない)。"""
    classes = {efficiency for _, efficiency in entries}
    if len(classes) < 2:
        return []
    top = max(classes)
    return sorted(cpu_id for cpu_id, efficiency in entries if efficiency == top)


def _kernel32():
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetSystemCpuSetInformation.argtypes = (ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG),
                                                  wintypes.HANDLE, wintypes.ULONG)
    kernel.GetSystemCpuSetInformation.restype = wintypes.BOOL
    kernel.SetProcessDefaultCpuSets.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.ULONG), wintypes.ULONG)
    kernel.SetProcessDefaultCpuSets.restype = wintypes.BOOL
    return kernel


def read_cpu_sets() -> list[tuple[int, int]]:
    import ctypes
    from ctypes import wintypes
    kernel = _kernel32()
    process = kernel.GetCurrentProcess()
    needed = wintypes.ULONG(0)
    kernel.GetSystemCpuSetInformation(None, 0, ctypes.byref(needed), process, 0)  # 必要長を得る (失敗が正常)
    buffer = ctypes.create_string_buffer(max(needed.value, INITIAL_BUFFER_BYTES))
    if not kernel.GetSystemCpuSetInformation(buffer, len(buffer), ctypes.byref(needed), process, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    return parse_cpu_sets(buffer.raw, needed.value)


def prefer_performance_cores() -> list[int]:
    """自 process の既定 CPU set を性能コアへ限定する。適用した CPU set Id を返す ([] = 適用なし)。"""
    if sys.platform != 'win32':
        return []
    import ctypes
    from ctypes import wintypes
    ids = performance_cpu_set_ids(read_cpu_sets())
    if not ids:
        return []
    kernel = _kernel32()
    array = (wintypes.ULONG * len(ids))(*ids)
    if not kernel.SetProcessDefaultCpuSets(kernel.GetCurrentProcess(), array, len(ids)):
        raise ctypes.WinError(ctypes.get_last_error())
    return ids


def requested() -> bool:
    return os.environ.get(PERFORMANCE_CORES_ENV) == '1'
