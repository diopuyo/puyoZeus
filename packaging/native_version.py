"""署名前のPEへWindows標準APIで版情報を付与する。"""
from __future__ import annotations

import ctypes
from pathlib import Path
import re
import struct

RT_VERSION = 16
RESOURCE_ID = 1
LANGUAGE = 0x0409
CODEPAGE = 1200
FIXED_SIGNATURE = 0xFEEF04BD
STRUCT_VERSION = 0x10000
FILE_FLAGS_MASK = 0x3F
WINDOWS_32 = 0x40004
FILE_DLL = 2
WORD_BITS = 16
WORD_MAX = 65535
ALIGNMENT = 4


def aligned(data: bytes) -> bytes:
    return data + bytes((-len(data)) % ALIGNMENT)


def block(key: str, value: bytes = b'', text: bool = True,
          children: tuple[bytes, ...] = ()) -> bytes:
    """VERSIONINFOの長さ・UTF-16・DWORD境界を構成する。"""
    length = len(value) // 2 if text else len(value)
    data = aligned(struct.pack('<HHH', 0, length, int(text)) + (key + '\0').encode('utf-16le'))
    data += value
    for child in children:
        data = aligned(data) + child
    return struct.pack('<H', len(data)) + data[2:]


def version_blob(version: str) -> bytes:
    """外側のJSONではなく署名対象のPEリソースに版を格納する。"""
    if not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('版はmajor.minor.patchで指定してください')
    parts = [int(part) for part in version.split('.')]
    if any(part > WORD_MAX for part in parts):
        raise ValueError('版の各数値は65535以下で指定してください')
    major_minor = (parts[0] << WORD_BITS) | parts[1]
    patch = parts[2] << WORD_BITS
    fixed = struct.pack('<13I', FIXED_SIGNATURE, STRUCT_VERSION, major_minor, patch,
                        major_minor, patch, FILE_FLAGS_MASK, 0, WINDOWS_32, FILE_DLL, 0, 0, 0)
    strings = tuple(block(key, (version + '\0').encode('utf-16le'))
                    for key in ('FileVersion', 'ProductVersion'))
    table = block(f'{LANGUAGE:04x}{CODEPAGE:04x}', children=strings)
    translation = block('Translation', struct.pack('<HH', LANGUAGE, CODEPAGE), text=False)
    return block('VS_VERSION_INFO', fixed, text=False, children=(
        block('StringFileInfo', children=(table,)), block('VarFileInfo', children=(translation,))))


def stamp_version(path: Path, version: str) -> None:
    """候補ファイルだけを更新する。署名を付ける前に実行する。"""
    blob = version_blob(version)
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.BeginUpdateResourceW.argtypes = [ctypes.c_wchar_p, ctypes.c_bool]
    api.BeginUpdateResourceW.restype = ctypes.c_void_p
    api.UpdateResourceW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_ushort, ctypes.c_void_p, ctypes.c_uint]
    api.UpdateResourceW.restype = ctypes.c_bool
    api.EndUpdateResourceW.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    api.EndUpdateResourceW.restype = ctypes.c_bool
    handle = api.BeginUpdateResourceW(str(path.resolve()), False)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(blob)
    if not api.UpdateResourceW(handle, RT_VERSION, RESOURCE_ID, LANGUAGE, buffer, len(blob)):
        error = ctypes.get_last_error()
        api.EndUpdateResourceW(handle, True)
        raise ctypes.WinError(error)
    if not api.EndUpdateResourceW(handle, False):
        raise ctypes.WinError(ctypes.get_last_error())
