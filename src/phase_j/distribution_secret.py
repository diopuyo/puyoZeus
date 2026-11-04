"""OBSの資格情報を現在のWindowsユーザーのDPAPIで保護する。"""
from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
import os

CRYPTPROTECT_UI_FORBIDDEN = 1


class DataBlob(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]


def crypt(data: bytes, decrypt: bool = False) -> bytes:
    if os.name != 'nt':
        raise OSError('OBSパスワードの保存にはWindows DPAPIが必要です')
    crypt32 = ctypes.WinDLL('crypt32', use_last_error=True)
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    buffer = ctypes.create_string_buffer(data)
    source = DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result = DataBlob()
    function = crypt32.CryptUnprotectData if decrypt else crypt32.CryptProtectData
    function.argtypes = [ctypes.POINTER(DataBlob), ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(DataBlob)]
    function.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    if not function(ctypes.byref(source), None, None, None, None,
                    CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(result)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        kernel32.LocalFree(result.data)


def protect(password: str) -> dict[str, str] | str:
    return {'protection': 'dpapi', 'value': base64.b64encode(crypt(password.encode('utf-8'))).decode('ascii')} if password else ''


def unprotect(value: object) -> str:
    if isinstance(value, str):
        return value
    if (not isinstance(value, dict) or value.get('protection') != 'dpapi'
            or not isinstance(value.get('value'), str)):
        raise ValueError('Invalid OBS password storage')
    return crypt(base64.b64decode(value['value'], validate=True), decrypt=True).decode('utf-8')
