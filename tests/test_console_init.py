"""src/console_init.py のテスト。"""
from __future__ import annotations

import os
import sys

from src.console_init import ascii_safe, init_console, to_windows_path


def test_init_console_idempotent() -> None:
    """何度呼んでも例外なく完了する。"""
    init_console()
    init_console()
    init_console()


def test_init_console_sets_pythonioencoding() -> None:
    """PYTHONIOENCODING が utf-8 にセットされる。"""
    init_console()
    assert os.environ.get("PYTHONIOENCODING") == "utf-8"


def test_init_console_sets_pythonutf8() -> None:
    """PYTHONUTF8 が 1 にセットされる (既存値があっても上書きしない)。"""
    init_console()
    assert os.environ.get("PYTHONUTF8") == "1"


def test_ascii_safe_passthrough_for_ascii() -> None:
    """ASCII のみ → そのまま返る。"""
    assert ascii_safe("hello world") == "hello world"


def test_ascii_safe_replaces_non_ascii() -> None:
    """非 ASCII → ? に置換。"""
    out = ascii_safe("hello あい")
    assert "?" in out
    # 置換後は ASCII のみ
    assert all(ord(c) < 128 for c in out)


def test_ascii_safe_handles_empty() -> None:
    """空文字 → 空文字。"""
    assert ascii_safe("") == ""


def test_to_windows_path_wsl() -> None:
    """/mnt/c/... → C:\..."""
    assert to_windows_path("/mnt/c/Users/ryouj/foo.png") == r"C:\Users\ryouj\foo.png"


def test_to_windows_path_mingw() -> None:
    """/c/... → C:\..."""
    assert to_windows_path("/c/Users/ryouj/foo.png") == r"C:\Users\ryouj\foo.png"


def test_to_windows_path_other_drive() -> None:
    """/mnt/d/... → D:\..."""
    assert to_windows_path("/mnt/d/data/x.txt") == r"D:\data\x.txt"


def test_to_windows_path_already_windows() -> None:
    """C:/foo/bar → C:\foo\bar."""
    assert to_windows_path("C:/foo/bar") == r"C:\foo\bar"


def test_to_windows_path_relative_unchanged() -> None:
    """相対パスは / を \ に正規化のみ。"""
    assert to_windows_path("data/foo/bar") == r"data\foo\bar"
