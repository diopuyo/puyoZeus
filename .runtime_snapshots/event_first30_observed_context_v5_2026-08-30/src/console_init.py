"""コンソール出力の文字化け再発防止モジュール。

ぷよぷよ analyzer のスクリプトを Windows 上で wsl 経由で実行すると、
WSL (Linux UTF-8) → Windows 端末 (cp932) のパイプで日本語が
モジバケすることがある。

このモジュールは全 CLI スクリプトの先頭で `init_console()` を呼ぶ
だけで、以下の対策を一括適用する:

    1. sys.stdout / sys.stderr を UTF-8 + errors='replace' で再設定
    2. 環境変数 PYTHONIOENCODING / PYTHONUTF8 を子プロセス用に設定
    3. Windows 環境では code page を utf-8 (65001) に切替試行
    4. (補助) print の wrapper で非 ASCII 文字を ASCII safe 化する関数を提供

利用例:
    # スクリプト先頭で 1 行追加
    from src.console_init import init_console
    init_console()

スクリプト側で `print` を ASCII のみで書く運用と組み合わせるのが安全。
"""
from __future__ import annotations

import os
import sys
from typing import Any


def _safe_reconfigure_stream(stream: Any) -> None:
    """sys.stdout / sys.stderr を UTF-8 で再設定 (Python 3.7+)。"""
    if not hasattr(stream, "reconfigure"):
        return
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def init_console() -> None:
    """コンソールを UTF-8 で安定化する。スクリプト先頭で呼ぶ。

    冪等: 何度呼んでも問題ない。例外は内部で握りつぶす。
    """
    _safe_reconfigure_stream(sys.stdout)
    _safe_reconfigure_stream(sys.stderr)

    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ.setdefault("PYTHONUTF8", "1")

    # Windows 端末の code page を UTF-8 に (Python from cmd/PowerShell の場合のみ)
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except Exception:
            pass


def ascii_safe(text: str) -> str:
    """非 ASCII 文字を ? に置換した安全な文字列を返す (デバッグ用)。"""
    return text.encode("ascii", errors="replace").decode("ascii")


def to_windows_path(path: str) -> str:
    """WSL / Mingw / POSIX 形式のパスを Windows フルパス (C:\\...) に変換する。

    用途: Ctrl+click で開けるパスとしてユーザに提示する時。

    変換規則:
        /mnt/c/foo/bar → C:\\foo\\bar  (WSL)
        /c/foo/bar     → C:\\foo\\bar  (Git Bash / Mingw)
        /d/foo/bar     → D:\\foo\\bar
        C:/foo/bar     → C:\\foo\\bar  (既存 Windows 形式の正規化)
        その他は path.replace('/', '\\\\') で正規化のみ。
    """
    s = str(path).rstrip("/")
    # /mnt/c/... 形式 (WSL)
    if s.startswith("/mnt/") and len(s) >= 7 and s[6] == "/":
        drive = s[5].upper()
        rest = s[7:].replace("/", "\\")
        return f"{drive}:\\{rest}" if rest else f"{drive}:\\"
    # /c/... 形式 (Mingw / Git Bash)
    if (
        len(s) >= 3 and s.startswith("/")
        and s[1].isalpha() and s[2] == "/"
    ):
        drive = s[1].upper()
        rest = s[3:].replace("/", "\\")
        return f"{drive}:\\{rest}" if rest else f"{drive}:\\"
    # 既存 Windows 形式 (C:/... など)
    return s.replace("/", "\\")


__all__ = ["ascii_safe", "init_console", "to_windows_path"]
