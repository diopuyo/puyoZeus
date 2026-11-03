"""起動前の実行環境チェック (Windows の VC++ ランタイム DLL)。

torch / numpy / scipy / opencv は VC++ 2015-2022 ランタイムに依存する。未導入の PC では
`import torch` が「DLL load failed」で落ち、原因が利用者に分からない。配布物は再頒布可能な
DLL を python/ へ同梱するが、同梱漏れ・削除・別ビルドの取り違えに備え、起動前に必要な DLL が
(配布物内 または OS から) 解決できることを確かめ、不足を日本語で案内する。
必要な DLL の一覧は packaging/scan_dll_deps.py の PE 走査結果に基づく (docs 参照)。
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

# 走査 (packaging/scan_dll_deps.py) で「配布物内の同居では解決できず、システム/同梱に頼る」と出た VC ランタイム
# + python.exe / 各拡張が直接必要とするもの。
REQUIRED_RUNTIME_DLLS = ('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll')
VC_REDIST_URL = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
EXIT_RUNTIME = 4


def system_loadable(name: str) -> bool:
    """OS の通常の探索でロードできるか (Windows 以外は常に False)。"""
    if sys.platform != 'win32':
        return False
    import ctypes
    try:
        ctypes.WinDLL(name)
        return True
    except OSError:
        return False


def missing_runtime_dlls(python_dir: Path, loadable: Callable[[str], bool] = system_loadable,
                         platform: str | None = None) -> list[str]:
    """python_dir に同梱されていず、OS からもロードできない必須 DLL 名を返す。
    Windows 以外 (開発の WSL 等) では検査対象外で空。"""
    if (platform or sys.platform) != 'win32':
        return []
    return [name for name in REQUIRED_RUNTIME_DLLS
            if not (python_dir / name).is_file() and not loadable(name)]


def runtime_guidance(missing: list[str]) -> str:
    """不足時の案内文 (利用者向け、日本語)。"""
    return ('必要なシステム部品 (Microsoft Visual C++ 再頒布可能パッケージ) が見つかりません。\n'
            f'  不足: {", ".join(missing)}\n'
            f'  対処: 次の公式ページから x64 版を入れて再起動してください: {VC_REDIST_URL}\n'
            '  (配布物の python フォルダに上記 DLL が同梱されていれば通常は不要です。'
            'フォルダの一部が削除されていないか確認してください)')
