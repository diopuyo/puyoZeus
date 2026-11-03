"""利用者が選んだショートカットだけを現在のユーザー範囲に作る。"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any


def shortcut_name(root: Path) -> str:
    return 'puyoZeus Live-' + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:10] + '.lnk'


def shortcut_directories() -> dict[str, Path]:
    """利用者の既知フォルダを使い、移動済みデスクトップにも追従する。"""
    import ctypes
    desktop = ctypes.create_unicode_buffer(260)
    ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, desktop)
    programs = Path(os.environ['APPDATA']) / 'Microsoft/Windows/Start Menu/Programs'
    return dict(desktop_shortcut=Path(desktop.value), start_menu_shortcut=programs,
                autostart=programs / 'Startup')


def migrate_shortcuts(root: Path, settings: dict[str, Any]) -> None:
    """同じインストール先の旧名リンクだけを新名へ移行する。"""
    if os.name == 'nt':
        legacy = [shortcut_name(root).replace('puyoZeus Live-', prefix, 1)
                  for prefix in ('PuyoLive-', 'puyoZeus-')]
        if any((directory / name).exists()
               for directory in shortcut_directories().values() for name in legacy):
            shortcuts(root, settings)


def shortcuts(root: Path, settings: dict[str, Any], remove: bool = False) -> None:
    """展開先ごとに名前を分け、別の配布版のリンクを上書きしない。"""
    if os.name != 'nt':
        return
    for key, directory in shortcut_directories().items():
        path = directory / shortcut_name(root)
        legacy = [directory / path.name.replace('puyoZeus Live-', prefix, 1)
                  for prefix in ('PuyoLive-', 'puyoZeus-')]
        if remove or not settings[key]:
            path.unlink(missing_ok=True)
            for old in legacy:
                old.unlink(missing_ok=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        # パスをコードに埋め込まず、環境変数として子へ渡す。
        script = "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:PUYO_LINK);$s.TargetPath=$env:PUYO_TARGET;$s.WorkingDirectory=$env:PUYO_ROOT;$s.IconLocation=$env:PUYO_ICON;$s.Save()"
        env = {**os.environ, 'PUYO_LINK': str(path), 'PUYO_TARGET': str(root / 'puyoZeus Live.vbs'), 'PUYO_ROOT': str(root),
               'PUYO_ICON': str(root / 'app/src/phase_j/brand/app_logo.ico')}
        subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
                       env=env, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        # 新しいリンクの作成に成功してから旧名を消す。
        for old in legacy:
            old.unlink(missing_ok=True)
