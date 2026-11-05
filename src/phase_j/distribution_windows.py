"""利用者が選んだショートカットだけを現在のユーザー範囲に作る。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def system_tool(name: str) -> str:
    paths = {'powershell': 'WindowsPowerShell/v1.0/powershell.exe',
             'taskkill': 'taskkill.exe', 'clip': 'clip.exe'}
    return str(Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / paths[name])


def link_stamp(path: Path, env: dict[str, str]) -> dict[str, Any]:
    return dict(size=path.stat().st_size, modified=path.stat().st_mtime_ns,
                target={key: value for key, value in env.items() if key.startswith('PUYO_')}) if path.is_file() else {}


def shortcut_name(root: Path) -> str:
    return 'puyoZeus Live.lnk'


def legacy_names(root: Path) -> list[str]:
    stamp = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:10]
    return [prefix + stamp + '.lnk' for prefix in ('puyoZeus Live-', 'PuyoLive-', 'puyoZeus-')]


def remove_owned_link(path: Path, root: Path) -> None:
    """リンクの実際の参照先を確認し、他の展開先のリンクは残す。"""
    if not path.is_file():
        return
    script = ("$ErrorActionPreference='Stop';"
              "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:PUYO_LINK);"
              "$target=[IO.Path]::GetFullPath($s.TargetPath);"
              "$root=[IO.Path]::GetFullPath($env:PUYO_ROOT).TrimEnd('\\')+'\\';"
              "if($target.StartsWith($root,[StringComparison]::OrdinalIgnoreCase)) {"
              "Remove-Item -LiteralPath $env:PUYO_LINK}")
    subprocess.run([system_tool('powershell'), '-NoProfile', '-NonInteractive', '-Command', script],
                   env={**os.environ, 'PUYO_LINK': str(path), 'PUYO_ROOT': str(root)},
                   check=True, creationflags=subprocess.CREATE_NO_WINDOW)


def shortcut_directories() -> dict[str, Path]:
    """利用者の既知フォルダを使い、移動済みデスクトップにも追従する。"""
    import ctypes
    desktop = ctypes.create_unicode_buffer(260)
    ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, desktop)
    programs = Path(os.environ['APPDATA']) / 'Microsoft/Windows/Start Menu/Programs'
    return dict(desktop_shortcut=Path(desktop.value), start_menu_shortcut=programs,
                autostart=programs / 'Startup')


def migrate_shortcuts(root: Path, settings: dict[str, Any]) -> None:
    """通常リンクは自分のものだけ削除し、自動起動の旧名は移行する。"""
    if os.name == 'nt':
        legacy = legacy_names(root)
        directories = shortcut_directories()
        for key, directory in directories.items():
            if key != 'autostart':
                for name in [shortcut_name(root), *legacy]:
                    remove_owned_link(directory / name, root)
        directories = {key: value for key, value in directories.items() if key == 'autostart'}
        refresh_existing_shortcuts(root, settings, directories)
        missing_startup = (settings.get('autostart') and 'autostart' in directories
                           and not (directories['autostart'] / shortcut_name(root)).exists())
        if missing_startup or any((directory / name).exists()
               for directory in directories.values() for name in legacy):
            shortcuts(root, settings)


def shortcut_environment(root: Path, key: str, path: Path) -> dict[str, str]:
    return {**os.environ, 'PUYO_LINK': str(path), 'PUYO_TARGET': str(root / 'puyoZeus Live.exe'),
            'PUYO_ROOT': str(root), 'PUYO_ICON': str(root / 'app/src/phase_j/brand/app_logo.ico'),
            'PUYO_ARGUMENTS': '-c "pass" --autostart' if key == 'autostart' else ''}


def refresh_existing_shortcuts(root: Path, settings: dict[str, Any], directories: dict[str, Path]) -> None:
    """共通名の旧候補も移行する。他の展開先を指すリンクは変更しない。"""
    script = ("$ErrorActionPreference='Stop';"
              "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:PUYO_LINK);"
              "$target=[IO.Path]::GetFullPath($s.TargetPath);"
              "$root=[IO.Path]::GetFullPath($env:PUYO_ROOT).TrimEnd('\\')+'\\';"
              "if($target.StartsWith($root,[StringComparison]::OrdinalIgnoreCase) -and "
              "($s.TargetPath -ne $env:PUYO_TARGET -or $s.Arguments -ne $env:PUYO_ARGUMENTS -or "
              "$s.IconLocation -ne ($env:PUYO_ICON+',0'))) {"
              "$s.TargetPath=$env:PUYO_TARGET;$s.Arguments=$env:PUYO_ARGUMENTS;"
              "$s.IconLocation=$env:PUYO_ICON;$s.WorkingDirectory=$env:PUYO_ROOT;$s.Save()}")
    for key, directory in directories.items():
        path = directory / shortcut_name(root)
        if key == 'autostart' and settings.get(key) and path.is_file():
            subprocess.run([system_tool('powershell'), '-NoProfile', '-NonInteractive', '-Command', script],
                           env=shortcut_environment(root, key, path), check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)


def shortcuts(root: Path, settings: dict[str, Any], remove: bool = False) -> None:
    """共通名は最後に選んだ展開先が持ち、削除は自分のリンクに限る。"""
    if os.name != 'nt':
        return
    cache_path = root / 'logs/shortcuts.json'
    try:
        cache = json.loads(cache_path.read_text(encoding='utf-8'))
        if not isinstance(cache, dict):
            cache = {}
    except (OSError, ValueError):
        cache = {}
    for key, directory in shortcut_directories().items():
        path = directory / shortcut_name(root)
        legacy = [directory / name for name in legacy_names(root)]
        if remove or key != 'autostart' or not settings.get(key, False):
            for old in [path, *legacy]:
                remove_owned_link(old, root)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        # パスをコードに埋め込まず、環境変数として子へ渡す。
        script = ("$ErrorActionPreference='Stop';$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:PUYO_LINK);"
                  "if($s.TargetPath -ne $env:PUYO_TARGET -or $s.Arguments -ne $env:PUYO_ARGUMENTS -or "
                  "$s.WorkingDirectory -ne $env:PUYO_ROOT -or $s.IconLocation -ne ($env:PUYO_ICON+',0')) {"
                  "$s.TargetPath=$env:PUYO_TARGET;$s.Arguments=$env:PUYO_ARGUMENTS;"
                  "$s.WorkingDirectory=$env:PUYO_ROOT;$s.IconLocation=$env:PUYO_ICON;$s.Save()}")
        env = shortcut_environment(root, key, path)
        if not path.exists() or cache.get(key) != link_stamp(path, env):
            subprocess.run([system_tool('powershell'), '-NoProfile', '-NonInteractive', '-Command', script],
                           env=env, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            cache[key] = link_stamp(path, env)
        # 新しいリンクの作成に成功してから旧名を消す。
        for old in legacy:
            remove_owned_link(old, root)
    if not remove:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(cache), encoding='utf-8')
        temporary.replace(cache_path)
