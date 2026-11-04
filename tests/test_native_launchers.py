"""署名付き入口へのショートカット移行を確認する。"""
from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
def test_shortcut_migration_targets_exe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import distribution_windows as windows
    directories = {key: tmp_path / key for key in ('desktop_shortcut', 'start_menu_shortcut', 'autostart')}
    monkeypatch.setattr(windows, 'shortcut_directories', lambda: directories)
    calls = []

    def run(command: list[str], **kwargs: object) -> None:
        env = kwargs['env']
        calls.append(env)
        Path(env['PUYO_LINK']).write_text('mock link', encoding='utf-8')

    monkeypatch.setattr(windows.subprocess, 'run', run)
    monkeypatch.setattr(windows, 'remove_owned_link', lambda path, root: path.unlink(missing_ok=True))
    for directory in directories.values():
        directory.mkdir()
        (directory / windows.legacy_names(tmp_path)[0]).write_text('old VBS link', encoding='utf-8')
    windows.migrate_shortcuts(tmp_path, dict.fromkeys(directories, True))
    assert len(calls) == 3
    assert all(item['PUYO_TARGET'] == str(tmp_path / 'puyoZeus Live.exe') for item in calls)
    assert all(item['PUYO_ICON'] == str(tmp_path / 'app/src/phase_j/brand/app_logo.ico') for item in calls)
    assert [item['PUYO_ARGUMENTS'] for item in calls] == ['', '', '-c "pass" --autostart']


@pytest.mark.parametrize('old', ['puyoZeus Live.exe', 'puyoZeus Live.vbs', 'other'])
def test_existing_common_link_migration(old: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import distribution_windows as windows
    root, directory = tmp_path / 'app root', tmp_path / 'links'
    root.mkdir()
    directory.mkdir()
    link = directory / windows.shortcut_name(root)
    target = root / old if old != 'other' else tmp_path / 'another/app.exe'
    env = {**os.environ, 'PUYO_LINK': str(link), 'PUYO_OLD_TARGET': str(target)}
    create = ("$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:PUYO_LINK);"
              "$s.TargetPath=$env:PUYO_OLD_TARGET;$s.Arguments='--autostart';$s.Save()")
    subprocess.run([windows.system_tool('powershell'), '-NoProfile', '-NonInteractive', '-Command', create],
                   env=env, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    monkeypatch.setattr(windows, 'shortcut_directories', lambda: {'autostart': directory})
    windows.migrate_shortcuts(root, {'autostart': True})
    read = ("[Console]::OutputEncoding=[Text.UTF8Encoding]::new();"
            "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:PUYO_LINK);"
            "@{target=$s.TargetPath;icon=$s.IconLocation;args=$s.Arguments}|ConvertTo-Json -Compress")
    result = subprocess.run([windows.system_tool('powershell'), '-NoProfile', '-NonInteractive', '-Command', read],
                            env=env, check=True, capture_output=True, text=True, encoding='utf-8',
                            creationflags=subprocess.CREATE_NO_WINDOW)
    actual = json.loads(result.stdout)
    expected = target if old == 'other' else root / 'puyoZeus Live.exe'
    assert actual['target'].casefold() == str(expected).casefold()
    assert actual['args'] == ('--autostart' if old == 'other' else '-c "pass" --autostart')
    if old != 'other':
        assert actual['icon'].casefold() == (str(root / 'app/src/phase_j/brand/app_logo.ico') + ',0').casefold()
