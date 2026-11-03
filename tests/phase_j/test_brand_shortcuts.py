"""旧版のリンク移行を隔離フォルダで検証する。実デスクトップは触らない。"""
from pathlib import Path
import os
import subprocess

import pytest
from src.phase_j import distribution_windows as windows


@pytest.mark.skipif(os.name != 'nt', reason='Windows専用ショートカット')
@pytest.mark.parametrize('prefix', ['PuyoLive-', 'puyoZeus-'])
def test_upgrade_migrates_only_same_installation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    root = tmp_path / 'installation'
    root.mkdir()
    desktop = tmp_path / 'desktop'
    desktop.mkdir()
    monkeypatch.setattr(windows, 'shortcut_directories', lambda: {'desktop_shortcut': desktop})
    name = windows.shortcut_name(root)
    old = desktop / name.replace('puyoZeus Live-', prefix, 1)
    other = desktop / windows.shortcut_name(tmp_path / 'other').replace('puyoZeus Live-', prefix, 1)
    old.write_text('旧リンク')
    other.write_text('別インストール')

    def create(command: list[str], **kwargs: object) -> None:
        env = kwargs['env']
        assert env['PUYO_TARGET'] == str(root / 'puyoZeus Live.bat')
        Path(env['PUYO_LINK']).write_text('新リンク')

    monkeypatch.setattr(windows.subprocess, 'run', create)
    windows.migrate_shortcuts(root, {'desktop_shortcut': True})
    assert (desktop / name).read_text() == '新リンク'
    assert not old.exists() and other.exists()
    windows.shortcuts(root, {'desktop_shortcut': True}, remove=True)
    assert not (desktop / name).exists() and other.exists()


@pytest.mark.skipif(os.name != 'nt', reason='Windows専用ショートカット')
def test_failed_migration_preserves_old_link(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(windows, 'shortcut_directories', lambda: {'desktop_shortcut': tmp_path})
    old = tmp_path / windows.shortcut_name(tmp_path).replace('puyoZeus Live-', 'PuyoLive-', 1)
    old.write_text('旧リンク')

    def fail(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, 'powershell')

    monkeypatch.setattr(windows.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        windows.migrate_shortcuts(tmp_path, {'desktop_shortcut': True})
    assert old.read_text() == '旧リンク'
