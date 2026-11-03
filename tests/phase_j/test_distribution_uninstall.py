"""アンインストールの保存・削除境界と動作中の拒否を検証する。"""
from pathlib import Path
import os

import pytest

from src.phase_j import distribution_uninstall as uninstall


@pytest.mark.parametrize('delete_data', [False, True])
def test_cleanup_preserves_only_selected_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             delete_data: bool) -> None:
    calls = []
    monkeypatch.setattr(uninstall, 'shortcuts', lambda *a, **k: calls.append(k))
    monkeypatch.setattr(uninstall, 'ensure_stopped', lambda *a: None)
    names = ['output/record.json', 'logs/runtime.log', 'distribution.json', 'puyo_live.json',
             'install.log', 'app/config/device_calibration/device.json']
    temporary = [*uninstall.TEMPORARY, 'app/src/__pycache__/sample.pyc',
                 'app/config/device_calibration/device.tmp']
    for name in names + temporary + ['unrelated.txt']:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('test', encoding='utf-8')
    uninstall.cleanup(tmp_path, delete_data)
    assert all((tmp_path / name).exists() != delete_data for name in names)
    assert all(not (tmp_path / name).exists() for name in temporary)
    assert (tmp_path / 'unrelated.txt').exists()
    assert calls == [dict(remove=True)]


@pytest.mark.skipif(os.name != 'nt', reason='Windowsの実ロック')
def test_running_instance_is_untouched(tmp_path: Path) -> None:
    from src.phase_j.distribution_app import acquire_instance
    (tmp_path / 'logs').mkdir()
    lock = acquire_instance(tmp_path)
    try:
        with pytest.raises(OSError, match='Exit puyoZeus'):
            uninstall.ensure_stopped(tmp_path, wait_seconds=0)
        assert (tmp_path / 'logs/instance.lock').exists()
    finally:
        lock.close()
    uninstall.ensure_stopped(tmp_path, wait_seconds=0)


def test_cleanup_rejects_outside_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        uninstall.checked_path(tmp_path, '../outside')


def test_cleanup_failure_does_not_remove_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def running(*args: object) -> None:
        raise OSError('running')
    monkeypatch.setattr(uninstall, 'ensure_stopped', running)
    monkeypatch.setattr(uninstall, 'shortcuts', lambda *a, **k: pytest.fail('登録を変更してはいけない'))
    with pytest.raises(OSError):
        uninstall.cleanup(tmp_path, True)
