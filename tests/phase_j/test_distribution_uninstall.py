"""アンインストールの保存・削除境界と動作中の拒否を検証する。"""
from pathlib import Path
import os

import pytest

from src.phase_j import distribution_uninstall as uninstall


@pytest.fixture(autouse=True)
def product_markers(tmp_path: Path) -> None:
    for name in uninstall.PRODUCT_MARKERS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}', encoding='utf-8')


@pytest.mark.parametrize('delete_data', [False, True])
def test_cleanup_preserves_only_selected_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             delete_data: bool) -> None:
    calls = []
    monkeypatch.setattr(uninstall, 'shortcuts', lambda *a, **k: calls.append(k))
    monkeypatch.setattr(uninstall, 'ensure_stopped', lambda *a: None)
    names = ['output/record.json', 'logs/runtime.log', 'distribution.json', 'puyo_live.json',
             'install.log', 'app/config/device_calibration/device.json']
    temporary = [*uninstall.TEMPORARY, 'app/config/device_calibration/device.tmp']
    retained = ['unrelated.txt', 'app/src/__pycache__/sample.pyc', 'python/Lib/__pycache__/sample.pyc']
    for name in names + temporary + retained:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('test', encoding='utf-8')
    uninstall.cleanup(tmp_path, delete_data)
    assert all((tmp_path / name).exists() != delete_data for name in names)
    assert all(not (tmp_path / name).exists() for name in temporary)
    assert all((tmp_path / name).exists() for name in retained)
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


def test_interactive_uninstall_refuses_running_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    """終了確認でnを選んだときは登録解除・削除へ進まない。"""
    from src.phase_j.distribution_app import acquire_instance
    (tmp_path / 'logs').mkdir()
    record = tmp_path / 'logs/record.json'
    record.write_text('keep', encoding='utf-8')
    lock = acquire_instance(tmp_path)
    monkeypatch.setattr('sys.argv', ['uninstall', '--root', str(tmp_path), '--interactive'])
    monkeypatch.setattr('builtins.input', lambda *a: 'n')
    monkeypatch.setattr(uninstall, 'shortcuts', lambda *a, **k: pytest.fail('登録を変更しない'))
    try:
        uninstall.main()
        assert record.read_text(encoding='utf-8') == 'keep'
        assert (tmp_path / 'logs/instance.lock').exists()
    finally:
        lock.close()


def test_direct_python_process_prevents_uninstall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ロックなしで同梱 Python が動いている場合も終了を促す。"""
    from types import SimpleNamespace
    executable = tmp_path / 'python/python.exe'
    executable.parent.mkdir()
    executable.write_bytes(b'')
    calls: list[dict] = []
    def running(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(stdout='12345\n')
    monkeypatch.setattr(uninstall.subprocess, 'run', running)
    with pytest.raises(OSError, match='Exit puyoZeus Live'):
        uninstall.ensure_stopped(tmp_path, wait_seconds=0)
    assert calls[0]['env']['PUYO_PYTHON'] == str(executable)
    assert calls[0]['env']['PUYO_CLEANUP_PID'] == str(os.getpid())
