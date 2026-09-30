"""ランチャーの 機器別較正の保存先 (calibration_location) と VC++ ランタイム事前チェックの試験。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.phase_j import launcher
from src.phase_j.launcher import LauncherConfigError, parse_user_config, prepare
from src.phase_j.launcher_runtime import REQUIRED_RUNTIME_DLLS, missing_runtime_dlls, runtime_guidance

DSHOW = dict(device_name='OBS Virtual Camera')


@pytest.fixture
def isolated_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, 'environ', os.environ.copy())


def args(config: Path) -> object:
    return launcher.parse_launcher_args(['--config', str(config), '--skip-manifest'])


def write_config(tmp_path: Path, **extra: object) -> Path:
    path = tmp_path / 'c.json'
    path.write_text(json.dumps(dict(DSHOW, output_dir=str(tmp_path / 'o'), **extra)), encoding='utf-8')
    return path


def test_calibration_default_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """既定 (app) は環境変数を設定せず、live_device の既存パスのまま (後方互換)。"""
    from src.phase_j.live_device import CALIBRATION_ROOT, DeviceConfig
    monkeypatch.delenv(launcher.CALIBRATION_DIR_ENV, raising=False)
    user = parse_user_config(DSHOW, tmp_path)
    assert user.calibration_location == 'app' and launcher.calibration_dir(user) is None
    assert DeviceConfig('cam', 0).calibration_path.parent == CALIBRATION_ROOT


def test_calibration_env_name_matches_live_device() -> None:
    from src.phase_j import live_device
    assert live_device.CALIBRATION_DIR_ENV == launcher.CALIBRATION_DIR_ENV


def test_calibration_localappdata_moves_only_the_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_device import CALIBRATION_ROOT, DeviceConfig
    user = parse_user_config(dict(DSHOW, calibration_location='localappdata'), tmp_path)
    target = launcher.calibration_dir(user, {'LOCALAPPDATA': str(tmp_path / 'lad')})
    assert target == tmp_path / 'lad' / 'PuyoLive' / 'device_calibration'
    default_name = DeviceConfig('cam', 0).calibration_path.name
    monkeypatch.setenv(launcher.CALIBRATION_DIR_ENV, str(target))
    moved = DeviceConfig('cam', 0).calibration_path
    assert moved == target / default_name and moved.parent != CALIBRATION_ROOT  # 機器 ID (ファイル名) は不変


def test_calibration_localappdata_requires_env(tmp_path: Path) -> None:
    user = parse_user_config(dict(DSHOW, calibration_location='localappdata'), tmp_path)
    with pytest.raises(LauncherConfigError, match='LOCALAPPDATA'):
        launcher.calibration_dir(user, {})
    with pytest.raises(LauncherConfigError, match='calibration_location'):
        parse_user_config(dict(DSHOW, calibration_location='home'), tmp_path)


@pytest.mark.usefixtures('isolated_environ')
def test_prepare_exports_calibration_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'lad'))
    prepare(args(write_config(tmp_path, calibration_location='localappdata')))
    expected = tmp_path / 'lad' / 'PuyoLive' / 'device_calibration'
    assert os.environ[launcher.CALIBRATION_DIR_ENV] == str(expected) and expected.is_dir()


@pytest.mark.usefixtures('isolated_environ')
def test_prepare_default_leaves_environment_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(launcher.CALIBRATION_DIR_ENV, raising=False)
    prepare(args(write_config(tmp_path)))
    assert launcher.CALIBRATION_DIR_ENV not in os.environ


def test_runtime_check_reports_missing_dlls(tmp_path: Path) -> None:
    assert missing_runtime_dlls(tmp_path, lambda name: False, 'win32') == list(REQUIRED_RUNTIME_DLLS)
    (tmp_path / 'vcruntime140.dll').write_bytes(b'x')  # 同梱されていれば OS 側は見ない
    assert 'vcruntime140.dll' not in missing_runtime_dlls(tmp_path, lambda name: False, 'win32')
    assert missing_runtime_dlls(tmp_path, lambda name: name == 'msvcp140.dll', 'win32') == ['vcruntime140_1.dll']
    assert missing_runtime_dlls(tmp_path, lambda name: False, 'linux') == []  # 開発 (WSL) では対象外
    text = runtime_guidance(['msvcp140.dll'])
    assert 'msvcp140.dll' in text and 'aka.ms/vs/17/release/vc_redist.x64.exe' in text


def test_prepare_stops_with_guidance_when_runtime_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(launcher, 'missing_runtime_dlls', lambda python_dir: ['msvcp140.dll'])
    with pytest.raises(SystemExit) as info:
        prepare(args(write_config(tmp_path)))
    assert info.value.code == launcher.EXIT_RUNTIME and 'Visual C++' in capsys.readouterr().err
