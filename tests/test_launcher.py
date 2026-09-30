"""配布ランチャー (src/phase_j/launcher.py) の設定検証・CLI 写像・終了コードの試験。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.run_live_pipeline_20260928 import parse_args
from src.phase_j import launcher
from src.phase_j.launcher import (EXIT_CONFIG, EXIT_MANIFEST, LauncherConfigError, PIPELINE_CONFIG_NAME,
                                  build_pipeline_argv, build_pipeline_config, load_user_config,
                                  parse_user_config, prepare)
from src.phase_j.launcher_manifest import build_manifest

ROOT = Path(__file__).resolve().parents[1]
DSHOW = dict(device_name='OBS Virtual Camera')


@pytest.fixture
def isolated_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    """既存パイプラインの parse_args は os.environ を書き換えるため、試験ごとに復元する。"""
    monkeypatch.setattr(os, 'environ', os.environ.copy())


def write_json(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def test_defaults_for_dshow(tmp_path: Path) -> None:
    user = parse_user_config(DSHOW, tmp_path)
    assert (user.source, user.device_index, user.port, user.host) == ('dshow', 0, 8765, '127.0.0.1')
    assert user.output_dir == (tmp_path / 'output').resolve() and user.mc_rollouts == 30


@pytest.mark.parametrize('data,needle', [
    ([], 'オブジェクト'),
    (dict(), 'device_name'),
    (dict(device_name='  '), 'device_name'),
    (dict(DSHOW, typo=1), '未知の設定項目'),
    (dict(DSHOW, port=80), 'port'),
    (dict(DSHOW, port=70000), 'port'),
    (dict(DSHOW, port='8765'), 'port'),
    (dict(DSHOW, port=True), 'port'),
    (dict(DSHOW, device_index=-1), 'device_index'),
    (dict(DSHOW, mc_rollouts=0), 'mc_rollouts'),
    (dict(DSHOW, source='usb'), 'source'),
    (dict(source='video'), 'video_path'),
    (dict(DSHOW, host=''), 'host'),
    (dict(source='video', video_path='a.mp4', start_sec=-1), 'start_sec'),
])
def test_invalid_config_reports_the_key(tmp_path: Path, data: object, needle: str) -> None:
    with pytest.raises(LauncherConfigError, match=needle):
        parse_user_config(data, tmp_path, allow_video=True)


def test_load_errors_are_actionable(tmp_path: Path) -> None:
    with pytest.raises(LauncherConfigError, match='puyo_live.example.json'):
        load_user_config(tmp_path / 'missing.json')
    broken = tmp_path / 'broken.json'
    broken.write_text('{"device_name": ', encoding='utf-8')
    with pytest.raises(LauncherConfigError, match='JSON が不正'):
        load_user_config(broken)


def test_relative_paths_follow_config_file_location(tmp_path: Path) -> None:
    path = write_json(tmp_path / 'puyo_live.json', dict(source='video', video_path='clips/a.mp4', output_dir='out'))
    user = load_user_config(path, allow_video=True)
    assert user.video_path == (tmp_path / 'clips/a.mp4').resolve() and user.output_dir == (tmp_path / 'out').resolve()


def test_dshow_pipeline_config_matches_shipped_example(tmp_path: Path) -> None:
    """配布既定 (CPU 専用・低優先度) が既存の live_dshow.example.json と食い違わない。"""
    example = json.loads((ROOT / 'config/live_dshow.example.json').read_text(encoding='utf-8'))
    config = build_pipeline_config(parse_user_config(DSHOW, tmp_path))
    assert config == example


def test_video_pipeline_config_keeps_window(tmp_path: Path) -> None:
    user = parse_user_config(dict(source='video', video_path='a.mp4', start_sec=10, end_sec=20.5), tmp_path, allow_video=True)
    config = build_pipeline_config(user)
    assert (config['realtime'], config['start_sec'], config['end_sec'], config['name']) == (True, 10.0, 20.5, 'a')


def test_argv_maps_to_existing_cli(tmp_path: Path) -> None:
    user = parse_user_config(dict(source='video', video_path='a.mp4'), tmp_path, allow_video=True)
    argv = build_pipeline_argv(user, tmp_path / 'p.json')
    assert argv == ['--config', str(tmp_path / 'p.json'), '--output', str(user.output_dir),
                    '--video', str(user.video_path)]
    assert build_pipeline_argv(parse_user_config(DSHOW, tmp_path), tmp_path / 'p.json')[-2:] == [
        '--output', str((tmp_path / 'output').resolve())]


def parse_with_pipeline(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> object:
    monkeypatch.chdir(ROOT)  # 既存 apply_config は config/live_defaults.json を相対パスで読む
    monkeypatch.setattr('sys.argv', ['live', *argv])
    return parse_args()


@pytest.mark.usefixtures('isolated_environ')
def test_dshow_argv_is_accepted_by_real_pipeline_parser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """写像結果を既存 parse_args へ実際に通し、意味 (機器入力・realtime・CPU) まで確認する。"""
    user = parse_user_config(dict(DSHOW, port=9001, mc_rollouts=7, output_dir=str(tmp_path / 'o')), tmp_path)
    config_path = write_json(tmp_path / PIPELINE_CONFIG_NAME, build_pipeline_config(user))
    options = parse_with_pipeline(monkeypatch, build_pipeline_argv(user, config_path))
    assert (options.source, options.port, options.mc_rollouts, options.cnn_device) == ('dshow', 9001, 7, 'cpu')
    assert options.realtime and options.lifecycle and options.input_config == config_path
    assert options.output == user.output_dir and options.host == '127.0.0.1'


@pytest.mark.usefixtures('isolated_environ')
def test_video_argv_is_accepted_by_real_pipeline_parser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user = parse_user_config(dict(source='video', video_path='clip.mp4', start_sec=5, end_sec=15), tmp_path, allow_video=True)
    config_path = write_json(tmp_path / PIPELINE_CONFIG_NAME, build_pipeline_config(user))
    options = parse_with_pipeline(monkeypatch, build_pipeline_argv(user, config_path))
    assert options.source == 'video' and options.video == user.video_path and options.realtime
    assert (options.start_sec, options.end_sec) == (5.0, 15.0) and options.input_config is None


class FakeCapture:
    """index ごとに 開けない / 開けるが無映像 / 映像あり を模す。"""

    def __init__(self, index: int) -> None:
        self.index = index

    def isOpened(self) -> bool:  # noqa: N802 (OpenCV の名前)
        return self.index != 0

    def read(self) -> tuple[bool, object]:
        import numpy as np
        return (True, np.zeros((1080, 1920, 3), np.uint8)) if self.index == 2 else (False, None)

    def release(self) -> None:
        pass


def test_probe_devices_reports_open_frame_and_size() -> None:
    rows = launcher.probe_devices(2, FakeCapture)
    assert [(r['index'], r['opened'], r['frame'], r['size']) for r in rows] == [
        (0, False, False, None), (1, True, False, None), (2, True, True, (1920, 1080))]
    text = launcher.format_devices(rows)
    assert '開けません' in text and '映像を読めません' in text and '1920x1080' in text


def test_list_devices_needs_no_config(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    original = launcher.probe_devices  # 差し替え後に自分自身を呼ばないよう先に退避
    monkeypatch.setattr(launcher, 'probe_devices', lambda: original(1, FakeCapture))
    assert launcher.main(['--list-devices', '--config', 'no_such.json']) == 0
    assert 'index 0' in capsys.readouterr().out


def launcher_args(config: Path, *extra: str) -> launcher.argparse.Namespace:
    return launcher.parse_launcher_args(['--config', str(config), *extra])


def test_prepare_exit_code_for_bad_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_json(tmp_path / 'c.json', dict(port=80))
    with pytest.raises(SystemExit) as info:
        prepare(launcher_args(path, '--skip-manifest'))
    assert info.value.code == EXIT_CONFIG and 'device_name' in capsys.readouterr().err


def test_prepare_writes_pipeline_config(tmp_path: Path) -> None:
    path = write_json(tmp_path / 'c.json', dict(DSHOW, output_dir=str(tmp_path / 'o')))
    user, argv = prepare(launcher_args(path, '--skip-manifest'))
    written = json.loads((user.output_dir / PIPELINE_CONFIG_NAME).read_text(encoding='utf-8'))
    assert written['name'] == 'OBS Virtual Camera' and argv[0] == '--config'


def test_require_manifest_fails_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(launcher, 'APP_ROOT', tmp_path / 'app')
    (tmp_path / 'app').mkdir()
    path = write_json(tmp_path / 'c.json', dict(DSHOW, output_dir=str(tmp_path / 'o')))
    with pytest.raises(SystemExit) as info:
        prepare(launcher_args(path, '--require-manifest'))
    assert info.value.code == EXIT_MANIFEST and 'MANIFEST.json がありません' in capsys.readouterr().err


def test_tampered_bundle_stops_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    app = tmp_path / 'app'
    (app / 'models').mkdir(parents=True)
    model = app / 'models' / 'm.pt'
    model.write_bytes(b'weights')
    manifest = build_manifest(app, [model])
    write_json(app / 'MANIFEST.json', manifest)
    monkeypatch.setattr(launcher, 'APP_ROOT', app)
    path = write_json(tmp_path / 'c.json', dict(DSHOW, output_dir=str(tmp_path / 'o')))
    prepare(launcher_args(path))  # 無改変は通る
    model.write_bytes(b'weightz')  # 同サイズで内容だけ改変
    with pytest.raises(SystemExit) as info:
        prepare(launcher_args(path))
    err = capsys.readouterr().err
    assert info.value.code == EXIT_MANIFEST and 'SHA256 不一致' in err and 'models/m.pt' in err


def test_video_source_is_rejected_in_distribution(tmp_path: Path) -> None:
    """配布版 (既定) は source=video を受け付けず、日本語で案内する。開発用の許可は明示引数のみ。"""
    data = dict(source='video', video_path='a.mp4')
    with pytest.raises(LauncherConfigError, match='OBS 仮想カメラ等の入力'):
        parse_user_config(data, tmp_path)
    with pytest.raises(LauncherConfigError, match='DirectShow'):
        load_user_config(write_json(tmp_path / 'c.json', data))
    assert parse_user_config(data, tmp_path, allow_video=True).source == 'video'


def test_prepare_rejects_video_unless_dev_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_json(tmp_path / 'c.json', dict(source='video', video_path='a.mp4', output_dir=str(tmp_path / 'o')))
    with pytest.raises(SystemExit) as info:
        prepare(launcher_args(path, '--skip-manifest'))
    assert info.value.code == EXIT_CONFIG and 'OBS 仮想カメラ等の入力' in capsys.readouterr().err
    user, _ = prepare(launcher_args(path, '--skip-manifest', '--dev-allow-video'))
    assert user.source == 'video'
