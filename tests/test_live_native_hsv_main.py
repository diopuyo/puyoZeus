"""配信本体の Rust HSV スイッチ・子への配送・失敗理由を検証する。"""
from __future__ import annotations

import argparse
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.image_reader import ColorClassifier, ImageReader
from src.phase_j import live_native_hsv, live_speed

KEY = live_speed.NATIVE_HSV_MAIN_KEY
ENV = 'PUYO_'+KEY.upper()


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """設定関数が変更した環境変数を次の試験へ残さない。"""
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(live_native_hsv.RECORD_PATH_ENV, raising=False)


@pytest.mark.parametrize('enabled', [False, True])
def test_launcher_cli_and_child_config(enabled: bool, tmp_path: Path) -> None:
    from src.phase_j.launcher import parse_user_config, build_pipeline_config
    user = parse_user_config(dict(device_name='OBS Virtual Camera', **{KEY: enabled}), tmp_path)
    config = build_pipeline_config(user)
    assert config.get(KEY, False) is enabled
    parser = argparse.ArgumentParser()
    live_speed.add_arguments(parser)
    options = parser.parse_args(['--live-native-hsv-main'] if enabled else [])
    options.cpu_threads, options.source, options.output = 1, 'dshow', tmp_path
    live_speed.configure(options, parser)
    assert os.environ[ENV] == str(int(enabled))
    root = str(Path(__file__).resolve().parents[1])
    code = (f'import sys; sys.path.insert(0, {root!r}); import json; '
            'from src.phase_j.live_speed import recognition_config; '
            'print(json.dumps(recognition_config({}, True)))')
    child = subprocess.run([sys.executable, '-B', '-c', code], env=os.environ,
                           capture_output=True, text=True, check=True)
    assert json.loads(child.stdout).get('enable_native_hsv_classifier', False) is enabled
    record = json.loads((tmp_path/live_speed.TRIAL_FLAGS_FILE).read_text(encoding='utf-8'))
    assert record['flags'][KEY] is enabled
    assert record['runtime'][KEY]['active'] is False
    assert Path(os.environ[live_native_hsv.RECORD_PATH_ENV]).parent == tmp_path


def test_default_off_preserves_recognition_config() -> None:
    from src.production_config import LIVE_NATIVE_HSV_MAIN_DEFAULT
    assert LIVE_NATIVE_HSV_MAIN_DEFAULT is live_speed.SPEED_DEFAULTS[KEY] is False
    start = live_speed.TRIAL_FLAG_KEYS.index(KEY)
    assert live_speed.TRIAL_FLAG_KEYS[start:start+2] == (KEY, 'live_verify_async')   # 後ろに別の鍵が続いてよい
    assert KEY not in live_speed.ARM_TRIAL_KEYS
    for live_input in (False, True):
        assert 'enable_native_hsv_classifier' not in live_speed.recognition_config({}, live_input)


@pytest.mark.parametrize('value', [None, 1, 'true', []])
def test_invalid_flag_rejected(value: object) -> None:
    with pytest.raises(ValueError, match=KEY):
        live_speed.validate({KEY: value})


@pytest.mark.parametrize('mode,reason', [
    ('missing', 'native_extension_unavailable'),
    ('symbol', 'classify_cells_hsv_unavailable'),
    ('vote', 'vote_mode_unsupported'),
    ('unsupported', 'hsv_classifier_unavailable'),
    ('inactive', 'activation_failed'),
    ('dll', 'native_extension_unavailable'),
])
def test_unavailable_records_reason(mode: str, reason: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV, '1')
    classifier = ColorClassifier(vote_mode=mode == 'vote')
    if mode == 'missing':
        monkeypatch.setitem(sys.modules, 'puyo_core', None)
    elif mode in ('symbol', 'inactive'):
        module = SimpleNamespace(**({'classify_cells_hsv': object()} if mode == 'inactive' else {}))
        monkeypatch.setitem(sys.modules, 'puyo_core', module)
    elif mode == 'dll':
        original_import = builtins.__import__
        def failing_import(name: str, *args: object, **kwargs: object) -> object:
            if name == 'puyo_core':
                raise OSError('DLL load failed')
            return original_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', failing_import)
    reader = ImageReader(classifier=classifier if mode != 'unsupported' else object())
    if mode != 'inactive':
        assert reader.enable_native_hsv() is False
    status = live_native_hsv.runtime_status(SimpleNamespace(_reader=reader))
    assert status['active'] is False
    assert status['reason'].startswith(reason)


def test_actual_activation_and_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip('puyo_core')
    from src.recognition_pipeline import RecognitionPipeline
    monkeypatch.setenv(ENV, '1')
    pipe = RecognitionPipeline(image_reader=ImageReader(), match_state_detector=None,
                               **live_speed.recognition_config({}, True))
    options = argparse.Namespace(output=tmp_path, **{KEY: True})
    live_speed.write_trial_flags(options)
    before = json.loads((tmp_path/live_speed.TRIAL_FLAGS_FILE).read_text(encoding='utf-8'))
    live_native_hsv.configure_record_path(tmp_path)
    assert live_native_hsv.record_runtime(pipe) == dict(active=True, reason='enabled')
    after = json.loads((tmp_path/live_speed.TRIAL_FLAGS_FILE).read_text(encoding='utf-8'))
    assert after['flags'] == before['flags'] and after['env'] == before['env']
    assert after['runtime'][KEY] == dict(active=True, reason='enabled')
    live_native_hsv.configure_record_path(None)
    assert live_native_hsv.RECORD_PATH_ENV not in os.environ


def test_off_does_not_inspect_reader() -> None:
    assert live_native_hsv.runtime_status(object()) == dict(active=False, reason='disabled')


def test_failed_activation_is_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """要求 ON のまま実効 OFF と失敗理由をファイルにも残す。"""
    monkeypatch.setenv(ENV, '1')
    monkeypatch.setitem(sys.modules, 'puyo_core', None)
    reader = ImageReader()
    assert reader.enable_native_hsv() is False
    live_speed.write_trial_flags(argparse.Namespace(output=tmp_path, **{KEY: True}))
    live_native_hsv.configure_record_path(tmp_path)
    live_native_hsv.record_runtime(SimpleNamespace(_reader=reader))
    record = json.loads((tmp_path/live_speed.TRIAL_FLAGS_FILE).read_text(encoding='utf-8'))
    assert record['flags'][KEY] is True
    assert record['runtime'][KEY]['active'] is False
    assert 'native_extension_unavailable' in record['runtime'][KEY]['reason']
