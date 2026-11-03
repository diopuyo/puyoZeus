"""R37 の配信設定が実際の認識器まで独立して届くことを確かめる。"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import pytest

from src import production_config
from src.phase_j import launcher, live_speed
from tests.launcher_legacy import legacy_user_config
from src.phase_j.live_process import ProcessRecognitionBridge
from src.recognition_pipeline import RecognitionPipeline

KEY = 'live_lockdown_score_zero_then_move_release'
PIPE_KEY = 'enable_lockdown_score_zero_then_move_release'


def _load_pipeline(config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> RecognitionPipeline:
    """認識ロジックと組立は実物を使い、画像・モデル資産の読込だけを置き換える。"""
    from src import recognition_pipeline as module
    from tests.test_recognition_pipeline import _StubImageReader, _StubMatchDetector, _empty_board

    reader = _StubImageReader(_empty_board(), _empty_board())
    monkeypatch.setattr(RecognitionPipeline, 'DEFAULT_CNN_MODEL_PATH', Path('__unused_model__'))
    monkeypatch.setattr(module, 'ImageReader', lambda **kwargs: reader)
    monkeypatch.setattr(module.MatchStateDetector, 'load_default', lambda: _StubMatchDetector(False))
    monkeypatch.setattr(module, '_build_score_zero_detector', lambda *args: None)
    monkeypatch.setattr(module.MatchEndDetector, 'load_default', lambda **kwargs: None)
    monkeypatch.setattr(module.TelopDetector, 'load_default', lambda: None)
    return RecognitionPipeline.load_default(
        load_score_ocr=False, load_next_detector=False, enable_chain_tracker=False, **config)


@pytest.mark.parametrize('data, expected, robust', [
    ({KEY: True}, True, False),
    ({KEY: False}, False, False),
    ({}, False, False),
    ({KEY: None}, False, False),
    ({KEY: False, 'live_frame_drop_robust': True}, False, True),
])
def test_json_to_real_pipeline(
    data: dict[str, Any], expected: bool, robust: bool,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON → ランチャ → CLI → 環境変数 → process 設定 → 実物の load_default。"""
    from scripts.run_live_pipeline_20260928 import parse_args

    monkeypatch.setattr(os, 'environ', {})
    # 前の起動で ON でも、今回の OFF・未指定・null で持ち越さない。
    monkeypatch.setenv(live_speed.SCORE_ZERO_THEN_MOVE_ENV, '1')
    path = tmp_path / 'puyo_live.json'
    path.write_text(json.dumps(legacy_user_config(dict(device_name='test', **data))), encoding='utf-8')
    user = launcher.load_user_config(path)
    assert getattr(user, KEY) is expected
    config_path = tmp_path / 'pipeline_config.json'
    config_path.write_text(json.dumps(launcher.build_pipeline_config(user)), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['live', *launcher.build_pipeline_argv(user, config_path)])
    parse_args()
    assert os.environ[live_speed.SCORE_ZERO_THEN_MOVE_ENV] == str(int(expected))
    bridge = object.__new__(ProcessRecognitionBridge)
    bridge.device, bridge.config_path = 'device.json', tmp_path / 'recognition_config.json'
    config = bridge.pipeline_config(**production_config.recognition_load_default_kwargs())
    assert json.loads(bridge.config_path.read_text(encoding='utf-8')) == config
    # spawn へ渡す辞書を直列化してから、子と同じファクトリで認識器を構築する。
    pipe = _load_pipeline(pickle.loads(pickle.dumps(config)), monkeypatch)
    assert getattr(pipe, '_' + PIPE_KEY) is expected
    assert pipe._enable_media_tick_counters is robust


def test_r37_default_uses_production_config() -> None:
    assert production_config.LIVE_LOCKDOWN_SCORE_ZERO_THEN_MOVE_DEFAULT is False
    assert live_speed.SPEED_DEFAULTS[KEY] is production_config.LIVE_LOCKDOWN_SCORE_ZERO_THEN_MOVE_DEFAULT


@pytest.mark.parametrize('value', [1, 'true'])
def test_r37_rejects_non_bool(value: object) -> None:
    with pytest.raises(ValueError, match=KEY):
        live_speed.validate({KEY: value})
