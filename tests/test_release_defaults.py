"""配布既定と常駐生成、24w2との差の未記録増加を検査する。"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.phase_j import launcher
from src.phase_j import distribution_runtime as runtime_module
from src.phase_j.distribution_config import validate_settings
from src.phase_j.release_config import distribution_user_config, release_user_config
from src.production_config import (RELEASE_LIVE_A_FLAGS, RELEASE_LIVE_B_FLAGS,
                                   RELEASE_LIVE_DEFAULTS, RELEASE_LIVE_MODEL_PATHS,
                                   RELEASE_LIVE_OBS_FLAGS, RELEASE_LIVE_ENV_DEFAULTS)

FIXTURES = Path(__file__).parent / 'fixtures'
OBS_SHA256 = '783a489b42a69dd1675e0c4fc02dcf88272ce7f2ce39473cda6c04c31d99623e'
USER_KEYS = {'device_index', 'port', 'overlay'}
DIAGNOSTIC_KEYS = {'live_stage_timing', 'live_verify_reason_log'}
ADOPTED_FLAGS = RELEASE_LIVE_A_FLAGS + RELEASE_LIVE_B_FLAGS + RELEASE_LIVE_OBS_FLAGS


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """実モデルは読み込まず、既存のパス検証まで通す仮資産を作る。"""
    for relative in RELEASE_LIVE_MODEL_PATHS.values():
        path = tmp_path / 'app' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('設定試験専用', encoding='utf-8')
    monkeypatch.setattr(launcher, 'APP_ROOT', tmp_path / 'app')
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    monkeypatch.delenv('PUYO_LIVE_DISABLE_POWER_THROTTLING', raising=False)
    return tmp_path


def obs_config(root: Path) -> dict[str, Any]:
    data = json.loads((FIXTURES / 'release_obs_24w2.json').read_text(encoding='utf-8'))
    for key in RELEASE_LIVE_MODEL_PATHS:
        data[key] = str(root / data[key].replace('\\', '/'))
    return data


def load_minimal(root: Path) -> launcher.UserConfig:
    path = root / 'minimal.json'
    path.write_text(json.dumps({'device_name': 'OBS Virtual Camera'}), encoding='utf-8')
    return launcher.load_user_config(path)


def test_reference_is_the_exact_24w2_input() -> None:
    raw = (FIXTURES / 'release_obs_24w2.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == OBS_SHA256
    assert len(json.loads(raw)) == 87
    assert len(RELEASE_LIVE_A_FLAGS) == 19
    assert len(RELEASE_LIVE_B_FLAGS) == 14
    assert len(RELEASE_LIVE_OBS_FLAGS) == 19


@pytest.mark.parametrize('key', ADOPTED_FLAGS)
def test_adopted_flags_reach_both_entries_and_pipeline(root: Path, key: str) -> None:
    generated = distribution_user_config(validate_settings({}), root, 8766)
    resident = launcher.parse_user_config(generated, root)
    direct = load_minimal(root)
    assert obs_config(root)[key] is True
    assert generated[key] is True
    assert getattr(resident, key) is True and getattr(direct, key) is True
    assert launcher.build_pipeline_config(resident)[key] is True
    assert launcher.build_pipeline_config(direct)[key] is True


def test_operating_values_and_models_are_complete(root: Path) -> None:
    user = load_minimal(root)
    for key, value in RELEASE_LIVE_DEFAULTS.items():
        assert getattr(user, key) == value
    for key, relative in RELEASE_LIVE_MODEL_PATHS.items():
        assert getattr(user, key) == root / 'app' / relative
    assert user.live_verify_async is True


@pytest.mark.parametrize('key', ADOPTED_FLAGS)
def test_explicit_false_is_not_replaced(root: Path, key: str) -> None:
    source = {key: False}
    assert release_user_config(source, root / 'app')[key] is False
    assert source == {key: False}


def test_loaded_explicit_false_and_relative_model_are_preserved(root: Path) -> None:
    path = root / 'explicit.json'
    (root / 'custom.json').write_text('設定試験専用', encoding='utf-8')
    changes = {key: False for key in RELEASE_LIVE_B_FLAGS}
    changes.update(device_name='camera', live_disable_power_throttling=False,
                   live_24h_model='custom.json', b3_ensemble=False)
    path.write_text(json.dumps(changes), encoding='utf-8')
    user = launcher.load_user_config(path)
    assert all(getattr(user, key) is False for key in RELEASE_LIVE_B_FLAGS)
    assert user.live_24h_model == root / 'custom.json'
    assert not user.b3_ensemble and not user.live_disable_power_throttling


def test_missing_model_does_not_fall_back(root: Path) -> None:
    (root / 'app' / RELEASE_LIVE_MODEL_PATHS['live_24h_model']).unlink()
    with pytest.raises(launcher.LauncherConfigError, match='live_24h_modelがありません'):
        load_minimal(root)


def test_known_24w2_differences_are_exhaustive(root: Path) -> None:
    resident_path, obs_path = root / 'resident.json', root / 'obs.json'
    resident_path.write_text(json.dumps(distribution_user_config(
        validate_settings({}), root, 8766)), encoding='utf-8')
    obs_path.write_text(json.dumps(obs_config(root)), encoding='utf-8')
    actual = asdict(launcher.load_user_config(resident_path))
    expected = asdict(launcher.load_user_config(obs_path))
    differences = {key: {'obs': expected[key], 'release': value}
                   for key, value in actual.items() if value != expected[key]}
    assert len(actual) == 110 and len(differences) == 5
    assert actual['live_verify_entry_fast_retry'] is expected['live_verify_entry_fast_retry'] is True
    for key in ('live_chain_forecast_recheck', 'live_chain_end_formula'):
        assert actual[key] is expected[key] is True
    assert set(differences) == USER_KEYS | DIAGNOSTIC_KEYS
    # 生JSONでは省電力が未指定。配布入口を通す前の差も隠さず固定する。
    raw = asdict(launcher.parse_user_config(obs_config(root), root))
    assert {key for key in actual if actual[key] != raw[key]} == (
        USER_KEYS | DIAGNOSTIC_KEYS | {'live_disable_power_throttling',
        'live_verify_entry_fast_retry', 'live_chain_forecast_recheck', 'live_chain_end_formula'})
    assert not raw['live_disable_power_throttling']
    assert actual['live_disable_power_throttling'] is True
    recorded = json.loads((FIXTURES / 'release_known_differences.json').read_text(encoding='utf-8'))
    assert recorded == {key: {'obs': raw[key], 'release': actual[key]} for key in actual
                        if raw[key] != actual[key] and key not in USER_KEYS | DIAGNOSTIC_KEYS}


@pytest.mark.parametrize('existing', [None, '0', 'custom', ''])
def test_runtime_start_writes_the_complete_configuration(root: Path, existing: str | None,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    for key in RELEASE_LIVE_ENV_DEFAULTS:
        if existing is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, existing)
    runtime = runtime_module.DistributionRuntime(root)
    runtime.settings = validate_settings(dict(auto_input=False, check_updates=False,
        device_name='selected camera', device_index=2, overlay={'x': 123, 'y': 456}))
    monkeypatch.setattr(runtime_module, 'available_port', lambda *_: 8777)
    monkeypatch.setattr(runtime_module, 'close_job', lambda *_: None)
    monkeypatch.setattr(runtime_module, 'create_job', lambda: None)
    monkeypatch.setattr(runtime_module, 'assign_job', lambda *_: None)
    monkeypatch.setattr(runtime_module.threading, 'Thread',
                        lambda **_: SimpleNamespace(start=lambda: None))
    calls = []

    def launch(command: list[str], **kwargs: Any) -> Any:
        calls.append((command, kwargs))
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(runtime_module.subprocess, 'Popen', launch)
    (root / 'puyo_live.json').write_text('{"old": true}', encoding='utf-8')
    runtime.start()
    written = json.loads((root / 'puyo_live.json').read_text(encoding='utf-8'))
    assert written == distribution_user_config(runtime.settings, root, 8777)
    assert written['overlay']['x'] == 123 and written['device_index'] == 2
    assert len(calls) == 1 and '--require-manifest' in calls[0][0]
    assert calls[0][1]['cwd'] == root / 'app'
    assert all(calls[0][1]['env'][key] == (RELEASE_LIVE_ENV_DEFAULTS[key] if existing is None else existing)
               for key in RELEASE_LIVE_ENV_DEFAULTS)
    assert launcher.load_user_config(root / 'puyo_live.json').live_24h


def test_generated_config_does_not_share_mutable_overlay(root: Path) -> None:
    settings = validate_settings({'overlay': {'x': 123}})
    first = distribution_user_config(settings, root, 8766)
    first['overlay']['x'] = 999
    assert distribution_user_config(settings, root, 8766)['overlay']['x'] == 123
