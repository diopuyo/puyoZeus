"""配布入口での環境と機器別色補正を実際の読み込みで確かめる。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.phase_j import launcher
from src.phase_j.distribution_config import validate_settings
from src.phase_j.release_config import distribution_user_config
from src.production_config import RELEASE_LIVE_ENV_DEFAULTS, RELEASE_LIVE_MODEL_PATHS


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    monkeypatch.setattr(launcher, 'APP_ROOT', tmp_path / 'app')
    for relative in RELEASE_LIVE_MODEL_PATHS.values():
        path = tmp_path / 'app' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('設定試験専用', encoding='utf-8')
    return tmp_path


@pytest.mark.parametrize('existing', [None, '0', 'custom', ''])
def test_launcher_prepare_preserves_explicit_environment(
    root: Path, monkeypatch: pytest.MonkeyPatch, existing: str | None,
) -> None:
    for key in RELEASE_LIVE_ENV_DEFAULTS:
        if existing is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, existing)
    path = root / 'launcher.json'
    path.write_text(json.dumps({'device_name': 'OBS Virtual Camera',
                               'output_dir': str(root / 'output')}), encoding='utf-8')
    monkeypatch.setattr(launcher, 'missing_runtime_dlls', lambda *_: [])
    monkeypatch.setattr(launcher, 'check_telop_templates', lambda *_: None)
    options = launcher.parse_launcher_args(
        ['--config', str(path), '--skip-manifest', '--check-only'])
    launcher.prepare(options)
    assert all(os.environ[key] == (RELEASE_LIVE_ENV_DEFAULTS[key] if existing is None else existing)
               for key in RELEASE_LIVE_ENV_DEFAULTS)


@pytest.mark.parametrize('device, expected', [
    ('OBS Virtual Camera', '601to709'), ('USB Capture', 'off'),
    ('OBS Virtual Camera 2', 'off'),
])
@pytest.mark.parametrize('explicit', [None, 'off', 'auto', '601to709'])
def test_device_color_default_and_explicit_override(
    root: Path, device: str, expected: str, explicit: str | None,
) -> None:
    settings = validate_settings({'device_name': device})
    generated = distribution_user_config(settings, root, 8766)
    assert launcher.parse_user_config(generated, root).dshow_color_correction == expected
    data = {'device_name': device}
    if explicit is not None:
        data['dshow_color_correction'] = explicit
    path = root / 'camera.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    user = launcher.load_user_config(path)
    assert user.dshow_color_correction == (expected if explicit is None else explicit)
