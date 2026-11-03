"""live_verify_soft_hold の配線 (SPEED_DEFAULTS → 環境変数 → 子 process、JSON → ランチャ、実効値の記録の末尾)。

配線パッチ (wiring_soft_hold.patch) の一部。本体コミット (2d42562f) には含まれない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.phase_j import launcher, live_speed, live_verify_split as split

KEY = 'live_verify_soft_hold'


def test_default_off_and_recorded_last() -> None:
    assert live_speed.SPEED_DEFAULTS[KEY] is False
    assert live_speed.TRIAL_FLAG_KEYS[-1] == KEY
    assert KEY not in live_speed.ARM_TRIAL_KEYS      # OFF/X/ALL の分類は既存の項目のまま
    assert split.FLAG_ENV == 'PUYO_'+KEY.upper()


@pytest.mark.parametrize('enabled', [True, False])
def test_cli_to_environment_and_record(enabled: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('os.environ', {})
    parser = argparse.ArgumentParser()
    live_speed.add_arguments(parser)
    options = parser.parse_args(['--live-verify-soft-hold'] if enabled else [])
    options.output, options.source, options.cpu_threads = tmp_path, 'dshow', 1
    live_speed.configure(options, parser)
    import os
    assert os.environ[split.FLAG_ENV] == str(int(enabled)) and split.enabled() is enabled
    record = json.loads((tmp_path/live_speed.TRIAL_FLAGS_FILE).read_text(encoding='utf-8'))
    assert list(record['flags'])[-1] == KEY and record['flags'][KEY] is enabled
    assert list(record['env'])[-1] == split.FLAG_ENV and record['env'][split.FLAG_ENV] == str(int(enabled))


@pytest.mark.parametrize('enabled', [True, False])
def test_json_launcher_forwards_flag(enabled: bool, tmp_path: Path) -> None:
    data = dict(source='dshow', device_name='OBS', live_verify_soft_hold=enabled)
    config = launcher.build_pipeline_config(launcher.parse_user_config(data, tmp_path))
    assert config.get(KEY, False) is enabled


def test_default_config_does_not_gain_the_key(tmp_path: Path) -> None:
    """既定 OFF の設定は、従来の pipeline_config.json と同じキー集合のまま (新しい欄を書かない)。"""
    config = launcher.build_pipeline_config(
        launcher.parse_user_config(dict(source='dshow', device_name='OBS'), tmp_path))
    assert KEY not in config


def test_non_bool_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(launcher.LauncherConfigError):
        launcher.parse_user_config(dict(source='dshow', device_name='OBS', live_verify_soft_hold='yes'), tmp_path)
