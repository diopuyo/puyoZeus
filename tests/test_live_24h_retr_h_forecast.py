"""retR の持ち時間 h を共通の連鎖の予測から作る旗 (live_24h_retr_h_forecast、依頼 F3、既定 OFF) の配送を検査する。"""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.phase_j import live_24h_config as config
from src.phase_j.live_ex_async import _compute
from src.phase_j._live_ex_frozen.src import dl1k_retr


@pytest.mark.parametrize('active', [False, True])
def test_child_sets_forecast_env(monkeypatch: pytest.MonkeyPatch, active: bool) -> None:
    seen = []
    monkeypatch.setenv('PUYO_RETR_H_FORECAST', str(int(not active)))
    connection = SimpleNamespace(recv=lambda: seen.append(dl1k_retr.h_forecast_enabled()), close=lambda: None)
    _compute(connection, dict(retr=True, retr_h_forecast=active))
    assert seen == [active]


def test_child_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    monkeypatch.setenv('PUYO_RETR_H_FORECAST', '1')
    connection = SimpleNamespace(recv=lambda: seen.append(dl1k_retr.h_forecast_enabled()), close=lambda: None)
    _compute(connection, dict(retr=True))
    assert seen == [False]


@pytest.mark.parametrize('value', ['yes', 1])
def test_invalid_flag_is_rejected(value: object, tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    assert parser.parse_args([]).live_24h_retr_h_forecast is False
    with pytest.raises(SystemExit):
        config.configure(SimpleNamespace(live_24h=True, live_24h_retr=True, live_24h_retr_h_forecast=value), parser)
    with pytest.raises(ValueError, match='live_24h_retr_h_forecast'):
        config.launcher_values(dict(live_24h=True, live_24h_retr=True, live_24h_retr_h_forecast=value), tmp_path)


def test_requires_retr(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        config.configure(SimpleNamespace(live_24h=True, live_24h_retr=False, live_24h_retr_h_forecast=True),
                         argparse.ArgumentParser())
    with pytest.raises(ValueError, match='live_24h_retr'):
        config.launcher_values(dict(live_24h=True, live_24h_retr=False, live_24h_retr_h_forecast=True), tmp_path)


def test_launcher_to_ex_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_ex_runtime
    from src.phase_j.launcher import build_pipeline_config, parse_user_config, USER_KEYS
    from tests.test_live_24h_launcher import data
    for key in (config.ENV_XC, config.ENV_RETR, config.ENV_RETR_H_FORECAST, config.ENV_ENABLED, config.ENV_AMA,
                config.ENV_DELAY, config.ENV_MODEL, config.ENV_TAPE_NEXT, config.ENV_MEDIA_FRESH, config.ENV_ROW_TIME,
                'PUYO_EX_WORK_BUDGET', 'PUYO_EX_WORK_LIMIT'):
        monkeypatch.setenv(key, os.environ.get(key, ''))
    user = parse_user_config(data(tmp_path, live_24h=True, live_24h_retr=True, live_24h_retr_h_forecast=True,
                                  live_24h_model='ex.json'), tmp_path)
    pipeline = build_pipeline_config(user)
    assert 'live_24h_retr_h_forecast' in USER_KEYS and pipeline['live_24h_retr_h_forecast'] is True
    config.configure(SimpleNamespace(**pipeline), argparse.ArgumentParser())
    monkeypatch.setattr(live_ex_runtime, 'settings_from_env', lambda defaults: defaults)
    assert config.ex_settings()['retr_h_forecast'] is True


def test_forecast_end_uses_forecast_steps() -> None:
    """予測 10 段なら、EX の 2 段の見積もりより後に終わる (PC2 の単体と同じ向き)。"""
    fc = SimpleNamespace(chain_id=100.9, steps=(40,) * 10, observed=(40, 40), last_valid_t=102.0, has_prediction=True)
    end = dl1k_retr.forecast_end(fc, 101.0)
    assert end > 100.0 + dl1k_retr.duration(2)
    assert end >= 100.0 + dl1k_retr.duration(10)
