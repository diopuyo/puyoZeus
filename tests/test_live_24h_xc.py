"""xc の設定配送・モデル列・保持を検査する。"""
import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.phase_j import live_24h_config as config
from src.phase_j.live_ex_async import _compute
from src.phase_j.live_24h_model import XC_COLUMNS, RETR_COLUMNS, validate_model
from src.phase_j.live_24h import Live24h, LEARNING_COLUMNS
from tests.test_live_24h_model import Worker, model, result


@pytest.mark.parametrize('active', [False, True])
def test_child_overrides_inherited_xc(monkeypatch: pytest.MonkeyPatch, active: bool) -> None:
    seen = []
    monkeypatch.setenv('PUYO_EX_XC', str(int(not active)))
    monkeypatch.setenv('PUYO_EX_RETR', '0')
    connection = SimpleNamespace(recv=lambda: seen.append(os.environ['PUYO_EX_XC']), close=lambda: None)
    _compute(connection, dict(retr=True, xc=active))
    assert seen == [str(int(active))]


@pytest.mark.parametrize('xc', [True, 'yes'])
def test_invalid_xc_is_rejected(xc: object, tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    assert parser.parse_args([]).live_24h_xc is False
    with pytest.raises(SystemExit):
        config.configure(SimpleNamespace(live_24h=True, live_24h_xc=xc), parser)
    with pytest.raises(ValueError, match='live_24h_xc'):
        config.launcher_values(dict(live_24h=True, live_24h_xc=xc), tmp_path)


def test_launcher_cli_settings_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_ex_runtime
    from src.phase_j.launcher import build_pipeline_config, parse_user_config, USER_KEYS
    from tests.test_live_24h_launcher import data
    for key in (config.ENV_XC, config.ENV_RETR, config.ENV_ENABLED, config.ENV_AMA,
                config.ENV_DELAY, config.ENV_MODEL, 'PUYO_EX_WORK_BUDGET', 'PUYO_EX_WORK_LIMIT'):
        monkeypatch.setenv(key, os.environ.get(key, ''))
    user = parse_user_config(data(tmp_path, live_24h=True, live_24h_retr=True,
                                  live_24h_xc=True, live_24h_model='ex.json'), tmp_path)
    pipeline = build_pipeline_config(user)
    assert 'live_24h_xc' in USER_KEYS and pipeline['live_24h_xc'] is True
    config.configure(SimpleNamespace(**pipeline), argparse.ArgumentParser())
    monkeypatch.setattr(live_ex_runtime, 'settings_from_env', lambda defaults: defaults)
    settings = config.ex_settings()
    assert settings['xc'] is True and settings['retr'] is True


@pytest.mark.parametrize('declared', [False, True])
def test_xc_model_columns(tmp_path: Path, declared: bool) -> None:
    from src.b3_column_spec import NET_LEDGER_FEATURE_NAMES
    calls, candidate = [], model([])
    candidate = model(calls)
    columns = (*NET_LEDGER_FEATURE_NAMES, *LEARNING_COLUMNS, *RETR_COLUMNS,
               *(XC_COLUMNS if declared else ()))
    candidate.static = candidate.exchange = SimpleNamespace(columns=columns)
    candidate.members = (1, 2, 3)
    validate_model(candidate)
    worker = Worker(dict(epoch=1, serial=2, extra=dict.fromkeys(LEARNING_COLUMNS + XC_COLUMNS, .7)))
    owner = Live24h(tmp_path, candidate, {}, lambda _: worker, use_ama=True, temperature_enabled=False)
    try:
        owner.predict(result())
        assert set(calls[0]) & set(XC_COLUMNS) == (set(XC_COLUMNS) if declared else set())
    finally:
        owner.close()


def test_xc_hold_side_reset_and_expiry() -> None:
    from src.phase_j.live_ex_hold import LiveExHold
    from tests.test_live_24h_hold import source, event
    hold, src = LiveExHold(), source()
    src['extra'].update(dict.fromkeys(XC_COLUMNS, 7.))
    hold.adopt(event(.2), src, 'ok')
    assert all(hold.adopt(event(.3), None, 'not_yet')[c] == 7. for c in XC_COLUMNS)
    reset = hold.adopt(event(.4, reset={'p1': True, 'p2': False}), None, 'stale')
    assert all(math.isnan(reset[c]) if c.endswith('_p1') else reset[c] == 7. for c in XC_COLUMNS)
    assert all(math.isnan(hold.adopt(event(.71), None, 'stale')[c]) for c in XC_COLUMNS)
