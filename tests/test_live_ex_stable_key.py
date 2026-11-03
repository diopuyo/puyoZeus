"""共通予測の番号と鍵・事象の一致を検査する。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.phase_j import live_24h_config as config
from src.phase_j.live_ex_engine import LiveExEngine
from src.phase_j.live_ex_stable_key import StableKeyExEngine
from src.phase_j._live_ex_frozen.src.dl1k_engine import ExEngine
from src.phase_j._live_ex_frozen.src.dl1k_regen_state import EventMonitor


def engine() -> StableKeyExEngine:
    result = object.__new__(StableKeyExEngine)
    result.epoch = 1
    side = lambda: SimpleNamespace(board=[], land_count=0, landing=None, state='CHAIN')
    result.state = SimpleNamespace(t=12., game_idx=2, sides=[side(), side()])
    result.chains = [dict(signature=(10.,)), dict(signature=(11.,))]
    result._resolving = lambda side: True
    result.forecasts = [SimpleNamespace(chain_id=9., has_prediction=True), None]
    result.retr_fire = SimpleNamespace(current=lambda side, now: result.forecasts[side])
    return result


def test_refind_keeps_key_and_does_not_fire() -> None:
    subject = engine()
    monitor = EventMonitor()
    monitor.observe(subject, SimpleNamespace(t=12., index=0))
    monitor.take()
    before = subject.probe()
    subject.chains[0]['signature'] = (12.,)
    monitor.observe(subject, SimpleNamespace(t=13., index=1))
    assert subject.probe() == before
    assert monitor.take() == []


@pytest.mark.parametrize('side', [0, 1])
def test_new_chain_or_reply_changes_key_and_events(side: int) -> None:
    subject = engine()
    monitor = EventMonitor()
    monitor.observe(subject, SimpleNamespace(t=12., index=0))
    monitor.take()
    before = subject.probe()
    subject.forecasts[side] = SimpleNamespace(chain_id=13., has_prediction=True)
    monitor.observe(subject, SimpleNamespace(t=13., index=1))
    events = monitor.take()
    assert subject.probe().key != before.key
    assert any(e['event'] == 'fire' and e['side'] == f'p{side + 1}' for e in events)
    assert any(e['event'] == 'target' for e in events)


@pytest.mark.parametrize('mode', ['missing_tracker', 'legacy_tracker', 'missing_forecast', 'no_prediction'])
def test_without_prediction_uses_signature(mode: str) -> None:
    subject = engine()
    subject.forecasts = [None, None]
    if mode == 'missing_tracker':
        del subject.retr_fire
    elif mode == 'legacy_tracker':
        subject.retr_fire = SimpleNamespace()
    elif mode == 'no_prediction':
        subject.forecasts[0] = SimpleNamespace(chain_id=9., has_prediction=False)
    assert subject.probe() == ExEngine.probe(subject)
    subject.chains[0]['signature'] = (13.,)
    assert subject.probe().key[2][0] == 13.


def test_expired_prediction_falls_back_and_boundary_still_changes() -> None:
    subject = engine()
    subject.retr_fire = SimpleNamespace(current=lambda side, now: subject.forecasts[side] if now < 13. else None)
    assert subject.probe().key[2][0] == 9.
    subject.state.t = 13.
    assert subject.probe() == ExEngine.probe(subject)
    subject.epoch += 1
    subject.state.game_idx += 1
    assert subject.probe().key[:2] == (2, 3)


def test_only_predicted_side_changes_and_fight_is_unchanged() -> None:
    subject = engine()
    assert subject.probe().key == (1, 2, (9., 11.))
    subject.state.sides[0].board = None
    assert not subject.probe().fight


def test_adapter_default_and_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ExEngine, '__init__', lambda self, *args: None)
    assert type(LiveExEngine({}, None).engine) is ExEngine
    assert type(LiveExEngine({}, None, stable_key=True).engine) is StableKeyExEngine


@pytest.mark.parametrize('retr', ['0', '1'])
def test_tracking_is_independent_of_retr_calculation(retr: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j._live_ex_frozen.src import dl1k_retr
    calls = []
    tracker = SimpleNamespace(observe=lambda *args: calls.append(('forecast', args)))
    monkeypatch.setenv('PUYO_EX_RETR', retr)
    monkeypatch.setattr(dl1k_retr, 'fire_tracker', lambda native: tracker)
    subject = object.__new__(StableKeyExEngine)
    subject.native = None
    subject.observe = lambda *args: calls.append(('state', args))
    subject.step(1., 2, [])
    subject.step(2., 2, [])
    assert [kind for kind, args in calls] == ['state', 'forecast', 'state', 'forecast']
    assert calls[0][1] == calls[1][1] == (1., 2, [])
    assert subject.retr_fire is tracker


def test_cli_default() -> None:
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    assert parser.parse_args([]).live_24h_stable_key is False
    assert parser.parse_args(['--live-24h-stable-key']).live_24h_stable_key is True
    assert parser.parse_args(['--no-live-24h-stable-key']).live_24h_stable_key is False


@pytest.mark.parametrize('value', [1, 'true', None])
def test_invalid_flag(value: object) -> None:
    with pytest.raises(ValueError):
        config.validate_stable_key(value)


def test_launcher_to_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_ex_runtime
    from src.phase_j.launcher import build_pipeline_config, parse_user_config, USER_KEYS
    from tests.test_live_24h_launcher import data
    with patch.dict(os.environ):
        user = parse_user_config(data(tmp_path, live_24h=True, live_24h_model='ex.json',
                                      live_24h_stable_key=True), tmp_path)
        pipeline = build_pipeline_config(user)
        assert 'live_24h_stable_key' in USER_KEYS and pipeline['live_24h_stable_key']
        config.configure(SimpleNamespace(**pipeline), argparse.ArgumentParser())
        monkeypatch.setattr(live_ex_runtime, 'settings_from_env', lambda defaults: defaults)
        assert config.ex_settings()['stable_key']
        config.configure(SimpleNamespace(live_24h=False), argparse.ArgumentParser())
        assert os.environ[config.ENV_STABLE_KEY] == '0'
        assert not config.ex_settings()['stable_key']
