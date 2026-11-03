"""鮮度を広げても保持は0.5秒。設定経路と媒体・実時計の境界を検査する。"""
import argparse
import math
import os
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.phase_j import live_24h_config as config
from src.phase_j import live_ex_async, live_ex_delay, live_ex_runtime
from src.phase_j.live_ex_fresh import validate_ex_fresh_sec
from src.phase_j.live_ex_hold import LiveExHold
from src.phase_j.live_ex_hold_policy import HoldPolicy, Snapshot
from src.phase_j.live_ex_row_time import RowTimeEx
from tests.test_live_24h_delay import completed, owner
from tests.test_live_24h_hold import event, source
from tests.test_live_24h_row_time import key


@pytest.mark.parametrize('value', [0, -1, math.inf, -math.inf, math.nan, True, '1', None])
def test_invalid_freshness(value: object) -> None:
    with pytest.raises(ValueError, match='0より大きい有限'):
        validate_ex_fresh_sec(value)


@pytest.mark.parametrize('fresh', [.5, 1.])
def test_full_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh: float) -> None:
    from src.phase_j.launcher import parse_user_config, build_pipeline_config, USER_KEYS
    from tests.test_live_24h_launcher import data
    for name in [v for k, v in vars(config).items() if k.startswith('ENV_')]:
        monkeypatch.setenv(name, os.environ.get(name, ''))
    for name in ('PUYO_EX_WORK_BUDGET', 'PUYO_EX_WORK_LIMIT'):
        monkeypatch.setenv(name, os.environ.get(name, ''))
    user = parse_user_config(data(tmp_path, live_24h=True, live_24h_ex_fresh_sec=fresh,
                                  live_24h_model='ex.json'), tmp_path)
    pipeline = build_pipeline_config(user)
    assert 'live_24h_ex_fresh_sec' in USER_KEYS
    assert pipeline['live_24h_ex_fresh_sec'] == fresh
    config.configure(SimpleNamespace(**pipeline), argparse.ArgumentParser())
    monkeypatch.setattr(live_ex_runtime, 'settings_from_env', lambda defaults: defaults)
    assert config.ex_settings()['ex_fresh_sec'] == fresh
    monkeypatch.setattr(live_ex_async.AsyncEx, '__init__', lambda self, settings: None)
    worker = live_ex_delay.DelayedAsyncEx(dict(logical_delay_ms=200, ex_fresh_sec=fresh, row_time=True))
    assert worker.fresh_sec == worker.hold.policy.fresh_sec == worker.row_time.fresh_sec == fresh


def test_cli_and_disabled_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    assert parser.parse_args([]).live_24h_ex_fresh_sec == .5
    options = parser.parse_args(['--live-24h-ex-fresh-sec', '1.0'])
    monkeypatch.setenv(config.ENV_EX_FRESH_SEC, '7')
    config.configure_ex_fresh_sec(options, parser, True)
    assert os.environ[config.ENV_EX_FRESH_SEC] == '1.0'
    config.configure_ex_fresh_sec(options, parser, False)
    assert os.environ[config.ENV_EX_FRESH_SEC] == '0.5'
    options.live_24h_ex_fresh_sec = 0
    with pytest.raises(SystemExit):
        config.configure_ex_fresh_sec(options, parser, True)


@pytest.mark.parametrize(('age', 'accepted'), [(.99, True), (1., True), (1.01, False)])
@pytest.mark.parametrize('media', [False, True])
def test_delay_boundary(monkeypatch: pytest.MonkeyPatch, age: float, accepted: bool, media: bool) -> None:
    obj = owner([completed(input_t=0., captured_at=0., eligible_at=.2)])
    obj.fresh_sec, obj.media_fresh, obj.latest_input_t = 1., media, age
    monkeypatch.setattr(live_ex_delay, 'perf_counter', lambda: age)
    assert (obj.available(age, age)[0] is not None) == accepted


@pytest.mark.parametrize(('age', 'accepted'), [(.99, True), (1., True), (1.01, False)])
def test_row_time_selection_and_prune(age: float, accepted: bool) -> None:
    obj = RowTimeEx(fresh_sec=1.)
    obj.boundary(4)
    obj.observe(key(0.))
    rows = deque([completed(input_t=0., received_at=10.1)])
    assert (obj.select(rows, 20., age, 20., None)[0] is not None) == accepted
    obj.floor = age
    assert bool(obj.prune(rows)) == accepted


@pytest.mark.parametrize(('age', 'accepted'), [(1.49, True), (1.5, True), (1.500001, False)])
@pytest.mark.parametrize('cache', [False, True])
def test_held_and_cache_deadline(age: float, accepted: bool, cache: bool) -> None:
    policy = HoldPolicy(fresh_sec=1.)
    snap = Snapshot({}, 1., 0., 'original', 'board')
    policy.previous['p1'] = snap
    policy.cache[('p1', 'board')] = snap
    row = dict(t=age, current_cache_keys={'p1': 'board'} if cache else {})
    selected, mode = policy.select(row, 'p1')
    assert (selected is not None) == accepted
    assert mode == ('cache' if cache else 'held') if accepted else mode == 'missing'
    assert not policy.usable(Snapshot({}, .9, 0., 'earlier', None), 1.5)


@pytest.mark.parametrize('media', [False, True])
def test_live_original_and_held_deadline(media: bool) -> None:
    obj = LiveExHold(media_fresh=media, fresh_sec=1.)
    k = dict(key=(1, 1, (0., None)), fight=True)
    value, _ = obj.available(11., 1., 11., k, source(), 'available', 1)
    assert value is not None
    value, why = obj.available(11.5, 1.5, 11.5, k, None, 'not_ready_or_stale', 1)
    assert value is not None and why == 'held'
    assert obj.available(11.500001, 1.500001, 11.500001, k, None, 'not_ready_or_stale', 1)[0] is None


@pytest.mark.parametrize(('age', 'accepted'), [(.99, True), (1., True), (1.01, False)])
def test_policy_original_boundary(age: float, accepted: bool) -> None:
    obj = LiveExHold(fresh_sec=1.)
    if accepted:
        assert obj.adopt(event(age), source(), 'ok')['ex_hold_mode_p1'] == 'original'
    else:
        with pytest.raises(ValueError, match='鮮度'):
            obj.adopt(event(age), source(), 'ok')
