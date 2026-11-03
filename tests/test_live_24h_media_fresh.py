"""媒体時刻での鮮度・保持と、既定OFFの設定経路を確認する。"""
import argparse
import math
import os
from types import SimpleNamespace

import pytest

from src.phase_j import live_24h_config as config
from src.phase_j import live_ex_delay as delay
from src.phase_j import live_ex_async as async_ex
from src.phase_j.live_ex_hold import LiveExHold
from tests.test_live_24h_delay import completed, owner
from tests.test_live_24h_hold import source
from src.phase_j._live_ex_frozen.src.dl1k_columns import LEARNING_COLUMNS


def media_owner(rows: list[dict]) -> delay.DelayedAsyncEx:
    subject = owner(rows)
    subject.media_fresh, subject.latest_input_t = True, 3.
    subject.hold = LiveExHold(media_fresh=True)
    return subject


@pytest.mark.parametrize(('t', 'accepted'), [(2.999, False), (3., True),
                                           (3.5, True), (3.500001, False)])
def test_media_freshness(monkeypatch: pytest.MonkeyPatch, t: float, accepted: bool) -> None:
    monkeypatch.setattr(delay, 'perf_counter', lambda: 100.)
    subject = media_owner([completed()])
    assert (subject.available(100., t)[0] is not None) == accepted


@pytest.mark.parametrize('cutoff', [10.199999, 10.2])
def test_logical_delay_still_required(monkeypatch: pytest.MonkeyPatch, cutoff: float) -> None:
    monkeypatch.setattr(delay, 'perf_counter', lambda: 100.)
    subject = media_owner([completed()])
    assert (subject.available(100., 3., cutoff)[0] is not None) == (cutoff == 10.2)


def test_prunes_old_media_and_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(delay, 'perf_counter', lambda: 100.)
    subject = media_owner([completed(input_t=2.49), completed(input_t=2.5), completed(epoch=3)])
    subject.poll()
    assert len(subject.completed) == 1 and subject.completed[0]['input_t'] == 2.5


def test_late_receipt_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(async_ex, 'perf_counter', lambda: 100.)
    monkeypatch.setattr(delay, 'perf_counter', lambda: 100.)
    subject = media_owner([])
    assert subject.available(100., 3.)[0] is None
    subject.outputs.put(dict(kind='result', value=completed()))
    assert subject.available(100., 3., 99.99)[0] is None
    assert subject.available(100., 3., 100.)[0] is not None
    assert subject.completed[0]['eligible_at'] == 100.


@pytest.mark.parametrize('flag', [False, True])
def test_owner_initializes_hold(monkeypatch: pytest.MonkeyPatch, flag: bool) -> None:
    monkeypatch.setattr(async_ex.AsyncEx, '__init__', lambda self, settings: None)
    subject = delay.DelayedAsyncEx(dict(logical_delay_ms=200, media_fresh=flag))
    assert subject.media_fresh is flag and subject.hold.media_fresh is flag


@pytest.mark.parametrize(('t', 'accepted'), [(.5, True), (.500001, False)])
def test_original_boundary(t: float, accepted: bool) -> None:
    hold, key = LiveExHold(True), dict(key=(1, 1, (0., None)), fight=True)
    value, _ = hold.available(100., t, 10.2, key, source() if accepted else None, 'stale', 1)
    assert (value is not None) == accepted


@pytest.mark.parametrize(('attached', 't', 'accepted'), [(.25, .75, True), (.25, .750001, False),
                                                       (.5, 1., True), (.5, 1.000001, False)])
def test_hold_boundary(attached: float, t: float, accepted: bool) -> None:
    hold, key = LiveExHold(True), dict(key=(1, 1, (0., None)), fight=True)
    hold.available(100., attached, 10.2, key, source(), 'available', 1)
    value, _ = hold.available(1000., t, 10.3, key, None, 'not_ready_or_stale', 1)
    assert (value is not None) == accepted


@pytest.mark.parametrize('event', ['fire', 'landing', 'ojama_fall', 'boundary', 'target'])
def test_revalidation_and_reset(event: str) -> None:
    hold, key = LiveExHold(True), dict(key=(1, 1, (0., None)), fight=True)
    first, _ = hold.available(10.2, .2, 10.2, key, source(), 'available', 1)
    again, _ = hold.available(100., .2, 10.2, key, None, 'stale', 1)
    assert first == again
    hold.observe(dict(key, resets=[dict(side='p1', event=event)]))
    value, _ = hold.available(100., .2, 10.2, key, None, 'stale', 1)
    assert all(math.isnan(value['extra'][c]) for c in LEARNING_COLUMNS if c.endswith('_p1'))
    assert all(value['extra'][c] == 1. for c in LEARNING_COLUMNS if c.endswith('_p2'))


@pytest.mark.parametrize('value', ['yes', 1])
def test_invalid_flag(value: object, tmp_path) -> None:
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    assert parser.parse_args([]).live_24h_media_fresh is False
    with pytest.raises(SystemExit):
        config.configure(SimpleNamespace(live_24h=True, live_24h_media_fresh=value), parser)
    with pytest.raises(ValueError, match='live_24h_media_fresh'):
        config.launcher_values(dict(live_24h=True, live_24h_media_fresh=value), tmp_path)


def test_launcher_to_settings(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_ex_runtime
    from src.phase_j.launcher import build_pipeline_config, parse_user_config, USER_KEYS
    from tests.test_live_24h_launcher import data
    for name in (*[v for k, v in vars(config).items() if k.startswith('ENV_')],
                 'PUYO_EX_WORK_BUDGET', 'PUYO_EX_WORK_LIMIT'):
        monkeypatch.setenv(name, os.environ.get(name, ''))
    user = parse_user_config(data(tmp_path, live_24h=True, live_24h_media_fresh=True,
                                  live_24h_model='ex.json'), tmp_path)
    pipeline = build_pipeline_config(user)
    assert 'live_24h_media_fresh' in USER_KEYS and pipeline['live_24h_media_fresh'] is True
    config.configure(SimpleNamespace(**pipeline), argparse.ArgumentParser())
    monkeypatch.setattr(live_ex_runtime, 'settings_from_env', lambda defaults: defaults)
    assert config.ex_settings()['media_fresh'] is True
    config.configure(SimpleNamespace(live_24h=False), argparse.ArgumentParser())
    assert config.ex_settings()['media_fresh'] is False
