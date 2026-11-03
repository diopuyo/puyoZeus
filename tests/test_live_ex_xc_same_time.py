"""観測の写し、16列だけの採用、二種類の状態子と履歴を検査する。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
import json
import os
from io import StringIO
from pathlib import Path
from queue import Queue
import struct
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from src.phase_j import live_ex_async as async_ex
from src.phase_j import live_ex_compute as calc
from src.phase_j import live_ex_xc_same_time as same
from src.phase_j.live_ex_fire_first import FireFirstState
from src.phase_j.live_ex_row_time import RowTimeEx
from src.phase_j._live_ex_frozen.src.dl1k_columns import LEARNING_COLUMNS
from src.phase_j._live_ex_frozen.src.dl1k_retr import COLUMNS as RETR
from tests.test_live_ex_xc import exchange


class SmallEngine:
    """実xcと実retRを使い、射影の副作用を確実に起こす小さな計測台。"""

    def projection(self) -> dict:
        self.chains[0]['all_clear_bonus'] = 2100
        self.state.sl_snap['projection_test'] = True
        return deepcopy(self.value)


def small_engine() -> SmallEngine:
    fixture, value = exchange()
    engine = SmallEngine()
    engine.__dict__.update(vars(fixture))
    del engine.projection
    engine.value = value
    engine.knowledge = SimpleNamespace(lookup=None)
    engine.chains[0]['all_clear_bonus'] = 0
    return engine


def freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, dict):
        return {k: freeze(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [freeze(v) for v in value]
    if callable(value):
        return repr(value)
    if hasattr(value, '__dict__') and not hasattr(value, '__spec__'):
        return freeze(vars(value))
    return value


@pytest.mark.parametrize('parallel', [False, True])
def test_real_xc_matches_compute_and_preserves_state(parallel: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_EX_RETR', '1')
    monkeypatch.setenv('PUYO_EX_XC', '1')
    monkeypatch.setenv('PUYO_RETR_FORECAST', '1')
    subject, reference = small_engine(), small_engine()
    reference.ex_speed = parallel
    fields = ('state', 'chains', 'knowledge', 'retr_fire')
    before = freeze({name: getattr(subject, name) for name in fields})
    before_key = subject.probe()
    monkeypatch.setattr(calc, 'proof_of', lambda *args: {})
    monkeypatch.setattr(calc, 'search_of', lambda *args: (None, None, None))
    actual, expected = same.observe_xc(subject), calc.compute(reference)
    assert expected['failure'] is None
    assert not any(str(w).startswith('error:') for w in actual['xc_why'])
    for column in same.COLUMNS:
        assert struct.pack('!d', actual['xc'][column]) == struct.pack('!d', expected['xc'][column]), column
    assert freeze({name: getattr(subject, name) for name in fields}) == before
    assert subject.probe() == before_key
    assert subject.chains[0]['all_clear_bonus'] == 0
    assert reference.chains[0]['all_clear_bonus'] == 2100
    assert not hasattr(subject, 'xc_diag')
    assert actual['xc_ms'] >= sum(actual['xc_ms_by_stage'].values())


@pytest.mark.parametrize('stage', ['snapshot', 'projection', 'xc'])
def test_failure_is_only_xc(stage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = SimpleNamespace(native=None, knowledge=SimpleNamespace(lookup=None), chains=[], state={})
    def fail(*args: Any) -> None:
        raise ValueError('意図した失敗')
    engine.projection = fail if stage == 'projection' else lambda: {}
    if stage == 'snapshot':
        monkeypatch.setattr(same, 'snapshot', fail)
    if stage == 'xc':
        monkeypatch.setattr(calc, 'compute_xc', fail)
    result = same.observe_xc(engine)
    assert set(result['xc']) == set(same.COLUMNS)
    assert all(math.isnan(v) for v in result['xc'].values())
    assert result['xc_why'][0].startswith('error:')
    values = dict.fromkeys((*LEARNING_COLUMNS, *RETR), 7.)
    values.update(dict.fromkeys(same.COLUMNS, 3.))
    same.select_xc(values, dict(result, fight=True, input_t=2., serial=1), None)
    assert all(values[c] == 7. for c in (*LEARNING_COLUMNS, *RETR))
    assert all(math.isnan(values[c]) for c in same.COLUMNS)


@pytest.mark.parametrize('state_type', [async_ex._State, FireFirstState])
@pytest.mark.parametrize('enabled,fight', [(False, True), (True, False), (True, True)])
def test_both_state_workers(state_type: type, enabled: bool, fight: bool,
                            monkeypatch: pytest.MonkeyPatch) -> None:
    output, engine = Queue(), SimpleNamespace(engine=object())
    engine.observe = lambda _: SimpleNamespace(fight=fight, key=(1, 1, (1, None)))
    engine.take_resets = lambda: []
    monkeypatch.setattr(async_ex, 'create_engine', lambda *_: engine)
    calls = []
    def observed(value: Any) -> dict:
        calls.append(value)
        return dict(xc=dict.fromkeys(same.COLUMNS, 2.), xc_ms=3.)
    monkeypatch.setattr(same, 'observe_xc', observed)
    connection = [None, None] if state_type is FireFirstState else None
    state = state_type(dict(xc_same_time=enabled), connection, output)
    state.observe(dict(epoch=1, packet={}, serial=9, input_t=4., captured_at=5.))
    output.get()
    event = output.get()
    assert event['serial'] == 9 and event['input_t'] == 4.
    assert ('xc' in event) == (enabled and fight)
    assert len(calls) == int(enabled and fight)
    if not enabled:
        assert set(event) == {'kind', 'serial', 'key', 'epoch', 'fight', 'input_t', 'captured_at',
                              'observed_at', 'requested_at', 'resets'}


@pytest.mark.parametrize('row_time', [False, True])
def test_history_epoch_and_row_selection(row_time: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_ex_delay import DelayedAsyncEx
    from tests.test_live_24h_delay import owner
    from src.phase_j.live_ex_hold import LiveExHold
    obj = owner([])
    obj.hold, obj.fresh_sec, obj.latest_input_t = LiveExHold(), 2., 3.
    obj.epoch = 1
    obj.row_time = RowTimeEx() if row_time else None
    obj.xc_history = obj.row_time or RowTimeEx()
    obj.protect_rows(1., 1)
    for serial, stamp in enumerate((1., 2., 3.), 1):
        obj.outputs.put(dict(kind='key', epoch=1, serial=serial, input_t=stamp, fight=True,
                             key=(1, 1, (1, None)), xc=dict.fromkeys(same.COLUMNS, stamp), resets=[]))
    obj.poll()
    key = obj.xc_history.key_at(2.5)
    assert key['serial'] == 2
    values = dict.fromkeys((*LEARNING_COLUMNS, *RETR, *same.COLUMNS), 7.)
    audit = same.select_xc(values, key, None)
    assert all(values[c] == 7. for c in (*LEARNING_COLUMNS, *RETR))
    assert all(values[c] == 2. for c in same.COLUMNS)
    assert audit['xc_source'] == 'same_time' and audit['xc_input_t'] == 2.
    obj.protect_rows(3., 2)
    assert obj.xc_history.key_at(4.) == {}


@pytest.mark.parametrize('held,source', [(False, 'result'), (True, 'hold')])
def test_fallback_keeps_values_and_side_times(held: bool, source: str) -> None:
    values = dict.fromkeys(same.COLUMNS, 7.)
    extra = dict(held=held, hold_audit=dict(ex_hold_source_input_t_p1=1., ex_hold_source_input_t_p2=2.))
    for key in ({}, dict(fight=False, xc={}), dict(fight=True)):
        audit = same.select_xc(values, key, extra)
        assert audit['xc_source'] == source
        assert audit['xc_input_t'] is None
        assert audit['xc_input_t_by_side'] == dict(p1=1., p2=2.)
        assert values == dict.fromkeys(same.COLUMNS, 7.)
    assert same.select_xc({}, {}, None)['xc_source'] == 'none'


def test_config_defaults() -> None:
    from src.phase_j import live_24h_config as config
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    assert parser.parse_args([]).live_24h_xc_same_time is False
    assert parser.parse_args(['--live-24h-xc-same-time']).live_24h_xc_same_time is True
    for bad in (1, 'true', None):
        with pytest.raises(ValueError):
            config.validate_xc_same_time(bad)
    with pytest.raises(ValueError, match='live_24h_xc'):
        config.validate_xc_same_time(True, False)


def test_configuration_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_ex_runtime, live_24h_config as config
    from src.phase_j.launcher import build_pipeline_config, parse_user_config, USER_KEYS
    from tests.test_live_24h_launcher import data
    with patch.dict(os.environ):
        user = parse_user_config(data(tmp_path, live_24h=True, live_24h_model='ex.json',
            live_24h_xc=True, live_24h_retr=True, live_24h_xc_same_time=True), tmp_path)
        pipeline = build_pipeline_config(user)
        assert 'live_24h_xc_same_time' in USER_KEYS and pipeline['live_24h_xc_same_time']
        config.configure(SimpleNamespace(**pipeline), argparse.ArgumentParser())
        monkeypatch.setattr(live_ex_runtime, 'settings_from_env', lambda defaults: defaults)
        assert config.ex_settings()['xc_same_time']
        config.configure(SimpleNamespace(live_24h=False), argparse.ArgumentParser())
        assert not config.ex_settings()['xc_same_time']
        assert os.environ[config.ENV_XC_SAME_TIME] == '0'


@pytest.mark.parametrize('enabled', [False, True])
def test_selection_and_result_audit(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_24h import Live24h
    columns = (*LEARNING_COLUMNS, *RETR, *same.COLUMNS)
    extra = dict(epoch=1, serial=3, extra=dict.fromkeys(columns, 7.), held=False, input_t=1.)
    key = dict(epoch=1, serial=9, input_t=2., fight=True, xc=dict.fromkeys(same.COLUMNS, 2.), xc_ms=1.)
    history = RowTimeEx()
    history.boundary(1)
    history.observe(key)
    engine_result = dict(extra=extra['extra'], epoch=1, serial=9, result=dict(xc=key['xc']))
    events = [dict(key, kind='key'), dict(kind='result', value=engine_result)]
    worker = SimpleNamespace(xc_history=history, available=lambda *a, **k: (extra, 'available'),
                             selection_record=lambda *a: {}, take_events=lambda: events)
    live = object.__new__(Live24h)
    live.settings, live.worker = dict(xc_same_time=enabled), worker
    live.ex_columns, live.clock, live.audit = columns, lambda: 5., StringIO()
    monkeypatch.setenv('PUYO_LIVE_EX_AUDIT', '1')
    values, signature, reason = live.selection(dict(frame=1, t_sec=2.))
    assert all(values[c] == 7. for c in (*LEARNING_COLUMNS, *RETR))
    assert all(values[c] == (2. if enabled else 7.) for c in same.COLUMNS)
    assert signature == ((1, 3), 'xc', 9) if enabled else signature == (1, 3)
    assert reason == 'available'
    live.flush_events()
    rows = [json.loads(line) for line in live.audit.getvalue().splitlines()]
    assert rows[0]['xc_source'] == ('same_time' if enabled else 'result')
    assert rows[1]['serial'] == rows[2]['value']['serial'] == 9
    assert ('xc' in rows[2]['value']) == enabled
    if enabled:
        assert rows[2]['value']['xc'] == rows[1]['xc']
        assert live.selection_audit['xc_serial'] == 9
    else:
        assert 'xc_source' not in live.selection_audit


@pytest.mark.parametrize('stable', [False, True])
@pytest.mark.parametrize('retr', [False, True])
def test_observe_tracks_forecast_once(stable: bool, retr: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_ex_engine import LiveExEngine
    from src.phase_j._live_ex_frozen.src import dl1k_retr
    calls = []
    tracker = SimpleNamespace(observe=lambda *args: calls.append(args))
    monkeypatch.setattr(dl1k_retr, 'fire_tracker', lambda _: tracker)
    monkeypatch.setenv('PUYO_EX_RETR', str(int(retr)))
    native = pytest.importorskip('puyo_core')
    wrapper = LiveExEngine({}, native, stable_key=stable, xc_same_time=True)
    engine = wrapper.engine
    monkeypatch.setattr(engine, 'observe', lambda *args: None)
    monkeypatch.setattr(engine, 'probe', lambda: None)
    monkeypatch.setattr(wrapper.monitor, 'observe', lambda *args: None)
    try:
        wrapper.observe(dict(t=1., game=1, raw=[{}, {}]))
        assert len(calls) == 1
    finally:
        wrapper.close()


@pytest.mark.parametrize('ex_changes', [False, True])
def test_xc_arriving_during_inference_preserves_other_columns(ex_changes: bool) -> None:
    from src.phase_j.live_24h import Live24h
    live = object.__new__(Live24h)
    live.ama, live.revision, live.settings = True, 0, dict(xc_same_time=True)
    live.ex_columns = (*LEARNING_COLUMNS, *RETR, *same.COLUMNS)
    values = dict.fromkeys(live.ex_columns, 7.)
    selections = iter([(values, (1, 2), 'available'),
                       (dict(values), ((1, 3 if ex_changes else 2), 'xc', 9), 'available')])
    live.selection = lambda _: next(selections)
    live.selection_audit, live.clock, live.audit = {}, lambda: 5., StringIO()
    live.flush_events = lambda: None
    calls = []
    live.infer = lambda row, vals: calls.append(dict(vals)) or .5
    live.predict(dict(frame=1, t_sec=2., row={}))
    assert len(calls) == 2
    assert all(math.isnan(calls[-1][c]) for c in same.COLUMNS)
    for column in (*LEARNING_COLUMNS, *RETR):
        assert math.isnan(calls[-1][column]) if ex_changes else calls[-1][column] == 7.
    assert json.loads(live.audit.getvalue())['xc_source'] == 'none'


@pytest.mark.parametrize('row_time', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_constructor_history_is_opt_in(row_time: bool, enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_ex_delay import DelayedAsyncEx
    monkeypatch.setattr(async_ex.AsyncEx, '__init__', lambda *args: None)
    worker = DelayedAsyncEx(dict(logical_delay_ms=200., row_time=row_time, xc_same_time=enabled))
    assert (worker.xc_history is not None) == enabled
    if row_time and enabled:
        assert worker.xc_history is worker.row_time
