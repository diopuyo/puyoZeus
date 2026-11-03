"""先行する観測があっても、行時刻の鍵・解除・保持を再現する。"""
import argparse
from queue import Queue
from types import SimpleNamespace
from pathlib import Path
from io import StringIO
import json
import os
from collections import deque
from copy import deepcopy

import pytest

from src.phase_j import live_24h_config as config
from src.phase_j import live_ex_async as async_ex
from src.phase_j import live_ex_delay as delay
from src.phase_j import live_ex_row_time as history
from src.phase_j.live_ex_hold import LiveExHold
from tests.test_live_24h_delay import completed, owner
from tests.test_live_24h_hold import source


def key(t: float = 3., serial: int = 3, **changes: object) -> dict:
    return dict(dict(kind='key', epoch=4, input_t=t, serial=serial, fight=True,
                     key=(4, 2, (3, None)), resets=[]), **changes)


def subject(monkeypatch: pytest.MonkeyPatch) -> delay.DelayedAsyncEx:
    monkeypatch.setattr(delay, 'perf_counter', lambda: 100.)
    monkeypatch.setattr(async_ex, 'perf_counter', lambda: 100.)
    obj = owner([completed(extra=source()['extra'], received_at=10.1)])
    obj.hold = LiveExHold()
    obj.row_time = history.RowTimeEx()
    obj.latest_input_t = 3.
    obj.inputs = Queue()
    obj.protect_rows(3., 4)
    obj.outputs.put(key())
    return obj


def test_future_submit_key_reset_and_prune_leave_previous_row(monkeypatch: pytest.MonkeyPatch) -> None:
    obj = subject(monkeypatch)
    before = obj.held_available(100., 3.1, 10.3)
    obj.submit(dict(t=20.), 4, 100.)
    obj.outputs.put(key(20., obj.serial, fight=False, resets=[dict(side='p1')]))
    obj.poll()
    after = obj.held_available(101., 3.1, 10.3)
    assert before == after and after[0] is not None
    assert len(obj.completed) == 1
    assert obj.held_available(101., 20., 100.)[0] is None


def test_later_reset_does_not_clear_held_previous_row(monkeypatch: pytest.MonkeyPatch) -> None:
    obj = subject(monkeypatch)
    assert obj.held_available(100., 3.1, 10.3)[0] is not None
    before = obj.held_available(100., 3.55, 10.8)
    assert before[1] == 'held'
    obj.outputs.put(key(3.7, 4, resets=[dict(side='p1'), dict(side='p2')]))
    obj.poll()
    assert obj.held_available(102., 3.55, 10.8) == before
    assert obj.held_available(102., 3.7, 11.)[0] is None


@pytest.mark.parametrize(('now', 'cutoff', 'present'), [
    (10.09, 10.3, False), (10.2, 10.199, False), (10.2, 10.2, True)])
def test_receipt_and_eligibility(monkeypatch: pytest.MonkeyPatch, now: float,
                                 cutoff: float, present: bool) -> None:
    obj = subject(monkeypatch)
    assert (obj.available(now, 3.1, cutoff)[0] is not None) == present


def test_new_key_for_same_row_reselects(monkeypatch: pytest.MonkeyPatch) -> None:
    obj = subject(monkeypatch)
    obj.outputs.get_nowait()
    assert obj.held_available(100., 3.1, 10.3)[0] is None
    obj.outputs.put(key())
    assert obj.held_available(100., 3.1, 10.3)[0] is not None


def test_prune_boundary_and_history_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    obj = subject(monkeypatch)
    obj.poll()
    obj.outputs.put(key(4., 4))
    obj.protect_rows(3.5, 4)
    obj.poll()
    assert len(obj.completed) == 1
    assert obj.row_time.key_at(3.5)['serial'] == 3
    obj.protect_rows(3.500001, 4)
    obj.poll()
    assert len(obj.completed) == 0 and obj.row_time.pruned == 1


@pytest.mark.parametrize('bound', ['MAX_KEYS', 'MAX_RESULTS', 'MAX_ROWS'])
def test_zero_bound_discards_without_error(monkeypatch: pytest.MonkeyPatch, bound: str) -> None:
    obj = subject(monkeypatch)
    obj.held_available(100., 3.1, 10.3)
    monkeypatch.setattr(history, bound, 0)
    if bound == 'MAX_KEYS':
        obj.row_time.observe(key(4., 4))
    else:
        obj.poll()
    count = 2 if bound == 'MAX_KEYS' else 1
    field = 'overflow_' + bound.removeprefix('MAX_').lower()
    assert obj.selection_record(3.1, None)[field] == count


def test_epoch_discards_old_history(monkeypatch: pytest.MonkeyPatch) -> None:
    obj = subject(monkeypatch)
    obj.held_available(100., 3.1, 10.3)
    obj.submit(dict(t=0.), 5, 101.)
    obj.outputs.put(key(4., 4))
    assert obj.held_available(101., 0., 101.)[0] is None
    assert not obj.row_time.keys and not obj.completed


@pytest.mark.parametrize('kind', ['keys', 'results', 'rows'])
def test_overflow_keeps_latest_selection_and_hold(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, kind: str,
) -> None:
    obj = subject(monkeypatch)
    value, _ = obj.held_available(100., 3.1, 10.3)
    limited = obj.row_time
    limited.floor = -float('inf')
    baseline = deepcopy(limited)
    rows = deque(completed(serial=i, input_t=t, received_at=10.1,
                           extra=source()['extra'])
                 for i, t in [(2, 3.2), (1, 3.), (3, 3.1)])
    if kind == 'keys':
        for t in (3.2, 3.3, 3.4):
            baseline.observe(key(t))
        monkeypatch.setattr(history, 'MAX_KEYS', 2)
        for t in (3.2, 3.3, 3.4):
            limited.observe(key(t))
        assert [k['input_t'] for k in limited.keys] == [3.3, 3.4]
        assert limited.floor == -float('inf')
    elif kind == 'results':
        monkeypatch.setattr(history, 'MAX_RESULTS', 2)
        assert [r['input_t'] for r in limited.prune(rows)] == [3.2, 3.1]
    else:
        for t in (3.2, 3.3, 3.4):
            baseline.held(100., t, 10.3, None, 'not_ready_or_stale')
        monkeypatch.setattr(history, 'MAX_ROWS', 2)
        for t in (3.2, 3.3, 3.4):
            limited.held(100., t, 10.3, None, 'not_ready_or_stale')
        assert sorted(i[1] for i in limited.rows) == [3.3, 3.4]

    def warnings() -> int:
        return len([r for r in caplog.records if r.name == history.__name__])

    assert warnings() == 1
    chosen = baseline.select(rows, 100., 3.4, 10.3, None)
    assert limited.select(limited.prune(rows), 100., 3.4, 10.3, None) == chosen
    assert limited.held(100., 3.4, 10.3, *chosen) == baseline.held(100., 3.4, 10.3, *chosen)
    assert limited.held(100., 3.45, 10.3, None, 'not_ready_or_stale') == baseline.held(
        100., 3.45, 10.3, None, 'not_ready_or_stale')
    count = dict(keys=2, results=2, rows=3)[kind]
    assert obj.selection_record(3.45, value)['overflow_' + kind] == count
    assert obj.selection_record(3.45, value)['overflow_' + kind] == count
    # 警告はインスタンスごとに 1 回。rows では上限を差し替えた後の baseline の held も上限を超える。
    assert warnings() == (2 if kind == 'rows' else 1)


def test_audit_contains_row_key_and_pruned_count(monkeypatch: pytest.MonkeyPatch) -> None:
    obj = subject(monkeypatch)
    value, _ = obj.held_available(100., 3.1, 10.3)
    obj.latest_input_t = 20.
    obj.outputs.put(key(20., 4, fight=False))
    value, _ = obj.held_available(101., 3.1, 10.3)
    record = obj.selection_record(3.1, value)
    assert record['key_serial'] == 3 and record['key_fight']
    assert record['latest_input_t'] == 20. and record['candidates'] == 1
    assert record['result_serial'] == 1 and record['result_input_t'] == 3.


def test_flag_defaults_off_and_cli() -> None:
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    assert not parser.parse_args([]).live_24h_row_time
    assert parser.parse_args(['--live-24h-row-time']).live_24h_row_time


@pytest.mark.parametrize('value', ['1', 1, None])
def test_invalid_flag(value: object) -> None:
    with pytest.raises(ValueError, match='bool'):
        config.validate_row_time(value)


def test_environment_and_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_ex_runtime
    monkeypatch.setattr(live_ex_runtime, 'settings_from_env', lambda defaults: {})
    parser = argparse.ArgumentParser()
    options = SimpleNamespace(live_24h_row_time=True, live_24h=True,
                              live_24h_ama=False, live_24h_delay_ms=200.)
    monkeypatch.setenv(config.ENV_ROW_TIME, '0')
    config.configure_row_time(options, parser, True)
    assert config.pipeline_values(options)['live_24h_row_time'] is True
    monkeypatch.setenv(config.ENV_ENABLED, '1')
    monkeypatch.setenv(config.ENV_AMA, '1')
    assert config.ex_settings()['row_time'] is True
    config.configure_row_time(options, parser, False)
    assert config.ex_settings()['row_time'] is False


def test_launcher_to_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_ex_runtime
    from src.phase_j.launcher import build_pipeline_config, parse_user_config, USER_KEYS
    from tests.test_live_24h_launcher import data
    for name in (*[v for k, v in vars(config).items() if k.startswith('ENV_')],
                 'PUYO_EX_WORK_BUDGET', 'PUYO_EX_WORK_LIMIT'):
        monkeypatch.setenv(name, os.environ.get(name, ''))
    user = parse_user_config(data(tmp_path, live_24h=True, live_24h_row_time=True,
                                  live_24h_model='ex.json'), tmp_path)
    pipeline = build_pipeline_config(user)
    assert 'live_24h_row_time' in USER_KEYS and pipeline['live_24h_row_time'] is True
    config.configure(SimpleNamespace(**pipeline), argparse.ArgumentParser())
    monkeypatch.setattr(live_ex_runtime, 'settings_from_env', lambda defaults: defaults)
    assert config.ex_settings()['row_time'] is True


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('row_time', [False, True])
def test_selection_audit_is_opt_in(monkeypatch: pytest.MonkeyPatch, enabled: bool,
                                   row_time: bool) -> None:
    from src.phase_j.live_24h import Live24h
    obj = subject(monkeypatch)
    obj.row_time.overflow_keys = 7
    if not row_time:
        obj.row_time = None
        obj.media_fresh = True
        obj.hold = LiveExHold(media_fresh=True)
    live = object.__new__(Live24h)
    live.worker, live.clock, live.ex_columns = obj, lambda: 100., ()
    live.audit = StringIO()
    monkeypatch.setenv('PUYO_LIVE_EX_AUDIT', '1' if enabled else '0')
    live.selection(dict(frame=93, t_sec=3.1, captured_at=10.3))
    lines = live.audit.getvalue().splitlines()
    assert len(lines) == int(enabled)
    if enabled:
        record = json.loads(lines[0])
        assert record['kind'] == 'ex_selection' and record['frame'] == 93
        assert record['row_time'] == row_time and record['selected_at'] == 100.
        if row_time:
            assert record['overflow_keys'] == 7
            assert record['overflow_results'] == record['overflow_rows'] == 0
        else:
            assert 'overflow_keys' not in record


def test_live_update_protects_base_and_pending_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_24h import Live24h
    live = object.__new__(Live24h)
    live.worker = subject(monkeypatch)
    live.settings = dict(row_time=True)
    live.epoch, live.game, live.base_result = 4, 2, dict(t_sec=3.1)
    live.flush_events = lambda: None
    live.update(dict(t_sec=20., game_idx=2, _ex=dict(t=20.)), 4, 100.)
    assert live.worker.row_time.floor == 3.1
    live.base_result = dict(t_sec=19.)
    live.update(dict(t_sec=20., game_idx=2, _ex=dict(t=20.)), 4, 100.)
    assert live.worker.row_time.floor == 18.


def test_history_is_pruned_during_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    obj = subject(monkeypatch)
    for index in range(100):
        t = 3. + index
        obj.protect_rows(t-2., 4)
        obj.outputs.put(key(t, index+4))
        obj.held_available(100.+index, t, 10.3+index)
    assert len(obj.row_time.keys) <= 4
    assert len(obj.row_time.rows) <= 4
    assert not obj.completed


@pytest.mark.parametrize('enabled', [False, True])
def test_constructor_flag(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setattr(async_ex.AsyncEx, '__init__', lambda self, settings: None)
    obj = delay.DelayedAsyncEx(dict(logical_delay_ms=200., row_time=enabled))
    assert (obj.row_time is not None) == enabled
