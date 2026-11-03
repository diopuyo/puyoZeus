"""同じ通知列を評価待ちより先に送り、世代・順序・行時刻を維持する。"""
import argparse
from copy import deepcopy
from io import StringIO
import os
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.phase_j import live_24h_config as config
from src.phase_j import live_b3, live_bridge, live_ex_delay
from src.phase_j.live_24h import Live24h
from src.phase_j.live_ex_row_time import RowTimeEx
from tests.test_live_24h_delay import owner
from tests.test_live_24h_launcher import data

WAIT_SEC = .05


def subject(monkeypatch: pytest.MonkeyPatch, early: bool, row_time: bool,
            clock: list[float]) -> live_b3.LiveB3:
    worker = owner([])
    worker.inputs = Queue()
    worker.events = []
    worker.row_time = RowTimeEx() if row_time else None
    monkeypatch.setattr(live_ex_delay, 'perf_counter', lambda: clock[0])
    live = object.__new__(Live24h)
    live.worker, live.audit = worker, StringIO()
    live.settings = dict(early_submit=early, row_time=row_time)
    live.ama, live.epoch, live.game = True, None, None
    live.base_result, live.signature = None, None
    b3 = object.__new__(live_b3.LiveB3)
    b3.live24h, b3.epoch, b3.generation, b3.adopted_game = live, None, 0, None
    b3.physical, b3.table, b3.fast_ema = set(), {}, False
    b3.display = display()
    monkeypatch.setattr(b3, '_new_display', display)
    monkeypatch.setattr(live_b3, 'RealtimeRows', lambda table: None)
    b3.async_engine = object()
    b3._async_update = lambda *args: None
    b3._display = lambda *args: None
    return b3


def display() -> SimpleNamespace:
    value = SimpleNamespace(game=None, chain=SimpleNamespace(observe=lambda frame: None))
    value.boundary = lambda game: setattr(value, 'game', game)
    return value


def notices() -> list[SimpleNamespace]:
    # 同じ試合、採用開始、試合切替、入力世代切替、採用解除を含む。
    states = [(0, 1, None), (0, 1, 1), (0, 1, 1), (0, 2, 2),
              (1, 2, 2), (1, 2, None), (1, 3, 3)]
    return [SimpleNamespace(b3=(dict(frame=i, t_sec=float(i), game_idx=game,
             adopted_game=adopted, _ex=dict(t=float(i), game=game,
             raw=[dict(side='p1', board=[i]), dict(side='p2', board=[i+1])])),
             (), (), epoch), captured_at=float(i), frame=i, t_sec=float(i),
             recognized_at=float(i)) for i, (epoch, game, adopted) in enumerate(states)]


def run(monkeypatch: pytest.MonkeyPatch, early: bool, row_time: bool) -> tuple[list, list]:
    clock = [0.]
    b3 = subject(monkeypatch, early, row_time, clock)
    bridge = object.__new__(live_bridge.RecognitionBridge)
    bridge.b3, bridge.meter, bridge.profile, bridge.batch_ready_at = b3, None, [], 0.
    sent, original = [], []
    for notice in notices():
        clock[0] = notice.captured_at
        original.append(deepcopy(notice.b3))
        iterator = bridge.timed_notice(notice)
        assert next(iterator) is notice
        assert b3.live24h.worker.inputs.qsize() == int(early)
        clock[0] += WAIT_SEC
        b3.update(notice, None)
        assert b3.live24h.worker.inputs.qsize() == 1
        sent.append(b3.live24h.worker.inputs.get_nowait())
        assert notice.b3 == original[-1]
        assert list(iterator) == []
    return sent, original


@pytest.mark.parametrize('row_time', [False, True])
def test_early_requests_preserve_packets_order_epochs(monkeypatch: pytest.MonkeyPatch,
                                                     row_time: bool) -> None:
    normal, original = run(monkeypatch, False, row_time)
    early, same = run(monkeypatch, True, row_time)
    assert original == same
    assert [r['epoch'] for r in early] == [1, 2, 2, 3, 5, 5, 6]
    for before, after in zip(normal, early, strict=True):
        assert before.pop('requested_at') - after.pop('requested_at') == pytest.approx(WAIT_SEC)
        assert before == after


def test_double_early_submit_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    b3 = subject(monkeypatch, True, True, [0.])
    notice = notices()[0]
    b3.submit_ex_early(notice)
    with pytest.raises(RuntimeError, match='未処理'):
        b3.submit_ex_early(notice)
    assert b3.live24h.worker.inputs.qsize() == 1


def test_mismatched_generation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    b3 = subject(monkeypatch, True, False, [0.])
    notice = notices()[0]
    b3.submit_ex_early(notice)
    b3.generation += 1
    with pytest.raises(RuntimeError, match='世代が不一致'):
        b3.update(notice, None)
    assert b3.live24h.worker.inputs.qsize() == 1


@pytest.mark.parametrize('ama', [False, True])
def test_disabled_flag_has_no_early_effect(monkeypatch: pytest.MonkeyPatch, ama: bool) -> None:
    b3 = subject(monkeypatch, not ama, False, [0.])
    b3.live24h.ama = ama
    b3.submit_ex_early(notices()[0])
    assert b3.live24h.worker.inputs.empty()
    assert b3.generation == 0 and b3.live24h.epoch is None


def test_launcher_to_settings_and_off_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_ex_runtime
    from src.phase_j.launcher import build_pipeline_config, parse_user_config, USER_KEYS
    with patch.dict(os.environ):
        user = parse_user_config(data(tmp_path, live_24h=True, live_24h_model='ex.json',
            live_24h_early_submit=True, live_24h_row_time=True), tmp_path)
        pipeline = build_pipeline_config(user)
        assert 'live_24h_early_submit' in USER_KEYS and pipeline['live_24h_early_submit']
        config.configure(SimpleNamespace(**pipeline), argparse.ArgumentParser())
        monkeypatch.setattr(live_ex_runtime, 'settings_from_env', lambda defaults: defaults)
        settings = config.ex_settings()
        assert settings['early_submit'] and settings['row_time']
        config.configure(SimpleNamespace(live_24h=False), argparse.ArgumentParser())
        assert os.environ[config.ENV_EARLY_SUBMIT] == '0'
        assert not config.ex_settings()['early_submit']


@pytest.mark.parametrize('bad', [1, 'true', None])
def test_bad_json_flag_is_rejected(tmp_path: Path, bad: object) -> None:
    with pytest.raises(ValueError, match='early_submitはbool'):
        config.launcher_values(data(tmp_path, live_24h=True, live_24h_model='ex.json',
                                   live_24h_early_submit=bad), tmp_path)


def test_cli_flag_defaults_off_and_can_enable() -> None:
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    assert not parser.parse_args([]).live_24h_early_submit
    assert parser.parse_args(['--live-24h-early-submit']).live_24h_early_submit
    assert not parser.parse_args(['--no-live-24h-early-submit']).live_24h_early_submit
