"""空き計算子への発火優先、観測列、鍵の分離と設定の試験。"""
from __future__ import annotations

import argparse
import os
import struct
from collections import deque
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.phase_j import live_24h_config as config
from src.phase_j import live_ex_async as async_ex
from src.phase_j.live_ex_fire_first import FireFirstState, close_workers
from src.phase_j.live_ex_row_time import RowTimeEx


class Connection:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.results: deque[dict] = deque()

    def send(self, value: dict) -> None:
        self.sent.append(value)

    def poll(self) -> bool:
        return bool(self.results)

    def recv(self) -> dict:
        return self.results.popleft()


def state(monkeypatch: pytest.MonkeyPatch) -> FireFirstState:
    engine = SimpleNamespace(observe=lambda p: SimpleNamespace(key=p['key'], fight=p['fight']),
                             close=lambda: None)
    monkeypatch.setattr(async_ex, 'create_engine', lambda *_: engine)
    owner = FireFirstState({}, [Connection(), Connection()], Queue())
    monkeypatch.setattr(owner, 'sync_standby', lambda: None)
    return owner


def submit(owner: FireFirstState, serial: int, key: int, epoch: int = 1,
           fight: bool = True) -> None:
    owner.observe(dict(packet=dict(t=float(serial), key=(epoch, key), fight=fight),
                       serial=serial, epoch=epoch, input_t=float(serial), captured_at=float(serial)))
    owner.pump()


def test_new_fight_bypasses_running_old_key(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = state(monkeypatch)
    submit(owner, 1, 1)
    submit(owner, 2, 2)
    assert [c.sent[0]['serial'] for c in owner.connections] == [1, 2]
    assert all(owner.running)
    assert owner.counts['fire_bypass'] == 1


def test_same_key_waits_and_keeps_every_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = state(monkeypatch)
    submit(owner, 1, 1)
    submit(owner, 2, 1)
    submit(owner, 3, 1)
    assert not owner.connections[1].sent
    owner.connections[0].results.append(dict(fight=True))
    owner.pump()
    assert [p['t'] for p in owner.connections[0].sent[-1]['packets']] == [2., 3.]
    assert owner.connections[0].sent[-1]['serial'] == 3


def test_both_busy_keeps_latest_then_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = state(monkeypatch)
    for serial in range(1, 5):
        submit(owner, serial, serial)
    assert owner.pending['serial'] == 4
    owner.connections[0].results.append(dict(fight=True))
    owner.pump()
    assert owner.connections[0].sent[-1]['serial'] == 4
    assert [p['t'] for p in owner.connections[0].sent[-1]['packets']] == [2., 3., 4.]


def test_epoch_clears_only_unsent_history(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = state(monkeypatch)
    submit(owner, 1, 1)
    submit(owner, 2, 1, epoch=2)
    assert [p['t'] for p in owner.connections[1].sent[-1]['packets']] == [2.]
    assert owner.connections[1].sent[-1]['epoch'] == 2


def test_calculate_receives_identical_history_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = state(monkeypatch)
    submit(owner, 1, 1)
    submit(owner, 2, 2)
    request = owner.connections[1].sent[0]
    seen: list[float] = []
    engine = SimpleNamespace(observe=lambda p: seen.append(p['t']), close=lambda: None,
                             calculate=lambda: dict(key=request['key'], fight=True,
                                                    extra=dict(value=sum(seen))))
    monkeypatch.setattr(async_ex, 'create_engine', lambda *_: engine)
    connection = Connection()
    connection.results.extend([request, None])
    connection.close = lambda: None
    async_ex._compute(connection, {})
    expected = struct.pack('!d', 1. + 2.)
    assert struct.pack('!d', connection.sent[0]['extra']['value']) == expected
    assert seen == [1., 2.]


def test_old_key_result_cannot_be_selected() -> None:
    owner = object.__new__(async_ex.AsyncEx)
    owner.outputs, owner.events = Queue(), []
    owner.serial, owner.epoch, owner.error = 2, 1, None
    owner.key = dict(key=(1, 2), serial=2, fight=True)
    owner.latest = dict(epoch=1, key=(1, 1), serial=1, fight=True)
    assert owner.available(2., 2.) == (None, 'key_mismatch')


def test_row_time_rejects_late_old_result() -> None:
    owner = RowTimeEx()
    owner.boundary(1)
    owner.observe(dict(epoch=1, key=(1, 2), input_t=2., fight=True))
    new = dict(epoch=1, key=(1, 2), serial=2, input_t=2., fight=True,
               received_at=2.1, eligible_at=2.2)
    old = dict(new, key=(1, 1), serial=1, input_t=1., received_at=2.3)
    assert owner.select(deque([new, old]), 2.4, 2.4, 2.4, None)[0] is new
    assert owner.select(deque([old]), 2.4, 2.4, 2.4, None)[0] is None


def test_standby_observes_without_publishing_result(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = state(monkeypatch)
    submit(owner, 1, 1, fight=False)
    FireFirstState.sync_standby(owner)
    sync = owner.connections[1].sent[-1]
    assert sync['sync_only'] and [p['t'] for p in sync['packets']] == [1.]
    owner.connections[1].results.append(dict(sync_only=True, fight=False))
    before = owner.outputs.qsize()
    owner.receive()
    assert owner.outputs.qsize() == before
    assert owner.running[1] is None
    submit(owner, 2, 2)
    assert owner.connections[1].sent[-1]['serial'] == 2
    assert not owner.connections[1].sent[-1].get('sync_only')
    assert [p['t'] for p in owner.connections[1].sent[-1]['packets']] == [2.]


def test_sync_request_does_not_calculate(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[float] = []
    probe = SimpleNamespace(key=(1, 1), fight=True)
    engine = SimpleNamespace(observe=lambda p: (seen.append(p['t']) or probe), close=lambda: None)
    monkeypatch.setattr(async_ex, 'create_engine', lambda *_: engine)
    connection = Connection()
    connection.close = lambda: None
    connection.results.extend([dict(epoch=1, key=probe.key, fight=True, sync_only=True,
                                    packets=[dict(t=1.)]), None])
    async_ex._compute(connection, {})
    assert seen == [1.]
    assert connection.sent[0]['sync_only']
    assert 'extra' not in connection.sent[0]


def test_close_drains_all_results_before_sending_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = state(monkeypatch)
    submit(owner, 1, 1)
    submit(owner, 2, 2)
    workers = []
    for connection in owner.connections:
        connection.results.append(dict(fight=True))
        worker = SimpleNamespace(alive=True)
        worker.is_alive = lambda w=worker: w.alive
        worker.join = lambda timeout: None
        worker.terminate = lambda: pytest.fail('停止前に受信すれば強制終了は不要')
        workers.append(worker)
        def stop(value: None, w: SimpleNamespace = worker) -> None:
            assert value is None
            assert all(not c.results for c in owner.connections)
            w.alive = False
        connection.send = stop
        connection.close = lambda: None
    close_workers(owner, workers)
    assert all(not w.alive for w in workers)


def test_cli_and_default() -> None:
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    assert parser.parse_args([]).live_24h_ex_fire_first is False
    assert parser.parse_args(['--live-24h-ex-fire-first']).live_24h_ex_fire_first is True
    assert parser.parse_args(['--no-live-24h-ex-fire-first']).live_24h_ex_fire_first is False


@pytest.mark.parametrize('value', [1, 'true', None])
def test_invalid_flag(value: object) -> None:
    with pytest.raises(ValueError):
        config.validate_ex_fire_first(value)


def test_launcher_to_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_ex_runtime
    from src.phase_j.launcher import build_pipeline_config, parse_user_config, USER_KEYS
    from tests.test_live_24h_launcher import data
    with patch.dict(os.environ):
        user = parse_user_config(data(tmp_path, live_24h=True, live_24h_model='ex.json',
                                      live_24h_ex_fire_first=True), tmp_path)
        pipeline = build_pipeline_config(user)
        assert 'live_24h_ex_fire_first' in USER_KEYS and pipeline['live_24h_ex_fire_first']
        config.configure(SimpleNamespace(**pipeline), argparse.ArgumentParser())
        monkeypatch.setattr(live_ex_runtime, 'settings_from_env', lambda defaults: defaults)
        assert config.ex_settings()['ex_fire_first']
        config.configure(SimpleNamespace(live_24h=False), argparse.ArgumentParser())
        assert os.environ[config.ENV_EX_FIRE_FIRST] == '0'
        assert not config.ex_settings()['ex_fire_first']
