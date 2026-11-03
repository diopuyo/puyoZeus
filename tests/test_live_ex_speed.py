"""候補順序・境界・既定OFFを照合する。"""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from threading import Event

import numpy as np
import pytest

from src.phase_j import live_24h_config as config
from src.phase_j.live_ex_speed import matching_positions


@pytest.mark.parametrize('size', [0, 1, 4, 5, 6, 20])
def test_positions(size: int) -> None:
    rng = np.random.default_rng(42)
    for length in (0, 3, 6, 100, 1000):
        tape = rng.integers(0, 4, length)
        codes = tape[:size] if size <= length else rng.integers(0, 4, size)
        expected = [q for q in range(len(tape)-len(codes)+1)
                    if (tape[q:q+len(codes)] == codes).all()]
        assert matching_positions(tape, codes) == expected


def test_repeated_candidates() -> None:
    tape, codes = np.ones(10), np.ones(4)
    candidates = matching_positions(tape, codes)
    assert candidates == list(range(7))
    assert min(candidates, key=lambda q: abs(q-2.5)) == 2


def test_cli_default_and_on() -> None:
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    assert parser.parse_args([]).live_24h_ex_speed is False
    assert parser.parse_args(['--live-24h-ex-speed']).live_24h_ex_speed is True


@pytest.mark.parametrize('value', [1, 'true', None])
def test_invalid_flag(value: object) -> None:
    with pytest.raises(ValueError):
        config.validate_ex_speed(value)


def test_launcher_to_child_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_ex_runtime
    from src.phase_j.launcher import build_pipeline_config, parse_user_config, USER_KEYS
    from tests.test_live_24h_launcher import data
    with patch.dict(os.environ):
        user = parse_user_config(data(tmp_path, live_24h=True, live_24h_model='ex.json',
                                      live_24h_ex_speed=True), tmp_path)
        pipeline = build_pipeline_config(user)
        assert 'live_24h_ex_speed' in USER_KEYS and pipeline['live_24h_ex_speed']
        config.configure(SimpleNamespace(**pipeline), argparse.ArgumentParser())
        monkeypatch.setattr(live_ex_runtime, 'settings_from_env', lambda defaults: defaults)
        assert config.ex_settings()['ex_speed']
        config.configure(SimpleNamespace(live_24h=False), argparse.ArgumentParser())
        assert os.environ[config.ENV_EX_SPEED] == '0'
        assert not config.ex_settings()['ex_speed']


def test_snapshot_separates_mutable_state() -> None:
    from src.phase_j.live_ex_speed import snapshot
    lookup = SimpleNamespace(table=object(), keys=np.arange(3), order=np.arange(3), last={})
    engine = SimpleNamespace(native=object(), knowledge=SimpleNamespace(lookup=lookup),
                             state={'board': [1]}, chains=[{'finished': False}], retr_pre={})
    clone, value = snapshot(engine, {'board': engine.state['board']})
    clone.state['board'][0] = 2
    clone.knowledge.lookup.last[0] = 'new'
    assert engine.state['board'] == [1] and lookup.last == {}
    assert clone.knowledge.lookup.table is lookup.table
    assert clone.knowledge.lookup.keys is lookup.keys
    assert value['board'] == [2]


@pytest.mark.parametrize('fail_search', [False, True])
def test_parallel_preserves_before_proof_and_joins(monkeypatch: pytest.MonkeyPatch,
                                                  fail_search: bool) -> None:
    from src.phase_j import live_ex_compute as calc
    from src.phase_j.live_ex_speed import parallel
    reached = Event()
    engine = SimpleNamespace(native=object(), knowledge=SimpleNamespace(lookup=None),
                             state={'board': [1]}, chains=[], retr_pre={})
    def retr(clone: object, value: dict, result: dict) -> None:
        assert reached.wait(timeout=5)
        result['retr'] = clone.state['board'][0]
        clone.retr_pre[0] = 'saved'
    def proof(original: object, value: dict) -> dict:
        original.state['board'][0] = 2
        value['death_sides'] = [0]
        reached.set()
        return {'ok': True}
    def search(original: object, value: dict) -> tuple:
        assert value['death_sides'] == [0]
        if fail_search:
            raise ValueError('search failed')
        return {}, None, None
    monkeypatch.setattr(calc, 'add_retr', retr)
    monkeypatch.setattr(calc, 'retr_details', lambda *args: None)
    monkeypatch.setattr(calc, 'xc_enabled', lambda: False)
    monkeypatch.setattr(calc, 'proof_of', proof)
    monkeypatch.setattr(calc, 'search_of', search)
    result, times = {}, {}
    if fail_search:
        with pytest.raises(ValueError, match='search failed'):
            parallel(engine, {}, result, times)
    else:
        assert parallel(engine, {}, result, times)
    assert result['retr'] == 1 and engine.state['board'] == [2]
    assert engine.retr_pre == {0: 'saved'}


def test_copy_failure_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_ex_speed as speed
    def fail(*args: object) -> None:
        raise TypeError('cannot copy')
    monkeypatch.setattr(speed, 'snapshot', fail)
    result = {}
    assert speed.parallel(object(), {}, result, {}) is False
    assert result['ex_speed']['parallel'] is False
