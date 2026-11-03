"""採用世代だけの切替で、到着済みの発火初期情報を失わない回帰検査。"""
from pathlib import Path
from copy import deepcopy
import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.phase_j.live_v49f_async import worker

INITIAL = dict(id=4, order=4, t=1., kind='initial', data=dict(input_sec=1., sources=[]))
SCORE = dict(id=4, order=4, t=2., kind='score', data=dict(t=2., score=40))


class Connection:
    def __init__(self, packets: list[dict]) -> None:
        self.packets = iter([*packets, None])
        self.results: list[dict] = []

    def recv(self) -> dict | None:
        return next(self.packets)

    def send(self, value: dict) -> None:
        self.results.append(value)

    def close(self) -> None:
        pass


class Engine:
    def __init__(self) -> None:
        self.base = SimpleNamespace(policy={}, native=None)
        self.closed = False

    def observe(self, *args: Any) -> tuple[None, None]:
        return None, None

    def close(self) -> None:
        self.closed = True


def execute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
            packets: list[dict]) -> tuple[list[dict], list[Engine]]:
    from src import prefire_origin_stream as origins
    from src.phase_j import live_v49f_engine
    engines: list[Engine] = []

    def load(*args: Any) -> Engine:
        engine = Engine()
        engines.append(engine)
        return engine

    monkeypatch.setattr(live_v49f_engine, 'load_engine', load)
    monkeypatch.setattr(origins, 'initialize', lambda *a: dict(after=[]))
    monkeypatch.setattr(origins, 'snapshot', lambda state, data, *a: dict(input_sec=data['input_sec']))
    monkeypatch.setattr(origins, 'on_score', lambda *a: None)
    monkeypatch.setattr(origins, 'on_formula', lambda *a: None)
    monkeypatch.setattr(origins, 'terminal_observation', lambda *a: None)
    connection = Connection(packets)
    worker(connection, (), tmp_path)
    return connection.results, engines


def packet(message: dict, generation: int, input_epoch: tuple = (0, True)) -> dict:
    return dict(id=generation, generation=generation, input_epoch=input_epoch,
                game=0, t=message['t'], raw=[], origins=[message], submitted_at=0.)


def test_adoption_change_preserves_origin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    results, engines = execute(monkeypatch, tmp_path, [packet(INITIAL, 1), packet(SCORE, 2)])
    assert len(results) == 3 and not any('fatal' in r for r in results)
    assert len(engines) == 1 and engines[0].closed


def test_input_epoch_change_reinitializes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    results, engines = execute(monkeypatch, tmp_path,
        [packet(INITIAL, 1), packet(INITIAL, 2, (1, True))])
    assert not any('fatal' in r for r in results)
    assert len(engines) == 2 and all(e.closed for e in engines)


def test_real_missing_initial_is_not_silently_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    results, _ = execute(monkeypatch, tmp_path,
        [packet(INITIAL, 1), packet(SCORE, 2, (1, True))])
    assert results[-1]['fatal'] == 'KeyError: 4'


def test_legacy_packets_keep_generation_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    packets = [packet(INITIAL, 1), packet(INITIAL, 2)]
    for item in packets:
        item.pop('input_epoch')
    results, engines = execute(monkeypatch, tmp_path, packets)
    assert not any('fatal' in r for r in results) and len(engines) == 2


@pytest.mark.parametrize('index', [0, 1])
@pytest.mark.parametrize('legacy', [True, False])
def test_recorded_origins_at_observed_adoption_boundary(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, index: int, legacy: bool) -> None:
    """実origin payloadと旧監査の境界を組合せ、4/6の欠落を決定的に再現する。"""
    path = Path(__file__).with_name('fixtures')/'v49f_adoption_boundary.json'
    case = json.loads(path.read_text(encoding='utf-8'))['cases'][index]
    packets = deepcopy(case['packets'])
    if legacy:
        for value in packets:
            value.pop('input_epoch')
    results, _ = execute(monkeypatch, tmp_path, packets)
    if legacy:
        assert results[-1]['fatal'] == f"KeyError: {case['key']}"
    else:
        assert not any('fatal' in r for r in results) and len(results) == 3
