"""B11の長時間保持・監査記録を検証する。"""
from pathlib import Path
from contextlib import ExitStack
import gc
import json

import pytest

from src.phase_j.live_spool import DiskMap, DiskRows, DiskFIFO
from src.phase_j.live_cache import RecentCache, bounded_chain_caches
from src.chain import ChainSimulator
from src.board import Board
from src.phase_j.live_retention import archived_tracker, CounterRegistry, archive_recognition_alerts
from src.exchange_event_tracker import ExchangeEventTracker
from tests.test_exchange_event_tracker import Models, fire


def test_spool_preserves_late_row_fields_and_reiteration(tmp_path: Path) -> None:
    rows = DiskRows(tmp_path/'rows.bin')
    first = {'value': 1}
    rows.append(first)
    first['later'] = 2
    rows.append({'value': 3})
    assert rows[0] == {'value': 1, 'later': 2}
    assert rows[-1] == {'value': 3}
    assert list(rows) == list(rows) == [first, {'value': 3}]
    rows.close()


def test_spool_empty_and_bounds(tmp_path: Path) -> None:
    rows = DiskRows(tmp_path/'rows.bin')
    assert not rows and list(rows) == []
    with pytest.raises(IndexError):
        rows[-1]
    rows.extend(range(10))
    assert rows[-2:] == [8, 9]
    assert rows[3] == 3
    assert rows.pending == 9 and len(rows) == 10
    rows.close()


def test_sse_disk_index_deduplicates_without_memory_keys(tmp_path: Path) -> None:
    rows = DiskMap(tmp_path/'sent.sqlite')
    for i in range(100):
        rows.setdefault(i, {'value': i})
    rows.setdefault(2, {'value': -1})
    assert len(rows) == 100
    assert list(rows.values())[2] == {'value': 2}
    rows.close()


def test_recent_cache_keeps_new_boards_after_capacity() -> None:
    cache = RecentCache(2)
    cache['a'], cache['b'] = 1, 2
    assert cache.get('a') == 1
    cache['c'] = 3
    assert list(cache) == ['a', 'c']
    assert cache.get('b') is None and cache.evictions == 1


def test_simulation_cache_scoped_and_same_result() -> None:
    simulator = ChainSimulator()
    board = Board()
    expected = simulator.simulate(board)
    old_cache = simulator._cache
    with bounded_chain_caches():
        assert isinstance(simulator._cache, RecentCache)
        simulator.simulate(board)
        actual = simulator.simulate(board)
        assert simulator._cache.hits == 1
        assert actual.chain_count == expected.chain_count
        assert actual.final_board.grid_bytes() == expected.final_board.grid_bytes()
        assert isinstance(ChainSimulator()._cache, RecentCache)
    assert simulator._cache is old_cache
    assert type(ChainSimulator()._cache) is dict


def test_boundary_archives_records_and_preserves_ids_and_save(tmp_path: Path) -> None:
    reference = ExchangeEventTracker(Models())
    with ExitStack() as stack:
        candidate = archived_tracker(ExchangeEventTracker, tmp_path/'spool', stack)(Models())
        for game in range(4):
            for tracker in (reference, candidate):
                tracker.boundary(game, game*10)
                fire(tracker)
                tracker.missing_input('test', game*10+1, 'S3', game)
            assert len(candidate.records) == 1
            assert len(candidate._contexts) == 1
            assert candidate.current.exchange_id == reference.current.exchange_id
            assert candidate.probability == reference.probability
        reference.save(tmp_path/'off.jsonl')
        candidate.save(tmp_path/'on.jsonl')
        assert (tmp_path/'off.jsonl').read_bytes() == (tmp_path/'on.jsonl').read_bytes()
        assert json.loads((tmp_path/'off.diagnostics.json').read_text()) == json.loads(
            (tmp_path/'on.diagnostics.json').read_text())


def test_counter_registry_releases_unused_trackers() -> None:
    class Counter:
        pass
    registry = CounterRegistry()
    owner = Counter()
    registry.append(owner)
    for _ in range(100):
        other = Counter()
        registry.append(other)
    assert len(registry) == 2
    del other
    gc.collect()
    assert list(registry) == [owner]


@pytest.mark.parametrize('size', [1, 2, 16, 128])
def test_spool_clear_retains_sequence_contract(tmp_path: Path, size: int) -> None:
    rows = DiskRows(tmp_path/'rows.bin')
    rows.extend(range(size))
    rows.clear()
    rows.append({'after': size})
    assert len(rows) == 1 and list(rows) == [{'after': size}]
    rows.close()


@pytest.mark.parametrize('capacity', [1, 2, 8, 32])
def test_cache_remains_bounded_across_many_generations(capacity: int) -> None:
    cache = RecentCache(capacity)
    for i in range(capacity*10):
        cache[i] = i
        assert cache.get(i) == i
        assert len(cache) <= capacity
    assert cache.hits == capacity*10 and cache.evictions == capacity*9


def test_archive_recognition_diagnostics_keeps_output_objects(tmp_path: Path) -> None:
    from types import SimpleNamespace
    alert = (1, 2, 3)
    pipe = SimpleNamespace(_erasure_monitor_1p=SimpleNamespace(alerts=[alert]),
        _transition_monitor_1p=SimpleNamespace(_alerts=[{'frame': 1}]))
    rows = DiskRows(tmp_path/'alerts.bin')
    frame_output = [(r, c) for _, r, c in pipe._erasure_monitor_1p.alerts]
    archive_recognition_alerts(pipe, rows)
    assert frame_output == [(2, 3)]
    assert pipe._erasure_monitor_1p.alerts == []
    assert pipe._transition_monitor_1p._alerts == []
    assert len(rows) == 2 and rows[0]['alert'] == alert
    rows.close()


def test_b11_waits_for_prior_then_uses_correct_interval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.measure_live_b11 as measure
    calls = []
    states = iter([True, False])
    monkeypatch.setattr(measure, 'b10_running', lambda pid: next(states))
    monkeypatch.setattr(measure.os, 'nice', lambda _: 0)
    monkeypatch.setattr(measure.time, 'sleep', lambda sec: calls.append('prior'))
    def idle(deadline: float, path: Path, name: str) -> bool:
        calls.append('idle')
        return True
    def execute(name: str, path: Path) -> dict:
        calls.append('long')
        assert measure.b9.END_SEC == 3412.0
        assert measure.os.environ['PUYO_MEMORY_INTERVAL'] == '300'
        return {'passed': True}
    measure.run(tmp_path, 'test', idle, execute)
    assert calls == ['prior', 'idle', 'long']
    assert json.loads((tmp_path/'status.json').read_text())['passed']


def test_b11_does_not_start_without_idle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.measure_live_b11 as measure
    monkeypatch.setattr(measure, 'b10_running', lambda pid: False)
    monkeypatch.setattr(measure.os, 'nice', lambda _: 0)
    def forbidden(*args: object) -> None:
        raise AssertionError('低負荷未達で起動した')
    measure.run(tmp_path, 'test', lambda *args: False, forbidden)


@pytest.mark.parametrize('size', [1, 2, 30, 300])
def test_disk_fifo_preserves_colors_and_reset(tmp_path: Path, size: int) -> None:
    fifo = DiskFIFO(tmp_path/'fifo.bin')
    values = [(i % 5+1, (i+1) % 5+1) for i in range(size)]
    for value in values:
        fifo.append(value)
    assert len(fifo) == size
    assert [fifo.popleft() for _ in values] == values
    assert not fifo
    with pytest.raises(IndexError):
        fifo.popleft()
    fifo.append((1, 2))
    fifo.clear()
    fifo.append((3, 4))
    assert fifo.popleft() == (3, 4)
    fifo.close()


def test_disk_fifo_interleaved_and_partial_reset(tmp_path: Path) -> None:
    fifo = DiskFIFO(tmp_path/'fifo.bin')
    fifo.append((1, 2))
    fifo.append((3, 4))
    assert fifo.popleft() == (1, 2)
    fifo.append((5, 1))
    assert fifo.popleft() == (3, 4)
    fifo.clear()
    fifo.append((2, 2))
    assert fifo.popleft() == (2, 2)
    fifo.close()
