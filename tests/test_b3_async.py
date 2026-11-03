"""全状態更新、独立snapshot、最新要求の置換とspawn終了を検証する。"""
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import pytest

from src.b3_async import AsyncB3, _StateLoop, calculate, snapshot
from src.b3_realtime import RealtimeRows
from src.b3_realtime_replay import replay_inputs
from src.b3_timing import TimingTable
from tests.b3_synth import build_record


class ConstantModel:
    def predict_row(self, row: dict) -> float:
        return .25


def packet(item: tuple, epoch: int = 1) -> bytes:
    return pickle.dumps((*item, epoch, True, dict(submitted_at=perf_counter(), captured_at=0.)))


@pytest.mark.parametrize('drop', [False, True])
def test_all_events_and_snapshot_parity(tmp_path: Path, drop: bool) -> None:
    items = list(replay_inputs(build_record(tmp_path)))
    sync = RealtimeRows(TimingTable())
    state = _StateLoop(TimingTable(), 'live', drop, False, None)
    expected = {}
    for item in items:
        row = sync.update(*item)
        if row is not None:
            expected[row['frame_idx']] = row
        state.accept(packet(item))
    assert state.counts['inputs'] == len(items)
    assert state.counts['triggers'] == len(expected)
    assert len(state.pending) == (1 if drop else len(expected))
    for request in state.pending:
        result = calculate(request, RealtimeRows(TimingTable()), ConstantModel(), False)
        assert result['error'] is None
        np.testing.assert_equal(result['row'], expected[request['frame']])
    np.testing.assert_equal(state.rows.state.sl_snap, sync.state.sl_snap)


def test_snapshot_does_not_alias_state(tmp_path: Path) -> None:
    items = list(replay_inputs(build_record(tmp_path)))
    rows = RealtimeRows(TimingTable())
    rows.update(*items[0])
    original = snapshot(rows)
    for item in items[1:]:
        rows.update(*item, emit=False)
    state, _ = pickle.loads(original)
    assert state.frame == items[0][0]['frame']
    assert state.frame != rows.state.frame


def test_spawn_missing_model_closes_children(tmp_path: Path) -> None:
    import multiprocessing as mp
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(RuntimeError, match='FileNotFoundError'):
        AsyncB3(tmp_path/'absent.joblib', TimingTable())
    assert {p.pid for p in mp.active_children()} == before


def test_boundary_clears_pending_even_without_trigger(tmp_path: Path) -> None:
    from copy import deepcopy
    items = list(replay_inputs(build_record(tmp_path)))
    state = _StateLoop(TimingTable(), 'live', True, False, None)
    first = next(i for i, item in enumerate(items) if item[2])
    state.accept(packet(items[first]))
    assert state.pending
    frame = deepcopy(items[first + 1][0])
    frame.update(game_idx=999, match_active=False)
    state.accept(packet((frame, (), ())))
    assert not state.pending


def test_new_epoch_accepts_restarted_frame_numbers(tmp_path: Path) -> None:
    items = list(replay_inputs(build_record(tmp_path)))
    state = _StateLoop(TimingTable(), 'live', True, False, None)
    for item in items:
        state.accept(packet(item))
    state.accept(packet(items[0], epoch=2))
    assert state.rows.state.frame == items[0][0]['frame']
    assert all(p['epoch'] == 2 for p in state.pending)


def test_nonfinite_prediction_becomes_failure(tmp_path: Path) -> None:
    class InvalidModel:
        def predict_row(self, row: dict) -> float:
            return float('nan')
    state = _StateLoop(TimingTable(), 'live', True, True, None)
    state.accept(packet(next(item for item in replay_inputs(build_record(tmp_path)) if item[2])))
    result = calculate(state.pending[0], RealtimeRows(TimingTable()), InvalidModel())
    assert result['error'].startswith('ValueError')
    assert result['raw_probability'] is None
    assert result['timings']['inference_ms'] >= 0
