"""撃ち返し補正の境界、対称性、再推論、配布の既定を検査する。"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.counter_fire_shift import (
    CounterFireShift, ENV_ENABLED, TIME_EPSILON, WINDOW_SEC, apply_shift, load_shift, onset_cell,
)
from src.phase_j.live_24h import Live24h
from src.prob_temperature import apply_temperature, load_temperature

TEMPERATURE = Path(__file__).parent/'fixtures/prob_temperature/prob_temperature_v1.json'


def row(active1: int = 0, active2: int = 1, **changes: float) -> dict:
    return dict(dict(chain_active_p1=active1, chain_active_p2=active2,
                     mk_xc_p1=0, mk_xc_p2=0, max_height_p1=11, max_height_p2=12,
                     chain_pred_ojama_p1=10, chain_pred_ojama_p2=20), **changes)


def test_explicit_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ENABLED, '0')
    assert load_shift(Path('absent.json')) is None
    off = CounterFireShift(None)
    for t in range(3):
        assert off.apply(.3, t, row(t % 2)) == .3
    assert off.previous is None


def test_distribution_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.production_config import DISTRIBUTION_COUNTER_FIRE_SHIFT
    from src.phase_j.release_config import release_environment
    # 2026-11-12 user 採用で 0.6.5 は既定 OFF (光の直しと二重の補正になるため)。明示の '1' で ON にできることも保つ。
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    assert DISTRIBUTION_COUNTER_FIRE_SHIFT is False
    assert load_shift() is None
    assert release_environment({})[ENV_ENABLED] == '0'
    assert release_environment({ENV_ENABLED: '1'})[ENV_ENABLED] == '1'


def test_sha(tmp_path: Path) -> None:
    path = tmp_path/'modified.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='SHA256'):
        load_shift(path, True)


def test_distribution_child_reads_enabled_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import subprocess
    import sys
    from src.phase_j.distribution_runtime import DistributionRuntime
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    runtime = DistributionRuntime(tmp_path)
    try:
        _, environment = runtime._heartbeat_environment()
        root = Path(__file__).resolve().parents[1]
        code = (f'import sys; sys.path.insert(0, {str(root)!r}); '
                'import json, os; from src.counter_fire_shift import load_shift; '
                'print(json.dumps([os.environ["PUYO_COUNTER_FIRE_SHIFT"], load_shift() is not None]))')
        result = subprocess.run([sys.executable, '-B', '-c', code], env=environment,
                                capture_output=True, text=True, check=True)
        assert json.loads(result.stdout) == ['0', False]
    finally:
        runtime.close_preview()


def test_live_requires_temperature(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='24層温度較正'):
        Live24h(tmp_path, use_ama=False, temperature_enabled=False,
                counter_fire_shift_enabled=True)


@pytest.mark.parametrize(('changes', 'cell'), [
    ({}, 0), ({'max_height_p1': 12}, 1), ({'mk_xc_p1': 1}, 3),
    ({'mk_xc_p1': 1, 'max_height_p1': 12}, 4),
    ({'mk_xc_p1': float('nan')}, -1), ({'max_height_p1': float('nan')}, -1),
    ({'mk_xc_p1': 1, 'chain_pred_ojama_p1': 20}, -1),
    ({'mk_xc_p1': 1, 'chain_pred_ojama_p1': 21}, -1),
    ({'mk_xc_p1': 1, 'chain_pred_ojama_p2': float('nan')}, -1),
    ({'chain_pred_ojama_p1': float('nan')}, 0),
])
def test_cell(changes: dict, cell: int) -> None:
    assert onset_cell(row(**changes), 1) == cell


def test_window_freeze_and_reset() -> None:
    tracker = CounterFireShift(load_shift(enabled=True))
    assert tracker.apply(.5, 0, row()) == .5
    p = tracker.apply(.5, 1, row(1))
    assert p > .5
    assert tracker.apply(.5, 1, row(1, mk_xc_p1=1)) == p
    assert tracker.apply(.5, 2.5, row(0, max_height_p1=float('nan'))) == p
    assert tracker.apply(.5, 2.5001, row()) == .5
    tracker.reset()
    assert tracker.apply(.5, 0, row(1)) == .5


def test_latest_excluded_and_next_same_side() -> None:
    tracker = CounterFireShift(load_shift(enabled=True))
    tracker.observe(0, row())
    assert tracker.apply(.5, .1, row(1)) > .5
    tracker.observe(.2, row(1, 0))
    assert tracker.apply(.5, .3, row(1, 1, mk_xc_p2=float('nan'))) == .5
    tracker.observe(.4, row(0, 0))
    assert tracker.apply(.5, .5, row(1, 0)) == .5


def test_simultaneous() -> None:
    tracker = CounterFireShift(load_shift(enabled=True))
    tracker.observe(0, row(0, 0))
    assert tracker.apply(.5, 1, row(1, 1)) == .5


def test_pc2_exact_time_boundary() -> None:
    tracker = CounterFireShift(load_shift(enabled=True))
    start = 10000.1
    tracker.observe(start-1, row())
    tracker.observe(start, row(1))
    end = start+WINDOW_SEC+TIME_EPSILON
    assert tracker.apply(.5, end, row(1)) > .5
    assert tracker.apply(.5, end+TIME_EPSILON, row(1)) == .5


@pytest.mark.parametrize('p', [0, .01, .3, .5, .99, 1])
@pytest.mark.parametrize('cell', [-1, 0, 1, 2, 3, 4, 17])
def test_symmetry_and_identity(p: float, cell: int) -> None:
    model = load_shift(enabled=True)
    assert abs(apply_shift(p, 1, cell, model) + apply_shift(1-p, -1, cell, model)-1) < 1e-12
    assert apply_shift(p, 1, cell, None) == p
    if cell not in (0, 1, 3, 4):
        assert apply_shift(p, 1, cell, model) == p


def test_live_once_after_temperature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live = Live24h(tmp_path, model=SimpleNamespace(predict_row=lambda r: .8), settings={},
                   worker_factory=lambda s: SimpleNamespace(close=lambda: None), use_ama=True,
                   temperature_path=TEMPERATURE, temperature_enabled=True,
                   counter_fire_shift_enabled=True, midgame_board_layer_enabled=False)
    live.selection_audit = {}
    monkeypatch.setattr(live, 'selection', lambda r: ({}, 'same', 'test'))
    monkeypatch.setattr(live, 'flush_events', lambda: None)
    result = dict(row=row(), frame=0, t_sec=0, epoch=1)
    live.predict(result)
    result.update(row=row(1), frame=1, t_sec=1)
    calibrated = apply_temperature(.8, result['row'], load_temperature(TEMPERATURE, True))
    expected = apply_shift(calibrated, 1, 0, load_shift(enabled=True))
    assert live.predict(result) == expected
    assert live.predict(result) == expected
    assert live.counter_fire_shift.time == 1
    live.worker = None
    live.update(dict(game_idx=2), epoch=2, captured_at=2.)
    assert live.counter_fire_shift.previous is None
    assert live.predict(result) == calibrated
    live.close()
