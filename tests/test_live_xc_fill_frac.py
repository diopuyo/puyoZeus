"""学習と同じ丸め、選択後の派生、左右と失効を検査する。"""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.b3_column_spec import NET_LEDGER_FEATURE_NAMES
from src.phase_j.live_24h import Live24h
from src.phase_j.live_24h_model import LEARNING_COLUMNS, XC_COLUMNS, validate_model
from src.phase_j.live_xc_fill_frac import XC_FILL_FRAC_COLUMNS, fill_fractions
from tests.test_live_24h_model import Worker, extra, model, result


@pytest.mark.parametrize('net,room,expected', [
    (1., 2., 1 / 3), (0., 0., 1.), (0., 5., 0.), (8., 0., 1.),
    (math.nan, 0., math.nan), (0., math.nan, math.nan),
    (math.nan, math.nan, math.nan), (16777217., 1., 16777217 / 16777218),
])
def test_float64_then_float32(net: float, room: float, expected: float) -> None:
    values = dict(xc_net_recv_p1=net, xc_death_room_p1=room)
    actual = fill_fractions(values, XC_FILL_FRAC_COLUMNS)[XC_FILL_FRAC_COLUMNS[0]]
    assert np.float32(actual).tobytes() == np.float32(expected).tobytes()
    assert math.isnan(fill_fractions(values, XC_FILL_FRAC_COLUMNS)[XC_FILL_FRAC_COLUMNS[1]])


def candidate(calls: list, declared: bool = True) -> SimpleNamespace:
    value = model(calls)
    columns = (*NET_LEDGER_FEATURE_NAMES, *LEARNING_COLUMNS, *XC_COLUMNS,
               *(XC_FILL_FRAC_COLUMNS if declared else ()))
    value.static = value.exchange = SimpleNamespace(columns=columns)
    value.members = (1, 2, 3)
    return value


@pytest.mark.parametrize('declared', [False, True])
@pytest.mark.parametrize('same_time', [False, True])
@pytest.mark.parametrize('held', [False, True])
@pytest.mark.parametrize('xc', [False, True])
def test_selection_and_model_gate(tmp_path: Path, declared: bool, same_time: bool,
                                  held: bool, xc: bool) -> None:
    calls, worker = [], Worker(extra())
    raw = dict(xc_net_recv_p1=1., xc_death_room_p1=2.,
               xc_net_recv_p2=2., xc_death_room_p2=3.)
    if xc:
        worker.extra['extra'].update(raw)
    worker.extra['held'] = held
    key = dict(fight=xc, xc={k: v * 2 if k.endswith('p1') else 0. for k, v in raw.items()},
               input_t=3., serial=7)
    worker.xc_history = SimpleNamespace(key_at=lambda _: key)
    value = candidate(calls, declared)
    validate_model(value)
    owner = Live24h(tmp_path, value, dict(xc=xc, xc_same_time=same_time),
                      lambda _: worker, use_ama=True, temperature_enabled=False)
    try:
        owner.predict(result())
        expected = fill_fractions(key['xc'] if same_time and xc else raw if xc else {},
                                  XC_FILL_FRAC_COLUMNS if declared else ())
        assert set(calls[0]) & set(XC_FILL_FRAC_COLUMNS) == set(expected)
        for name, number in expected.items():
            assert np.float32(calls[0][name]).tobytes() == np.float32(number).tobytes()
    finally:
        owner.close()


@pytest.mark.parametrize('xc_only', [False, True])
def test_inference_expiry_recomputes(tmp_path: Path, xc_only: bool) -> None:
    calls, worker = [], Worker(extra(xc_net_recv_p1=1., xc_death_room_p1=2.))
    value = candidate(calls)
    owner = Live24h(tmp_path, value, {}, lambda _: worker, use_ama=True, temperature_enabled=False)
    count = 0

    def selection(_: dict) -> tuple:
        nonlocal count
        count += 1
        owner.selection_audit = {}
        signature = ((1, 2), 'xc', count) if xc_only else (1, count)
        return dict(worker.extra['extra']), signature, 'test'

    owner.selection = selection
    try:
        owner.predict(result())
        assert len(calls) == 2
        assert calls[0][XC_FILL_FRAC_COLUMNS[0]] == float(np.float32(1 / 3))
        assert all(math.isnan(calls[1][c]) for c in XC_FILL_FRAC_COLUMNS)
    finally:
        owner.close()


def test_side_swap_commutes() -> None:
    raw = dict(xc_net_recv_p1=1., xc_death_room_p1=2.,
               xc_net_recv_p2=3., xc_death_room_p2=1.)
    swapped = {k: raw[k[:-2] + ('p2' if k.endswith('p1') else 'p1')] for k in raw}
    normal = fill_fractions(raw, XC_FILL_FRAC_COLUMNS)
    reverse = fill_fractions(swapped, XC_FILL_FRAC_COLUMNS)
    assert list(reverse.values()) == list(reversed(normal.values()))
    assert reverse[XC_FILL_FRAC_COLUMNS[0]] != 1 - normal[XC_FILL_FRAC_COLUMNS[0]]


def test_ama_off_does_not_infer(tmp_path: Path) -> None:
    calls = []
    owner = Live24h(tmp_path, candidate(calls), {}, use_ama=False, temperature_enabled=False)
    try:
        assert owner.predict(result()) == result()['raw_probability']
        assert calls == []
    finally:
        owner.close()


def test_hold_reset_expiry_and_outside_fight() -> None:
    from src.phase_j.live_ex_hold import LiveExHold
    from tests.test_live_24h_hold import event, source
    hold, raw = LiveExHold(), source()
    raw['extra'].update(xc_net_recv_p1=1., xc_death_room_p1=2.,
                        xc_net_recv_p2=2., xc_death_room_p2=3.)
    initial = fill_fractions(hold.adopt(event(.2), raw, 'ok'), XC_FILL_FRAC_COLUMNS)
    held = fill_fractions(hold.adopt(event(.3), None, 'stale'), XC_FILL_FRAC_COLUMNS)
    assert held == initial
    reset = fill_fractions(hold.adopt(event(.4, reset=dict(p1=True, p2=False)), None, 'stale'),
                           XC_FILL_FRAC_COLUMNS)
    assert math.isnan(reset[XC_FILL_FRAC_COLUMNS[0]])
    assert reset[XC_FILL_FRAC_COLUMNS[1]] == initial[XC_FILL_FRAC_COLUMNS[1]]
    for key in (event(.71), event(.8, fight=False)):
        missing = fill_fractions(hold.adopt(key, None, 'stale'), XC_FILL_FRAC_COLUMNS)
        assert all(math.isnan(v) for v in missing.values())


@pytest.mark.parametrize('mode', [0., 1.])
def test_prediction_matrix_preserves_side_and_rounding(mode: float) -> None:
    from src.b3_train_model import TwoStateModel
    calls = []

    def predict_proba(matrix: np.ndarray) -> np.ndarray:
        calls.append(matrix.copy())
        return np.array([[.2, .8]])

    names = list(reversed(XC_FILL_FRAC_COLUMNS))
    evaluator = SimpleNamespace(columns=names, model=SimpleNamespace(predict_proba=predict_proba))
    value = TwoStateModel(evaluator, evaluator)
    raw = dict(xc_net_recv_p1=1., xc_death_room_p1=2.,
               xc_net_recv_p2=3., xc_death_room_p2=1.)
    row = dict(mode_exchange=mode, **fill_fractions(raw, XC_FILL_FRAC_COLUMNS))
    assert value.predict_row(row) == .8
    assert calls[0].dtype == np.float32
    assert calls[0].tobytes() == np.asarray([[.75, 1 / 3]], np.float32).tobytes()
