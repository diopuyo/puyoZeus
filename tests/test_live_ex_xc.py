"""学習との順序・全16列の一致と、診断を残す経路を検査する。"""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.phase_j import live_ex_compute as compute
from src.phase_j.live_ex_engine import LiveExEngine, EventMonitor
from src.phase_j._live_ex_frozen.src.dl1k_engine import ExEngine
from src.phase_j._live_ex_frozen.src.dl1k_retr import compute_retr
from src.phase_j._live_ex_frozen.src.dl1k_xc import COLUMNS, compute_xc
from tests.test_live_fire_board_retr import fixture
from tests.test_live_fire_board import boards


def exchange() -> tuple:
    engine = fixture()
    _, fired = boards()
    raw = [dict(raw_cnn_board=fired.tolist(), formula=dict(valid=False)), {}]
    ExEngine.step(engine, .9, 1, raw)
    raw[0]['formula'] = dict(valid=True, product=100)
    ExEngine.step(engine, 1.01, 1, raw)
    value = dict(pending=[0, 1], rate=70, board_prediction_active=[True, False],
                 reconciliation=[dict(times=[1.5]), None], boards=[None, None])
    engine.projection = lambda: deepcopy(value)
    engine.probe = lambda: SimpleNamespace(fight=True, key=(0, 1, (1, None)))
    return engine, value


@pytest.mark.parametrize('enabled', [False, True])
def test_calculate_matches_training_before_proof(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_EX_RETR', '1')
    monkeypatch.setenv('PUYO_EX_XC', str(int(enabled)))
    reference, value = exchange()
    compute_retr(reference, value)
    expected, why = compute_xc(reference, None)
    engine, _ = exchange()
    wrapper = object.__new__(LiveExEngine)
    wrapper.engine, wrapper.monitor = engine, EventMonitor()
    order = []
    actual_xc = compute.compute_xc

    def xc(e: object, v: dict | None) -> tuple:
        assert e.retr_diag[1]['I'] == 6
        order.append('xc')
        return actual_xc(e, v)

    def proof(e: object, v: dict) -> dict:
        order.append('proof')
        v['board_prediction_active'] = [False, False]
        v['reconciliation'] = [None, None]
        return dict(ok=True, death_sides=[])

    monkeypatch.setattr(compute, 'compute_xc', xc)
    monkeypatch.setattr(compute, 'proof_of', proof)
    monkeypatch.setattr(compute, 'search_of', lambda *a: (order.append('search'), None, None))
    result = wrapper.calculate()
    assert order == (['xc'] if enabled else []) + ['proof', 'search']
    assert not result['result']['failure']
    if enabled:
        for column in COLUMNS:
            a, b = result['extra'][column], expected[column]
            assert a == b or math.isnan(a) and math.isnan(b), column
        assert result['result']['xc_why'] == why
    else:
        assert not set(COLUMNS) & set(result['extra'])
    assert result['result']['retr_attack'][1] == dict(
        board_prediction_active=True, reconciliation_is_none=False)
    assert all(v >= 0 for v in result['result']['compute_ms_by_stage'].values())


def test_failed_xc_does_not_discard_retr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_EX_RETR', '1')
    monkeypatch.setenv('PUYO_EX_XC', '1')
    engine, _ = exchange()
    engine.xc_diag = {'stale': True}
    monkeypatch.setattr(compute, 'compute_xc', lambda *a: 1 / 0)
    monkeypatch.setattr(compute, 'proof_of', lambda *a: {})
    monkeypatch.setattr(compute, 'search_of', lambda *a: (None, None, None))
    value = compute.compute(engine)
    assert value['failure'] is None and value['retr']['retR_r_p2'] == 6
    assert value['xc_diag'] is None and all(math.isnan(x) for x in value['xc'].values())
    assert all(w.startswith('error:') for w in value['xc_why'])


def test_flush_keeps_only_small_diagnostics(tmp_path: Path) -> None:
    from src.phase_j.live_24h import Live24h
    owner = object.__new__(Live24h)
    detail = dict(retr_why=['board', 'ama'], retr_ms=2., retr_diag=[{'I': 6}, None],
                  retr_attack=[{'board_prediction_active': True}], xc_why=['forecast', 'board'],
                  xc_diag={'net': [0, 6]}, compute_ms_by_stage=dict.fromkeys(compute.STAGES, 1.),
                  retr_ama={'calls': 3, 'hits': 2}, compute_ms=5., output={'work_units': 4},
                  boards=[[1] * 78])
    event = dict(kind='result', value=dict(result=detail, extra={}))
    owner.worker = SimpleNamespace(take_events=lambda: [event])
    written = []
    owner.write = written.append
    owner.flush_events()
    actual = written[0]['value']
    for key in ('retr_why', 'retr_ms', 'retr_diag', 'retr_attack', 'xc_why', 'xc_diag',
                'compute_ms_by_stage', 'retr_ama'):
        assert actual[key] == detail[key]
    assert 'boards' not in actual and 'result' not in actual
    json.dumps(actual)
