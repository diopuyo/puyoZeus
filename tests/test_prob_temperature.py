"""PC2 の strata/apply_T を独立の比較元として配信の較正を検査する。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import production_config
from src.prob_temperature import ENV_ENABLED, apply_temperature, load_temperature
from src.phase_j.live_24h import Live24h

FIXTURE = Path(__file__).parent / 'fixtures/prob_temperature/prob_temperature_v1.json'
ROWS = 4096
TOLERANCE = 1e-12
PH = ['序盤', '中盤', '終盤']
EX = ['撃ち合いなし', '撃ち合い中']
CB = ['0〜2', '3〜5', '6〜9', '10 以上']


def strata(d: pd.DataFrame) -> pd.DataFrame:
    # PC2 calib.py の strata を転記。chain_active の NaN は比較で偽になる。
    ph = lambda x: np.where(x <= 18, 0, np.where(x >= 48, 2, 1))
    t1, t2 = d.cells_total_p1.to_numpy(float), d.cells_total_p2.to_numpy(float)
    p = np.maximum(ph(t1), ph(t2)); p[~(np.isfinite(t1) & np.isfinite(t2))] = -1
    ex = ((d.chain_active_p1 > 0) | (d.chain_active_p2 > 0)).to_numpy().astype(int)
    cd = (d.cells_color_p1-d.cells_color_p2).abs().to_numpy(float)
    cb = np.select([cd <= 2, cd <= 5, cd <= 9, cd >= 10], [0, 1, 2, 3], -1)
    cb[~np.isfinite(cd)] = -1
    pe = d[['sl_pending_p1', 'sl_pending_p2']]
    pend = np.where(pe.isna().any(axis=1), -1, (pe.fillna(0) > 0).any(axis=1).astype(int))
    d['phase'] = p; d['ex'] = ex; d['cb'] = cb; d['pend'] = pend
    d['c2'] = np.where(p >= 0, p*2+ex, -1)
    d['c3'] = np.where((p >= 0) & (cb >= 0), (p*2+ex)*4+cb, -1)
    return d


def reference(probabilities: np.ndarray, rows: pd.DataFrame, model: dict) -> np.ndarray:
    d = strata(rows.copy())
    clipped = np.clip(probabilities, 1e-6, 1-1e-6)
    X = np.log(clipped/(1-clipped))
    cname2 = lambda c: f'{PH[c//2]}・{EX[c%2]}'
    cname3 = lambda c: f'{cname2(c//4)}・差 {CB[c%4]}'
    model = dict(model, T2=np.array([model['T2'][cname2(c)] for c in range(6)]),
                 T3=np.array([model['T3'][cname3(c)] for c in range(24)]))

    def apply_T(idx: np.ndarray, model: dict) -> np.ndarray:
        # PC2 calib.py の apply_T を転記 (外部変数 d/X も同じ形)。
        x = X[idx]; T = np.full(len(idx), model['T0'])
        if model['kind'] >= 2:
            c = d.c2.to_numpy()[idx]; T = np.where(c >= 0, model['T2'][np.maximum(c, 0)], T)
        if model['kind'] >= 3:
            c = d.c3.to_numpy()[idx]; T = np.where(c >= 0, model['T3'][np.maximum(c, 0)], T)
        return 1/(1+np.exp(-T*x))

    return apply_T(np.arange(len(rows)), model)


def sample() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(20261109)
    rows = pd.DataFrame({f'{name}_p{side}': rng.choice(values, ROWS)
                        for name, values in (
                            ('cells_total', [0, 18, 19, 47, 48, 78, np.nan, np.inf]),
                            ('cells_color', [0, 2, 3, 5, 6, 9, 9.5, 10, 40, np.nan, np.inf]),
                            ('chain_active', [0, 1, np.nan]), ('sl_pending', [0, 1, np.nan]))
                        for side in (1, 2)})
    probabilities = rng.random(ROWS)
    probabilities[:4] = [0, 1, 1e-8, 1-1e-8]
    return rows, probabilities


@pytest.mark.parametrize('kind', [1, 2, 3])
def test_reference_and_swap(kind: int) -> None:
    model = dict(load_temperature(FIXTURE, True), kind=kind)
    rows, probabilities = sample()
    actual = np.array([apply_temperature(p, row, model)
                       for p, row in zip(probabilities, rows.to_dict('records'))])
    assert np.max(np.abs(actual-reference(probabilities, rows, model))) < TOLERANCE
    swapped = rows.rename(columns={f'{name}_p{side}': f'{name}_p{3-side}'
                                   for name in ('cells_total', 'cells_color', 'chain_active', 'sl_pending')
                                   for side in (1, 2)})
    reverse = np.array([apply_temperature(1-p, row, model)
                        for p, row in zip(probabilities, swapped.to_dict('records'))])
    assert np.max(np.abs(actual + reverse - 1)) < TOLERANCE


def test_off_is_exact_identity(tmp_path: Path) -> None:
    assert load_temperature(tmp_path/'missing.json', False) is None
    for p in (0., 1e-9, .3, .5, 1.):
        assert apply_temperature(p, {}, None) == p


@pytest.mark.parametrize('corrupt', [False, True])
def test_startup_error(tmp_path: Path, corrupt: bool) -> None:
    path = tmp_path/'transform.json'
    if corrupt:
        path.write_bytes(FIXTURE.read_bytes() + b' ')
    with pytest.raises(ValueError if corrupt else FileNotFoundError):
        Live24h(tmp_path, use_ama=False, temperature_path=path, temperature_enabled=True)
    assert not (tmp_path/'live_24h.jsonl').exists()


def test_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    assert production_config.DISTRIBUTION_PROB_TEMPERATURE is True
    assert load_temperature(FIXTURE) == load_temperature(FIXTURE, True)
    monkeypatch.setenv(ENV_ENABLED, '0')
    assert load_temperature(Path('missing.json')) is None


def test_predict_revalidate_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    def predict_row(row: dict) -> float:
        calls.append(dict(row))
        return .8
    live = Live24h(tmp_path, model=SimpleNamespace(predict_row=predict_row), settings={},
                   worker_factory=lambda settings: SimpleNamespace(close=lambda: None),
                   use_ama=True, temperature_path=FIXTURE, temperature_enabled=True)
    row = dict(cells_total_p1=18, cells_total_p2=19, cells_color_p1=2,
               cells_color_p2=5, chain_active_p1=0, chain_active_p2=1)
    result = dict(row=row, frame=1, t_sec=1., epoch=1, raw_probability=.1)
    expected = apply_temperature(.8, row, load_temperature(FIXTURE, True))
    live.epoch = 1
    live.selection_audit = {}
    monkeypatch.setattr(live, 'flush_events', lambda: None)
    monkeypatch.setattr(live, 'selection', lambda result: ({}, 'first', 'test'))
    display = SimpleNamespace(probability=live.predict(result), failed=False)
    assert display.probability == expected
    live.revalidate(display)
    assert len(calls) == 1
    monkeypatch.setattr(live, 'selection', lambda result: ({}, 'second', 'test'))
    live.revalidate(display)
    assert display.probability == expected
    assert len(calls) == 2
    assert abs(display.probability-apply_temperature(expected, row, live.temperature)) > TOLERANCE
    live.close()
