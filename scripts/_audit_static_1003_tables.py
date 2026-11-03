"""試合等重みの較正・差分・CV・分布外監査を再現する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import argparse
from pathlib import Path
from typing import Any
from scripts._audit_static_1003_extract import OUT, guard, save_json
import numpy as np
import pandas as pd
from src.exchange_event_features import D_COLUMNS, G_COLUMNS, g_features, PHASE_BOUNDS

PHASE_ROOT = Path('/mnt/d/puyo_analyzer/wt_phase/logs')
REFERENCE = Path(_public_dev_text('<repo>/logs'))
BINS, CELLS, CHAIN_NORM = 10, 72, 19
EPSILON = 1e-7


def loss(p: Any, y: Any) -> np.ndarray:
    """最終勝者に対する自然対数損失。"""
    p = np.clip(p, EPSILON, 1 - EPSILON)
    return -np.log(np.where(np.asarray(y) == 1, p, 1 - p))


def add_axes(frame: pd.DataFrame, design: np.ndarray) -> pd.DataFrame:
    """1P視点の差分と、小さい方を遅い方とする局面を作る。"""
    index = {n: i for i, n in enumerate(D_COLUMNS)}
    total = design[:, index['board_puyo_total']] * CELLS
    diff = design[:, index['diff_board_puyo_total']] * CELLS
    sign = frame['sign'].to_numpy() if 'sign' in frame else np.ones(len(frame))
    frame['amount_diff'] = np.rint(diff * sign)
    frame['chain_diff'] = np.rint(design[:, index['diff_current_max_chain']] * CHAIN_NORM * sign)
    frame['ojama_diff'] = np.rint(design[:, index['diff_board_ojama_count']] * CELLS * sign)
    for mode, amounts in [('slow', np.minimum(total, total - diff)),
                           ('legacy_max', np.maximum(total, total - diff))]:
        rounded = np.rint(amounts)
        frame['U_' + mode] = np.where(rounded <= 18, 'early', np.where(rounded < 48, 'middle', 'late'))
        frame.loc[~np.isfinite(amounts), 'U_' + mode] = 'missing'
    return frame


def summary(g: pd.DataFrame) -> dict:
    """条件付きでも元の試合等重みを維持して集計する。"""
    w = g.weight.to_numpy()
    result = dict(rows=len(g), games=g.game.nunique(), weight=float(w.sum()),
                  outcome=float(np.average(g.y, weights=w)))
    if 'row_id' in g:
        result['unique_rows'] = g.row_id.nunique()
    if 'dt' in g:
        result['seconds'] = float(g.dt.sum())
    for model in ('R1s', 'LR'):
        p = g[model].to_numpy()
        means = np.average(p, weights=w)
        bands = np.minimum((p * BINS).astype(int), BINS - 1)
        ece = sum(abs(np.dot((p - g.y)[bands == i], w[bands == i])) for i in range(BINS)) / w.sum()
        result.update({model + '_p': means, model + '_LL': np.average(loss(p, g.y), weights=w),
                       model + '_ECE': ece, model + '_near50': np.average(abs(p - .5) <= .05, weights=w)})
    result['delta_LL'] = result['R1s_LL'] - result['LR_LL']
    return result


def tables(frame: pd.DataFrame, prefix: str) -> None:
    """全体・局面・分位・確率帯を同じ母集団から作る。"""
    groups = [('all', 'all', frame)]
    axes = ['U_slow', 'U_legacy_max', 'amount_diff', 'chain_diff', 'ojama_diff']
    for axis in axes:
        labels = frame[axis] if axis.startswith('U_') else pd.qcut(frame[axis], 5, duplicates='drop')
        labels = labels.astype(str).replace('nan', 'missing')
        groups += [(axis, str(b), g) for b, g in frame.groupby(labels, observed=True)]
    results, calibration = [], []
    for axis, band, g in groups:
        results.append(dict(axis=axis, band=band, **summary(g)))
        for model in ('R1s', 'LR'):
            labels = np.minimum((g[model] * BINS).astype(int), BINS - 1)
            for pbin, part in g.groupby(labels, observed=True):
                calibration.append(dict(axis=axis, band=band, model=model, bin=int(pbin),
                    rows=len(part), games=part.game.nunique(), weight=part.weight.sum(),
                    prediction=np.average(part[model], weights=part.weight),
                    outcome=np.average(part.y, weights=part.weight)))
    pd.DataFrame(results).to_csv(OUT / f'{prefix}_strata.csv', index=False)
    pd.DataFrame(calibration).to_csv(OUT / f'{prefix}_calibration.csv', index=False)


def cv() -> None:
    """3seedのOOF損失を反復平均。全行3倍を独立標本とは数えない。"""
    guard()
    root = REFERENCE / 'r2_baselines_148_20260926'
    rows = pd.read_csv(root / 'rows.csv')
    d = np.load(root / 'design.npy', mmap_mode='r')
    predictions = pd.read_csv(PHASE_ROOT / 'gfe_phase_reference_20261002/predictions.csv',
                             usecols=['row_id', 'seed', 'fold', 'label', 'V0', 'R1'])
    verify_stage1(predictions)
    frame = rows.iloc[predictions.row_id.to_numpy()].reset_index(drop=True)
    frame['game'], frame['y'] = frame.game_id, predictions.label.to_numpy()
    frame['R1s'], frame['LR'] = predictions.R1.to_numpy(), predictions.V0.to_numpy()
    frame['seed'], frame['fold'] = predictions.seed.to_numpy(), predictions.fold.to_numpy()
    frame['weight'] = 1 / frame.groupby('game').game.transform('size') / frame.game.nunique()
    frame = add_axes(frame, d[predictions.row_id.to_numpy()])
    with np.load(PHASE_ROOT / 'gfe_phase_stage1_20261002/user_phase.npz') as z:
        missing = z['missing'][predictions.row_id.to_numpy()].astype(bool)
    frame.loc[missing, ['U_slow', 'U_legacy_max']] = 'missing'
    frame.loc[missing, ['amount_diff', 'chain_diff', 'ojama_diff']] = np.nan
    tables(frame, 'cv')
    seed_results = [dict(seed=int(seed), **summary(g)) for seed, g in frame.groupby('seed')]
    pd.DataFrame(seed_results).to_csv(OUT / 'cv_seeds.csv', index=False)
    training_bounds(rows, d)
    save_json('cv_population.json', dict(rows=len(rows), prediction_rows=len(predictions),
        games=rows.game_id.nunique(), videos=rows.video_id.nunique(), seeds=3,
        note='静止学習行OOF。表示経路・表示秒は保存されておらずG_fe表示時間そのものではない。'))
    print('CV完了', summary(frame), flush=True)


def verify_stage1(reference: pd.DataFrame) -> None:
    """referenceの旧LR列が段1のV0予測と同じ行・seed・foldであることを検算する。"""
    columns = ['row_id', 'seed', 'fold', 'V0']
    offset = 0
    for chunk in pd.read_csv(PHASE_ROOT / 'gfe_phase_stage1_20261002/predictions.csv',
                             usecols=columns, chunksize=100000):
        expected = reference.iloc[offset:offset + len(chunk)]
        np.testing.assert_array_equal(chunk[columns[:3]], expected[columns[:3]])
        np.testing.assert_allclose(chunk.V0, expected.V0, atol=1e-12, rtol=0)
        offset += len(chunk)
    assert offset == len(reference)
    save_json('cv_baseline_identity.json', dict(rows=offset, stage1_reference_V0_equal=True))


def training_bounds(rows: pd.DataFrame, d: np.ndarray) -> None:
    """モデルに渡す59列全体の範囲を保存する。"""
    with np.load(REFERENCE / 'r2_causal_phase_fe_148_20260926/phase_inputs.npz') as z:
        thresholds = np.quantile(z['elapsed'], PHASE_BOUNDS)
        phases = np.column_stack((z['fill'], np.searchsorted(thresholds, z['elapsed'], side='left')))
    x = g_features(d, rows.A.to_numpy(), rows.sign.to_numpy(), phases)
    bounds = pd.DataFrame({'column': G_COLUMNS, 'min': np.min(x, axis=0),
                           'p01': np.quantile(x, .01, axis=0), 'p99': np.quantile(x, .99, axis=0),
                           'max': np.max(x, axis=0), 'median': np.median(x, axis=0)})
    bounds.to_csv(OUT / 'training_bounds.csv', index=False)
    np.save(OUT / 'training_sample.npy', x[np.arange(0, len(x), 20)])


def main() -> None:
    """再現可能な集計入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', action='store_true')
    if parser.parse_args().cv:
        cv()


if __name__ == '__main__':
    main()
