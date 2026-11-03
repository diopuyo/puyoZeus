"""段2悪化を表示行・記録入力・CV時間再重み付けに分けて診断する。"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scripts.gfe_phase_stage1_20261002 import memory_guard, datasets, REFERENCE, OUTPUT, save_json
from scripts.gfe_phase_stage2_replay_20261002 import SET1, SET2, EVALSET, record_path, model_directory
from src.exchange_event_features import D_COLUMNS, G_COLUMNS, g_features, PHASE_BOUNDS
from src.exchange_event_user_phase import USER_COLUMNS, user_phase, user_features

OUT = Path('logs/gfe_phase_stage2_diag_20261002')
STAGE2 = Path('logs/gfe_phase_stage2_20261002')
EPSILON, BINS, GAMES = 1e-7, 10, 115
TIME_EDGES = np.array([0, 10, 20, 30, 45, 60, 90, 120, 180, 300, np.inf])
QUANTILES = [0, .01, .25, .5, .75, .99, 1]


def loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """段2採点と同じクリップで行損失を作る。"""
    p = np.clip(p, EPSILON, 1-EPSILON)
    return -np.log(np.where(y == 1, p, 1-p))


def display_frames() -> pd.DataFrame:
    """115試合の表示行に試合等重みを割り当てる。"""
    parts = []
    for set_name, sources, labels in [('set1', SET1, EVALSET/'labels.json'),
                                      ('set2', SET2, EVALSET/'set2/labels.json')]:
        displays = []
        for source in sources:
            with np.load(STAGE2/'baseline/replay'/source/'display.npz') as a:
                with np.load(STAGE2/'V3/replay'/source/'display.npz') as b:
                    np.testing.assert_array_equal(a['t_sec'], b['t_sec'])
                    displays.append(pd.DataFrame(dict(t=a['t_sec'], p0=a['display_p1'],
                        p3=b['display_p1'], route=a['source'], route3=b['source'], source=source)))
        display = pd.concat(displays).sort_values('t')
        assert display.t.is_unique
        games = json.loads(labels.read_text())['games']
        for game in games:
            d = display[(display.t >= game['start']) & (display.t < game['end'])].copy()
            d['set'], d['game'], d['y'] = set_name, game['game'], int(game['winner'] == '1P')
            d['elapsed'], d['progress'] = d.t-game['start'], (d.t-game['start'])/(game['end']-game['start'])
            d['weight'], d['duration'] = 1/(GAMES*len(d)), game['end']-game['start']
            parts.append(d)
    result = pd.concat(parts, ignore_index=True)
    result['l0'], result['l3'] = loss(result.p0, result.y), loss(result.p3, result.y)
    result['delta'], result['dp'] = result.l3-result.l0, result.p3-result.p0
    result['time_bin'] = np.minimum(np.searchsorted(TIME_EDGES, result.elapsed, side='right')-1, BINS-1)
    result['progress_bin'] = np.minimum((result.progress*BINS).astype(int), BINS-1)
    expected = json.loads((STAGE2/'METRICS.json').read_text())['decisions']['V3']['overall']['difference']
    assert abs((result.delta*result.weight).sum()-expected) < 1e-12
    return result


def grouped(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """寄与の和が元のM_time差に戻る形で分解する。"""
    rows = []
    for key, g in d.groupby(keys, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        w = g.weight.to_numpy()
        rows.append(dict(zip(keys, key), frames=len(g), mass=w.sum(),
            baseline=np.average(g.l0, weights=w), V3=np.average(g.l3, weights=w),
            difference=np.average(g.delta, weights=w), contribution=np.dot(g.delta, w),
            probability_delta=np.average(g.dp, weights=w)))
    return pd.DataFrame(rows)


def calibration(d: pd.DataFrame) -> pd.DataFrame:
    """G_fe同一表示行を使い、試合等重みで確率帯と実勝率を比較する。"""
    rows = []
    for model, column in [('baseline', 'p0'), ('V3', 'p3')]:
        bins = np.minimum((d[column]*BINS).astype(int), BINS-1)
        for band in range(BINS):
            g = d[bins == band]
            if len(g):
                rows.append(dict(model=model, bin=band, frames=len(g), mass=g.weight.sum(),
                    prediction=np.average(g[column], weights=g.weight),
                    outcome=np.average(g.y, weights=g.weight)))
    return pd.DataFrame(rows)


def summarize_display(d: pd.DataFrame) -> None:
    """全試合・時間・経路・較正の原表を保存する。"""
    for name, keys in [('games', ['set', 'game']), ('routes', ['route']),
                       ('route_transitions', ['route', 'route3']), ('time', ['time_bin']),
                       ('progress', ['progress_bin']), ('route_time', ['route', 'time_bin'])]:
        grouped(d, keys).to_csv(OUT/f'{name}.csv', index=False)
    g = d[(d.route == 'G_fe') & (d.route3 == 'G_fe')]
    calibration(g).to_csv(OUT/'calibration.csv', index=False)
    save_json(OUT/'display_summary.json', dict(frames=len(d), games=GAMES,
        route_changes=int((d.route != d.route3).sum()), gfe_frames=len(g),
        difference=float(np.dot(d.delta, d.weight)),
        probability_delta_quantiles=dict(zip(map(str, QUANTILES), np.quantile(g.dp, QUANTILES).tolist())),
        gfe_probability_delta_weighted=float(np.average(g.dp, weights=g.weight)),
        gfe_winner_rate=float(np.average(g.y, weights=g.weight))))


def extract_record(source: str) -> dict[str, np.ndarray]:
    """記録のstatic_callを欠測補完前のDとともに復元する。"""
    cache, values = {}, []
    current_t, current_game = np.nan, -1
    with gzip.open(record_path(source), 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            if r['kind'] == 'update':
                current_t, current_game = r['args']['tuple'][3:5]
            elif r['kind'] == 'static':
                cache[r['key']] = r['input']['d_features']['array']
            elif r['kind'] == 'static_call':
                values.append((cache[r['key']], r['m0_probability_1p'],
                    r['elapsed_sec'], r['source_side'], current_t, current_game))
    return dict(raw=np.asarray([v[0] for v in values]), p=np.array([v[1] for v in values]),
        elapsed=np.array([v[2] for v in values]), side=np.array([v[3] for v in values]),
        time=np.array([v[4] for v in values]), game=np.array([v[5] for v in values]))


def stats(x: np.ndarray, columns: tuple, name: str) -> list[dict]:
    """欠測率・0率・範囲・分位・定数列を全列で計測する。"""
    result = []
    for i, col in enumerate(columns):
        v = x[:, i]
        finite = v[np.isfinite(v)]
        q = np.quantile(finite, QUANTILES) if len(finite) else np.full(len(QUANTILES), np.nan)
        result.append(dict(dataset=name, column=col, rows=len(v), nan=float(np.isnan(v).mean()),
            zero=float((v == 0).mean()), constant=bool(len(finite) and np.min(finite) == np.max(finite)),
            **dict(zip(('min','p01','p25','p50','p75','p99','max'), q))))
    return result


def model_features(raw: np.ndarray, p: np.ndarray, side: np.ndarray,
                   elapsed: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """学習とは異なる補完タイミングも明示して提供変換を再現する。"""
    own, diff = (raw[:, D_COLUMNS.index(c)] for c in ('board_puyo_total','diff_board_puyo_total'))
    phase, missing = user_phase(own, diff)
    progress = own-diff/2
    fill = np.searchsorted(PHASE_BOUNDS, np.where(np.isfinite(progress), np.clip(progress, 0, 1), 0), side='left')
    d = np.nan_to_num(raw, nan=0.0)
    x54 = user_features(d, p, 1-2*side, phase, missing)
    x59 = g_features(d, p, 1-2*side, np.column_stack((fill, np.searchsorted(thresholds, elapsed, side='left'))))
    return x54, x59


def input_analysis() -> None:
    """最終学習入力と全評価記録呼出しの列分布を比較する。"""
    memory_guard()
    rows, design, phases = datasets(REFERENCE)
    assert tuple(json.loads((REFERENCE/'logs/r2_baselines_148_20260926/columns.json').read_text())) == D_COLUMNS
    thresholds = np.array(json.loads(Path('models/exchange_event_v1/manifest.json').read_text())['elapsed_thresholds'])
    with np.load(OUTPUT/'user_phase.npz') as u:
        phase, missing = u['phase'], u['missing']
    report = stats(design, D_COLUMNS, 'train_D')
    x54 = user_features(design, rows.A.to_numpy(), rows.sign.to_numpy(), phase, missing)
    report += stats(x54, USER_COLUMNS, 'train54')
    del x54
    x59 = g_features(design, rows.A.to_numpy(), rows.sign.to_numpy(),
        np.column_stack((phases['fill'], np.searchsorted(thresholds, phases['elapsed'], side='left'))))
    report += stats(x59, G_COLUMNS, 'train59')
    del x59
    counts, arrays = [], []
    for source in (*SET1, *SET2):
        memory_guard()
        r = extract_record(source)
        x54, x59 = model_features(r['raw'], r['p'], r['side'], r['elapsed'], thresholds)
        np.savez_compressed(OUT/f'recorded_{source}.npz', **r, x54=x54, x59=x59)
        counts.append(dict(source=source, calls=len(x54), raw_missing=int(np.isnan(r['raw']).sum()),
            saturation_nan=float(np.isnan(r['raw'][:, D_COLUMNS.index('saturation_chain_upper')]).mean())))
        arrays.append(r)
        print('record', source, len(x54), flush=True)
    raw = np.concatenate([r['raw'] for r in arrays])
    p, side, elapsed = (np.concatenate([r[k] for r in arrays]) for k in ('p','side','elapsed'))
    x54, x59 = model_features(raw, p, side, elapsed, thresholds)
    report += stats(raw, D_COLUMNS, 'record_raw')
    report += stats(x54, USER_COLUMNS, 'record54') + stats(x59, G_COLUMNS, 'record59')
    pd.DataFrame(report).to_csv(OUT/'input_columns.csv', index=False)
    pd.DataFrame(counts).to_csv(OUT/'record_counts.csv', index=False)
    save_json(OUT/'input_metadata.json', dict(thresholds=thresholds.tolist(), train_rows=len(rows),
        recorded_calls=len(p), train_opponent_missing=int(missing.sum()),
        note='記録static_callの復元。段2で新規生成する仮想着弾入力は別途captureで検証。'))


def weighted_cv(d: pd.DataFrame) -> None:
    """表示時刻の周辺分布を合わせる診断。新しいCVや採否には使わない。"""
    memory_guard()
    rows, _, phases = datasets(REFERENCE)
    cv = pd.read_csv(OUTPUT/'predictions.csv', usecols=['row_id','seed','label','V0','V3'])
    ids = cv.row_id.to_numpy()
    elapsed = phases['elapsed']
    duration = rows.groupby('game_id').t_sec.transform('max').to_numpy()-phases['start']
    progress = np.divide(elapsed, duration, out=np.zeros_like(elapsed), where=duration > 0)
    tb = np.minimum(np.searchsorted(TIME_EDGES, elapsed, side='right')-1, BINS-1)
    pb = np.minimum((progress*BINS).astype(int), BINS-1)
    l0, l3 = loss(cv.V0.to_numpy(), cv.label.to_numpy()), loss(cv.V3.to_numpy(), cv.label.to_numpy())
    results, cells = [], []
    base_game = 1/rows.groupby('game_id').row_id.transform('size').to_numpy()
    schemes = [('行等重み', np.ones(len(cv))), ('試合等重み', base_game[ids])]
    for target_name, target in [('全表示', d), ('G_fe表示', d[d.route == 'G_fe'])]:
        for axis, bins in [('time_bin', tb), ('progress_bin', pb)]:
            target_mass = np.bincount(target[axis], weights=target.weight, minlength=BINS)
            target_mass /= target_mass.sum()
            counts = np.bincount(bins[ids], minlength=BINS)
            weights = np.divide(target_mass, counts, out=np.zeros(BINS), where=counts > 0)[bins[ids]]
            schemes.append((target_name+'_'+axis, weights))
            for band in range(BINS):
                mask = bins[ids] == band
                cells.append(dict(target=target_name, axis=axis, bin=band, cv_rows=int(mask.sum()),
                    target_mass=target_mass[band], cv_mass=float(mask.mean()),
                    baseline=float(l0[mask].mean()), V3=float(l3[mask].mean()),
                    difference=float((l3-l0)[mask].mean())))
    for name, w in schemes:
        results.append(dict(weighting=name, baseline=np.average(l0, weights=w),
            V3=np.average(l3, weights=w), difference=np.average(l3-l0, weights=w),
            effective_rows=w.sum()**2/np.dot(w,w)))
    pd.DataFrame(results).to_csv(OUT/'cv_reweighted.csv', index=False)
    pd.DataFrame(cells).to_csv(OUT/'cv_time_cells.csv', index=False)


def main() -> None:
    """資源条件を守り、既存成果物には書き込まない。"""
    os.nice(max(0, 19-os.nice(0)))
    memory_guard()
    d = display_frames()
    summarize_display(d)
    weighted_cv(d)
    input_analysis()
    print('diagnostic analysis complete', flush=True)


if __name__ == '__main__':
    main()
