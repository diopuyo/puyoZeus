"""診断集計の足し戻し・入力列順・再生一致を検証する。"""
from __future__ import annotations

import ast
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scripts._diag_gfe_stage2_analyze import OUT, STAGE2, GAMES, display_frames, calibration, stats, model_features
from scripts.gfe_phase_stage1_20261002 import memory_guard, save_json
from scripts.gfe_phase_stage2_replay_20261002 import SET1, SET2
from src.exchange_event_features import D_COLUMNS, G_COLUMNS
from src.exchange_event_user_phase import USER_COLUMNS

BOOTSTRAPS, SEED, EXPECTED_SOURCES = 10000, 0, 14
M0_INDEX = len(D_COLUMNS)


def route_intervals() -> None:
    """試合単位の寄与を同じ復元抽出で比較する。"""
    d = display_frames()
    d['contribution'] = d.delta*d.weight
    matrix = d.pivot_table(index=['set','game'], columns='route', values='contribution', aggfunc='sum', fill_value=0)*GAMES
    index = np.random.default_rng(SEED).integers(0, len(matrix), (BOOTSTRAPS, len(matrix)))
    result = []
    for column in matrix:
        v = matrix[column].to_numpy()
        ci = np.quantile(v[index].mean(axis=1), [.025,.975])
        result.append(dict(route=column, contribution=v.mean(), lower=ci[0], upper=ci[1]))
    pd.DataFrame(result).to_csv(OUT/'route_intervals.csv', index=False)
    raw = pd.read_csv(OUT/'landing_display_detail.csv')
    raw['p0'], raw['p3'] = raw.gfe0, raw.gfe3
    calibration(raw).to_csv(OUT/'landing_raw_calibration.csv', index=False)


def verify_captures() -> None:
    """54/59列を独立に再構築し、全再生の保存値同一性を確かめる。"""
    thresholds = np.array(json.loads(Path('models/exchange_event_v1/manifest.json').read_text())['elapsed_thresholds'])
    rows, raw_arrays, x54_arrays, x59_arrays, callers = [], [], [], [], []
    for source in (*SET1, *SET2):
        memory_guard()
        directory = OUT/'capture'/source
        done = json.loads((directory/'DONE.json').read_text())
        assert all(done['byte_identity'].values())
        with np.load(directory/'inputs.npz') as data:
            raw, x54, x59 = data['raw'], data['x54'], data['x59']
            if done.get('scope','full') == 'full':
                original = json.loads((STAGE2/'baseline/replay'/source/'DONE.json').read_text())
                assert len(raw) == original['gfe_calls'], source
            from scipy.special import expit
            a, b = model_features(raw, expit(x54[:,M0_INDEX]), np.zeros(len(raw)), data['elapsed'], thresholds)
            np.testing.assert_allclose(a, x54, rtol=0, atol=1e-12)
            np.testing.assert_allclose(b, x59, rtol=0, atol=1e-12)
            rows.append(dict(source=source, calls=len(raw), identity=True, seconds=done['elapsed_seconds'],
                scope=done.get('scope','full')))
            raw_arrays.append(raw); x54_arrays.append(x54); x59_arrays.append(x59)
            if 'caller' in data:
                callers.append(data['caller'])
            else:
                with np.load(OUT/'capture_V3'/source/'inputs.npz') as tagged:
                    np.testing.assert_array_equal(tagged['raw'],raw)
                    callers.append(tagged['caller'])
    assert len(rows) == EXPECTED_SOURCES
    report = stats(np.concatenate(raw_arrays), D_COLUMNS, 'actual_raw')
    x54, x59, caller = np.concatenate(x54_arrays), np.concatenate(x59_arrays), np.concatenate(callers)
    report += stats(x54, USER_COLUMNS, 'actual54')+stats(x59, G_COLUMNS, 'actual59')
    for name in np.unique(caller):
        report += stats(x54[caller == name], USER_COLUMNS, 'actual_'+name)
    pd.DataFrame(report).to_csv(OUT/'all_actual_columns.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT/'capture_verification.csv', index=False)
    save_json(OUT/'INPUT_SCHEMA.json', dict(raw=list(D_COLUMNS), x54=list(USER_COLUMNS),
        x59=list(G_COLUMNS), context=['t_sec','game_idx'],
        training='段2最終学習のrows.A（seed0外側OOF）', thresholds=thresholds.tolist()))
    save_json(OUT/'VERIFICATION.json', dict(captures=len(rows), calls=len(x54),
        transformations=True, full_gfe_call_counts_match=True,
        full_byte_identity=sum(r['scope'] == 'full' for r in rows),
        prefix_identity=sum(r['scope'] != 'full' for r in rows),
        memory_guard='各updateで空き11GiB以上を要求', nice=19))


def verify_candidate_inputs() -> None:
    """V3側の再生でもモデル呼出しの時刻・D・54/59列が同じか確認する。"""
    directory = OUT/'capture_V3/c1'
    done = json.loads((directory/'DONE.json').read_text())
    assert all(done['byte_identity'].values())
    with np.load(directory/'inputs.npz') as candidate, np.load(OUT/'capture/c1/inputs.npz') as baseline:
        for name in ('context','elapsed','raw','x54','x59'):
            np.testing.assert_array_equal(candidate[name],baseline[name])
    save_json(OUT/'V3_INPUT_IDENTITY.json', dict(source='c1', calls=done['calls'],
        exact_arrays=['context','elapsed','raw','x54','x59'], stage2_display_events_byte_identity=True))


def verify_totals() -> None:
    """試合・時間・経路の表が元の差に戻ることを確認する。"""
    expected = json.loads((OUT/'display_summary.json').read_text())['difference']
    for name in ('games','routes','time','progress','route_time','route_transitions'):
        frame = pd.read_csv(OUT/f'{name}.csv')
        assert abs(frame.contribution.sum()-expected) < 1e-12, name
    assert len(pd.read_csv(OUT/'games.csv')) == GAMES
    for path in Path('scripts').glob('_diag_gfe_stage2_*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno-node.lineno+1
                assert length <= 50, (path.name,node.name,length)
                assert node.returns is not None, (path.name,node.name,'戻り値型なし')
    for name in ('RESOURCE_CHECK.json','S2_SCOPE.json'):
        path = OUT/name
        if path.exists():
            save_json(path,json.loads(path.read_text(encoding='utf-8-sig')))
    print('足し戻し・構文検査OK', flush=True)


def main() -> None:
    """すべて完了した後だけ全入力の検証を確定する。"""
    os.nice(max(0,19-os.nice(0)))
    memory_guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--tables-only', action='store_true')
    args = parser.parse_args()
    route_intervals()
    verify_totals()
    if not args.tables_only:
        verify_captures()
        verify_candidate_inputs()


if __name__ == '__main__':
    main()
