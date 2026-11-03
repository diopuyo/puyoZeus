"""採用済み学習行の火力列分布とS3木の分岐利得を監査する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from scripts._audit_firepower_1003_inventory import OUT, resources

REFERENCE = Path(_public_dev_text('<repo>/logs/r2_baselines_148_20260926'))
QUANTILES = (.1, .5, .9, .95, .99)
FIRE_EXACT = ('current_max_chain', 'diff_current_max_chain', 'own_sent', 'opp_sent',
              'net_own_minus_opp', 'own_more', 'opp_more', 'own_score_missing',
              'opp_score_missing', 'counter_margin_self', 'counter_margin_opp',
              'ojama_net_balance', 'ojama_forecast', 'ojama_net_balance_synced', 'ojama_margin')
RAW_CSV = Path(_public_dev_text('<repo>/data/verify/labeled_win_full148_2026-08-14/labeled_win_full148.csv'))
TIME_FLOAT32_TOLERANCE = .001


def fire_column(name: str) -> bool:
    """調査列を実際のmanifest名から選ぶ。"""
    return name in FIRE_EXACT or name.startswith('NF_ojama_') or (
        name.startswith(('own_k', 'opp_k', 'diff_k')))


def stats(values: np.ndarray) -> dict[str, Any]:
    """欠測と実値ゼロを区別して分布を算出する。"""
    good = values[np.isfinite(values)]
    q = np.quantile(good, QUANTILES) if len(good) else np.full(len(QUANTILES), np.nan)
    return dict(n=len(values), missing=int((~np.isfinite(values)).sum()),
                zero=int((values == 0).sum()), mean=float(good.mean()) if len(good) else np.nan,
                **{f'p{round(p*100)}': float(v) for p, v in zip(QUANTILES, q)})


def training() -> None:
    """既存採用行のみを読む。ティア未確認行を追加しない。"""
    manifest = json.loads(Path('models/exchange_event_v3/manifest.json').read_text())
    dcols = json.loads((REFERENCE / 'columns.json').read_text())
    rows, details = [], []
    for model in ('G_fe_R1s', 'S1', 'S3'):
        resources()
        if model == 'G_fe_R1s':
            x = np.load(REFERENCE / 'design.npy', mmap_mode='r')
            cols = dcols
            meta = pd.read_csv(REFERENCE / 'rows.csv', usecols=['video_id', 'game_id'])
            game = 'game_id'
        else:
            x = np.load(Path('logs/e15') / f'{model}_prime.npy', mmap_mode='r')
            cols = manifest['models'][model + '_prime_light']['columns']
            meta = pd.read_csv('logs/e15/rows.csv', usecols=['video_id', 'game'])
            game = 'game'
        details.append(dict(model=model, rows=len(x), videos=meta.video_id.nunique(),
                            games=len(meta.drop_duplicates(['video_id', game])),
                            columns=59 if model == 'G_fe_R1s' else len(cols)))
        for i, col in enumerate(cols):
            if fire_column(col):
                rows.append(dict(model=model, column=col, **stats(x[:, i])))
    pd.DataFrame(rows).to_csv(OUT / 'training_stats.csv', index=False)
    pd.DataFrame(details).to_csv(OUT / 'training_populations.csv', index=False)


def gains() -> None:
    """木の訓練内分岐利得。汎化寄与や因果効果とは解釈しない。"""
    manifest = json.loads(Path('models/exchange_event_v3/manifest.json').read_text())
    cols = manifest['models']['S3_prime_light']['columns']
    model = joblib.load('models/exchange_event_v3/S3_prime_light.joblib')
    gain, splits = np.zeros(len(cols)), np.zeros(len(cols), dtype=int)
    for stage in model._predictors:
        for tree in stage:
            for node in tree.nodes:
                if not node['is_leaf']:
                    i = node['feature_idx']
                    gain[i] += max(0., node['gain'])
                    splits[i] += 1
    result = pd.DataFrame(dict(column=cols, gain=gain, split_count=splits,
                              gain_fraction=gain/gain.sum(), fire=[fire_column(c) for c in cols]))
    result.sort_values('gain', ascending=False).to_csv(OUT / 's3_tree_gain.csv', index=False)


def training_imputation() -> None:
    """補完前CSVと採用行キーを対応させ、ゼロと欠測補完を区別する。"""
    resources()
    columns = [c for c in json.loads((REFERENCE / 'columns.json').read_text()) if fire_column(c)]
    keys = ['video_id', 'game_idx', 'frame', 'side']
    rows = pd.read_csv(REFERENCE / 'rows.csv')
    raw = pd.read_csv(RAW_CSV, usecols=keys+columns).set_index(keys)
    assert raw.index.is_unique
    lookup = rows.rename(columns={'source_side': 'side'}).copy()
    lookup['game_idx'] = lookup.game_id.str.rsplit(':', n=1).str[-1].astype(int)
    aligned = raw.reindex(pd.MultiIndex.from_frame(lookup[keys]))
    assert pd.MultiIndex.from_frame(lookup[keys]).isin(raw.index).all()
    missing = aligned.isna().to_numpy()
    design = np.load(REFERENCE / 'design.npy', mmap_mode='r')
    dcols = json.loads((REFERENCE / 'columns.json').read_text())
    np.testing.assert_allclose(aligned.fillna(0).to_numpy(), design[:, [dcols.index(c) for c in columns]], atol=1e-6)
    selected = initial_d_indices(rows)
    result = []
    for model, ids in [('G_fe_R1s', np.arange(len(rows))), ('S1', selected), ('S3', selected)]:
        for i, col in enumerate(columns):
            result.append(dict(model=model, column=col, n=len(ids), imputed_zero=int(missing[ids, i].sum())))
    pd.DataFrame(result).to_csv(OUT / 'training_imputation.csv', index=False)


def initial_d_indices(base: pd.DataFrame) -> np.ndarray:
    """E15の発火前最後のD行を復元し、保存済み行との一致を確認する。"""
    event = pd.read_csv('logs/e15/rows.csv')
    indices = np.full(len(event), -1, dtype=int)
    groups = dict(tuple(base.groupby('game_id', sort=False)))
    for (video, game), part in event.groupby(['video_id', 'game'], sort=False):
        candidates = groups[f'{video}:{game}']
        # E15は時刻ソートではなく、保存された行順のpast.iloc[-1]を使う。
        times = candidates.t_sec.to_numpy()
        positions = [np.flatnonzero(times < trigger)[-1] for trigger in part.trigger_sec]
        indices[part.index] = candidates.row_id.to_numpy()[positions]
    assert (indices >= 0).all()
    original = np.load(REFERENCE / 'design.npy', mmap_mode='r')
    stored = np.load('logs/e15/S1_prime.npy', mmap_mode='r')
    repair_d_indices(indices, original, stored, event, groups)
    np.testing.assert_allclose(original[indices], stored[:, :original.shape[1]], atol=1e-6)
    return indices


def repair_d_indices(indices: np.ndarray, original: np.ndarray, stored: np.ndarray,
                     event: pd.DataFrame, groups: dict) -> None:
    """浮動小数点時刻境界は、保存D全列との一致を追加条件として解決する。"""
    bad = np.flatnonzero(~np.isclose(original[indices], stored[:, :original.shape[1]], atol=1e-6).all(axis=1))
    evidence = []
    for i in bad:
        row = event.iloc[i]
        group = groups[f'{row.video_id}:{row.game}']
        match = np.isclose(original[group.row_id], stored[i, :original.shape[1]], atol=1e-6).all(axis=1)
        candidates = group[match & (group.t_sec <= row.trigger_sec+TIME_FLOAT32_TOLERANCE)]
        assert len(candidates), ('保存Dと一致する過去行なし', int(i))
        selected = candidates.iloc[-1]
        evidence.append(dict(row=int(i), selected=int(selected.row_id), alternatives=len(candidates),
                             time_minus_trigger=float(selected.t_sec-row.trigger_sec)))
        indices[i] = int(selected.row_id)
    (OUT / 'training_d_alignment.json').write_text(json.dumps(dict(
        exact_feature_verified_rows=len(event), time_tolerance=TIME_FLOAT32_TOLERANCE, repaired=evidence)))


def main() -> None:
    """軽量な学習行監査だけを実行する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    print(resources(), flush=True)
    training()
    gains()
    training_imputation()
    print('training audit complete', flush=True)


if __name__ == '__main__':
    main()
