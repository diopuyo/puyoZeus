"""S3着弾経路の内部G_feと表示保持をイベント原票で突合する。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scripts._diag_gfe_stage2_analyze import (
    OUT, STAGE2, display_frames, grouped, calibration, stats, model_features, loss,
)
from scripts.gfe_phase_stage1_20261002 import memory_guard, save_json
from src.exchange_event_features import D_COLUMNS, G_COLUMNS
from src.exchange_event_user_phase import USER_COLUMNS

M0_INDEX, PROBABILITY_BINS, HASH_DECIMALS = len(D_COLUMNS), 10, 7


def events(source: str, mode: str) -> pd.DataFrame:
    """同時刻の最後の着弾値を表示との照合候補にする。"""
    rows = []
    with (STAGE2/mode/'replay'/source/'events.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line)
            for r in record['values']:
                if r['source'] == 'S3_landing':
                    rows.append(dict(t=r['t_sec'], event_p=r['p1'], base=r['base_p1'],
                        gfe=r['gfe_p1'], incoming1=r['incoming'][0], incoming2=r['incoming'][1],
                        response=bool(r.get('response_selected', False)),
                        gfe_no_response=r.get('gfe_no_response_p1', np.nan),
                        gfe_response=r.get('gfe_response_p1', np.nan)))
    return pd.DataFrame(rows).sort_values('t', kind='stable').drop_duplicates('t', keep='last')


def landing() -> None:
    """合成前のS3が不変かを同じ表示時刻で直接確かめる。"""
    d = display_frames()
    parts = []
    for source, frame in d[d.route == 'S3_landing'].groupby('source'):
        merged = frame.sort_values('t')
        for mode, suffix in [('baseline','0'), ('V3','3')]:
            e = events(source, mode)
            e = e.rename(columns={c: c+suffix for c in e if c != 't'})
            merged = pd.merge_asof(merged, e, on='t', direction='backward')
        assert np.allclose(merged.event_p0, merged.p0, atol=1e-12)
        assert np.allclose(merged.event_p3, merged.p3, atol=1e-12)
        parts.append(merged)
    combined = pd.concat(parts, ignore_index=True)
    assert np.array_equal(combined.base0, combined.base3)
    def band(values: pd.Series) -> np.ndarray:
        return np.minimum((values*PROBABILITY_BINS).astype(int), PROBABILITY_BINS-1)
    combined['gfe_band0'], combined['gfe_band3'] = band(combined.gfe0), band(combined.gfe3)
    combined['response'] = combined.response0
    for name, keys in [('landing_base', ['set']), ('landing_gfe_bands', ['gfe_band3']),
                       ('landing_response', ['response'])]:
        grouped(combined, keys).to_csv(OUT/f'{name}.csv', index=False)
    calibration(combined).to_csv(OUT/'landing_calibration.csv', index=False)
    save_json(OUT/'landing_summary.json', dict(frames=len(combined),
        display_matches=True, base_s3_identical=True,
        raw_gfe_delta_mean=float(np.average(combined.gfe3-combined.gfe0, weights=combined.weight)),
        raw_gfe_LL_baseline=float(np.average(loss(combined.gfe0, combined.y), weights=combined.weight)),
        raw_gfe_LL_V3=float(np.average(loss(combined.gfe3, combined.y), weights=combined.weight)),
        model_gfe_quantiles={m: np.quantile(combined['gfe'+m], [0,.01,.25,.5,.75,.99,1]).tolist() for m in ('0','3')}))
    combined.to_csv(OUT/'landing_display_detail.csv', index=False)


def captured() -> None:
    """実際のモデル境界入力と記録の入力との差を別表にする。"""
    report, checks = [], []
    for path in sorted((OUT/'capture').glob('*/inputs.npz')):
        with np.load(path) as cap:
            source = path.parent.name
            for k, cols in [('raw', D_COLUMNS), ('x54', USER_COLUMNS), ('x59', G_COLUMNS)]:
                report += stats(cap[k], cols, 'actual_'+source+'_'+k)
            if 'caller' in cap:
                for caller in np.unique(cap['caller']):
                    mask = cap['caller'] == caller
                    report += stats(cap['x54'][mask], USER_COLUMNS, 'actual_'+source+'_'+caller)
            with np.load(OUT/f'recorded_{source}.npz') as rec:
                # D同一性とM0値の同一性を混同しない。記録M0は旧収集器の値。
                width = len(D_COLUMNS)*np.dtype('<f8').itemsize
                lookup = set(np.nan_to_num(rec['raw']).round(HASH_DECIMALS).astype('<f8').view(f'V{width}').ravel().tolist())
                keys = np.nan_to_num(cap['raw']).round(HASH_DECIMALS).astype('<f8').view(f'V{width}').ravel().tolist()
                checks.append(dict(source=source, calls=len(keys), matched_D=sum(k in lookup for k in keys),
                    actual_m0_zero=float((cap['x54'][:,M0_INDEX] == 0).mean()),
                    recorded_m0_zero=float((rec['x54'][:,M0_INDEX] == 0).mean())))
    pd.DataFrame(report).to_csv(OUT/'actual_input_columns.csv', index=False)
    pd.DataFrame(checks).to_csv(OUT/'actual_record_comparison.csv', index=False)


def main() -> None:
    """診断表だけを更新する。"""
    os.nice(max(0,19-os.nice(0)))
    memory_guard()
    landing()
    captured()


if __name__ == '__main__':
    main()
