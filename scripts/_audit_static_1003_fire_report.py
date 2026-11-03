"""静止全フレーム火力差の分位、互角率、場面一覧を集計する。"""
from __future__ import annotations
import gzip
import json
from scripts._audit_static_1003_extract import OUT, BASE, SOURCES, guard, save_json
from scripts._audit_static_1003_tables import summary
from scripts._audit_static_1003_display import episodes, NEAR50
import numpy as np
import pandas as pd

METRICS = ('chain', 'immediate', 'ignite', 'k1', 'k2', 'k3', 'k4', 'k5')
HIGH_QUANTILE = .8


def diagnose_missing(missing: pd.DataFrame) -> list[dict]:
    """取りこぼしを黙って除外せず、入力状態と保存盤面を照合する。"""
    reasons = []
    for source, g in missing.groupby('source_record'):
        wanted = set(g.t)
        with gzip.open(BASE / f'collect/records/{source}.jsonl.gz', 'rt') as stream:
            for line in stream:
                r = json.loads(line)
                if r['kind'] != 'update' or r['args']['tuple'][3] not in wanted:
                    continue
                args = r['args']['tuple']
                sides = [args[0]['namespace'][s]['namespace'] for s in ('p1', 'p2')]
                assert any(s['confirmed_board'] is None for s in sides)
                reasons.append(dict(source=source, t=args[3], reason='confirmed_board_missing'))
    assert len(reasons) == len(missing)
    return reasons


def main() -> None:
    """密な表示時間を母数にし、長い試合の過大重みを除く。"""
    guard()
    static = pd.read_pickle(OUT / 'display_static.pkl')
    source = pd.concat([pd.read_csv(OUT / f'{s}_fire_dense.csv', float_precision='round_trip')
                        for s in SOURCES], ignore_index=True)
    desired = static[static.stable_zero].copy()
    f = desired.merge(source, on=['source_record', 't'], how='left', validate='one_to_one', indicator=True)
    missing = f[f._merge != 'both']
    missing.to_csv(OUT / 'fire_missing.csv', index=False)
    save_json('fire_exclusions.json', diagnose_missing(missing))
    f = f[f._merge == 'both'].copy()
    f['weight'] = f.dt / f.groupby('game').dt.transform('sum') / f.game.nunique()
    results, high = [], []
    for metric in METRICS:
        sign = -1 if metric == 'ignite' else 1
        f[metric + '_diff'] = sign * (f['p1_' + metric] - f['p2_' + metric])
        values = f[metric + '_diff']
        groups = pd.qcut(values, 5, duplicates='drop')
        for band, g in f.groupby(groups, observed=True):
            results.append(dict(metric=metric, band=str(band), min=g[metric + '_diff'].min(),
                max=g[metric + '_diff'].max(), mean_diff=np.average(g[metric + '_diff'], weights=g.weight), **summary(g)))
        # 全行分位の固定閾値。0だと大差にならないため、厳密に0より大きい差だけ数える。
        cutoff = np.quantile(abs(values), HIGH_QUANTILE)
        mask = (abs(values) >= cutoff) & (abs(values) > 0)
        large = f[mask]
        high.append(dict(metric=metric, cutoff=cutoff, **summary(large),
            near50_seconds=large[abs(large.R1s - .5) <= NEAR50].dt.sum(),
            near50_LR_seconds=large[abs(large.LR - .5) <= NEAR50].dt.sum()))
        episodes(f, mask & (abs(f.R1s - .5) <= NEAR50), f'fire_{metric}_near50_episodes')
    pd.DataFrame(results).to_csv(OUT / 'fire_quantiles.csv', index=False)
    pd.DataFrame(high).to_csv(OUT / 'fire_high_difference.csv', index=False)
    f.to_pickle(OUT / 'fire_joined.pkl')
    feedback = f[(f['set'] == 'set2') & f.t.between(4240, 4247, inclusive='left')]
    feedback.to_csv(OUT / 'feedback5_fire.csv', index=False)
    save_json('fire_population.json', dict(frames=len(f), games=f.game.nunique(), seconds=f.dt.sum(),
        unique_inputs=f.key.nunique(), known_next_p1=float(f.p1_known_next.mean()),
        known_next_p2=float(f.p2_known_next.mean()), **{'summary': summary(f)}))
    print('火力全フレーム集計完了', len(f), flush=True)


if __name__ == '__main__':
    main()
