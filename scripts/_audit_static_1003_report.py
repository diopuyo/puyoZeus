"""静止勝率監査の全集計表と場面原表への索引を生成する。"""
from __future__ import annotations
import argparse
import html
import json
from scripts._audit_static_1003_extract import OUT, guard
import numpy as np
import pandas as pd

CORE = ('display_strata', 'display_calibration', 'display_slices', 'display_games',
        'cv_strata', 'cv_calibration', 'cv_seeds', 'fire_quantiles', 'fire_high_difference',
        'motion_quantiles', 'large_move_reasons', 'motion_reasons_all', 'switch_summary',
        'freeze_summary', 'ood_columns', 'training_bounds')


def additional_tables() -> None:
    """切替先・凍結種別・指摘5を読みやすい集計表へまとめる。"""
    switches = pd.read_csv(OUT / 'switch_jumps.csv')
    rows = []
    for (before, after, reset), g in switches.groupby(['before', 'after', 'reset']):
        rows.append(dict(before=before, after=after, reset=reset, switches=len(g), games=g.game.nunique(),
            raw_p95=abs(g.raw_delta).quantile(.95), raw_max=abs(g.raw_delta).max(),
            adv_p95=abs(g.adv_delta).quantile(.95), adv_max=abs(g.adv_delta).max()))
    pd.DataFrame(rows).to_csv(OUT / 'switch_summary.csv', index=False)
    f = pd.read_csv(OUT / 'freezes.csv')
    rows = []
    for name, g in [('all', f), ('recomputed_same_value', f[f.updates > 1]), ('no_recompute', f[f.updates == 1])]:
        rows.append(dict(kind=name, episodes=len(g), games=g.game.nunique(),
            median=g.seconds.median(), p95=g.seconds.quantile(.95), maximum=g.seconds.max(),
            max_input_age=g.max_age.max()))
    pd.DataFrame(rows).to_csv(OUT / 'freeze_summary.csv', index=False)
    feedback_timeline()


def feedback_timeline() -> None:
    """指摘5を元動画1秒ごとに区切り、後半の火力立ち上がりを隠さない。"""
    f = pd.read_pickle(OUT / 'display_static.pkl')
    x = np.load(OUT / 'display_static_x.npy')
    mask = f['set'].eq('set2') & f.t.between(4240, 4247, inclusive='left')
    selected = f[mask].copy()
    selected['p1_total'] = np.rint(x[mask, 1] * 72)
    selected['p2_total'] = selected.p1_total - selected.amount_diff
    selected['p1_chain'] = np.rint(x[mask, 7] * 19)
    selected['p2_chain'] = selected.p1_chain - selected.chain_diff
    cols = ['R1s', 'LR', 'p1_total', 'p2_total', 'p1_chain', 'p2_chain']
    result = selected.groupby(np.floor(selected.t))[cols].mean()
    fire = pd.read_csv(OUT / 'feedback5_fire.csv')
    cols = ['p1_k1', 'p2_k1', 'p1_k5', 'p2_k5']
    result = result.join(fire.groupby(np.floor(fire.t))[cols].mean())
    result['stable_frames'] = fire.groupby(np.floor(fire.t)).size()
    result.to_csv(OUT / 'feedback5_timeline.csv')


def preview() -> None:
    """指摘5の既存入力による仮計算を表示し、完成表と混同しない。"""
    f = pd.read_pickle(OUT / 's1_frames.pkl')
    f = f[(f.t >= 4240) & (f.t < 4247)]
    print('指摘5・再生完了前の入力仮計算', f[['display_p1', 'p_lr', 'p_r1_calc']].agg(['min', 'mean', 'max']).to_string())
    with np.load(OUT / 's1_inputs.npz') as z:
        x = z['x'][f['call'].to_numpy()]
    from src.exchange_event_features import G_COLUMNS
    cols = ['board_puyo_total', 'diff_board_puyo_total', 'current_max_chain', 'diff_current_max_chain', 'm0_logit']
    print(pd.DataFrame(x, columns=G_COLUMNS)[cols].agg(['min', 'mean', 'max']).to_string())


def compact() -> None:
    """最終返答用の主要な実測値を明示する。"""
    for name in ('display_population', 'fire_population', 'ood_summary', 'display_uncertainty'):
        print(name, (OUT / f'{name}.json').read_text())
    print(pd.read_csv(OUT / 'display_slices.csv').to_string(index=False))
    f = pd.read_csv(OUT / 'feedback5_fire.csv')
    cols = ['p1_chain', 'p2_chain', 'p1_immediate', 'p2_immediate', 'p1_ignite', 'p2_ignite',
            'p1_k1', 'p2_k1', 'p1_k5', 'p2_k5', 'amount_diff', 'R1s', 'LR']
    print('指摘5 STABLE', f[cols].agg(['min', 'mean', 'max']).to_string())
    local = pd.read_csv(OUT / 'local_sensitivity.csv')
    local = local[local.feedback5]
    impact = local.groupby('column').actual_minus_median.mean().sort_values(key=abs, ascending=False)
    print('指摘5 局所感度', impact.head(12).to_string())
    for name in ('freezes', 'switch_jumps', 'stale_episodes', 'amount_near50_episodes'):
        f = pd.read_csv(OUT / f'{name}.csv')
        print(name, len(f), f.select_dtypes('number').quantile([.5, .95, 1]).to_string())


def all_tables() -> None:
    """大きな場面原表はCSV、集計表は省略せずHTMLへ埋め込む。"""
    parts = ['<!doctype html><meta charset="utf-8"><title>静止勝率監査 全表</title>',
             '<style>body{font:14px sans-serif;margin:24px;color:#222}table{border-collapse:collapse;white-space:nowrap}td,th{border:1px solid #bbb;padding:5px}section{overflow:auto;margin:32px 0}h1{font-size:24px}</style>',
             '<h1>静止勝率監査 全表 — 802efbc / R1s + D-v2h2b</h1>',
             '<p>合否・採否を判定しない。確率は1P。表示表はG_fe表示秒を試合内で正規化した試合等重み。CVは3seedのOOFを反復平均。U_slowは小さい方、U_legacy_maxは既存実装の大きい方。LLは自然対数。</p>']
    for name in (*CORE, 'feedback5_timeline'):
        f = pd.read_csv(OUT / f'{name}.csv')
        parts += [f'<section><h2>{html.escape(name)}</h2>', f.to_html(index=False, float_format=lambda x: f'{x:.7g}'), '</section>']
    local = pd.read_csv(OUT / 'local_sensitivity.csv')
    local = local[local.feedback5].groupby('column')[['actual_minus_median', 'p20', 'p50', 'p80']].mean().reset_index()
    local.to_csv(OUT / 'feedback5_local_summary.csv', index=False)
    parts += ['<section><h2>指摘5 局所部分依存（非加法的、因果寄与ではない）</h2>', local.to_html(index=False), '</section>']
    parts += ['<h2>全原表（場面は元動画秒）</h2><ul>']
    for path in sorted(OUT.glob('*.csv')):
        parts.append(f'<li><a href="{path.name}">{path.name}</a> ({path.stat().st_size:,} bytes)</li>')
    parts.append('</ul>')
    (OUT / 'ALL_TABLES.html').write_text('\n'.join(parts), encoding='utf-8')


def main() -> None:
    """出力完成後に全表を整形し、元の数値はCSVに保持する。"""
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    if args.preview:
        preview()
        return
    additional_tables()
    all_tables()
    compact()


if __name__ == '__main__':
    main()
