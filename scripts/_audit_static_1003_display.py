"""本番R1sの表示時間、較正、変動、凍結、局所感度を監査する。"""
from __future__ import annotations
import argparse
import json
from typing import Any
from scripts._audit_static_1003_extract import OUT, EVAL, SOURCES, guard, save_json
from scripts._audit_static_1003_tables import add_axes, tables, summary, loss
import joblib
import numpy as np
import pandas as pd
from src.exchange_event_features import D_COLUMNS, G_COLUMNS

FRAME_SEC, TOLERANCE = 1 / 30, 1e-8
RANGE_TOLERANCE = 2e-6  # 学習float32と提供float64の丸め差を範囲外に数えない。
QUANTILES = (0, .5, .9, .95, .99, 1)
LARGE_AMOUNT, NEAR50, LARGE_MOVE = 12, .05, .15
NEIGHBORS, BOOTSTRAPS, BOOTSTRAP_SEED = 200, 5000, 1003
M0_INDEX = G_COLUMNS.index('m0_logit')
PHASE_START = G_COLUMNS.index('fill_early')
PHASE_STOP = G_COLUMNS.index('m0_logit_x_fill_early')


def source_frame(source: str, lr: Any, hgb: Any) -> tuple[pd.DataFrame, np.ndarray]:
    """成功した最後のG_fe通知と表示値の一致を全行で保証する。"""
    dest = OUT / 'replay' / source
    with np.load(dest / 'display.npz') as z:
        frame = pd.DataFrame({k: z[k] for k in z.files if z[k].ndim == 1})
    frame = frame.rename(columns={'t_sec': 't'})
    with np.load(dest / 'static_inputs.npz') as z:
        ids = np.searchsorted(z['t'], frame.t, side='right') - 1
        x = z['x'][np.maximum(ids, 0)]
        p, update = z['p'][np.maximum(ids, 0)], z['t'][np.maximum(ids, 0)]
    selected = frame.source.eq('G_fe').to_numpy()
    assert (ids[selected] >= 0).all()
    np.testing.assert_allclose(frame.display_p1[selected], p[selected], rtol=0, atol=TOLERANCE)
    np.testing.assert_allclose(frame.display_p1[selected], hgb.predict_proba(x[selected])[:, 1],
                               rtol=0, atol=TOLERANCE)
    old = pd.read_pickle(OUT / f'{source}_frames.pkl').set_index('t')
    frame['forecast'], frame['net'] = old.loc[frame.t, 'forecast'].to_numpy(), old.loc[frame.t, 'net'].to_numpy()
    frame['old_source'], frame['old_probability'] = old.loc[frame.t, 'source'].to_numpy(), old.loc[frame.t, 'display_p1'].to_numpy()
    frame['R1s'], frame['LR'] = frame.display_p1, lr.predict_proba(x)[:, 1]
    frame['updated_t'], frame['source_record'] = update, source
    frame['input_id'] = source + ':' + pd.Series(ids).astype(str)
    frame['seconds_since_update'] = frame.t - update
    frame['stable_zero'] = frame.state1.eq('STABLE') & frame.state2.eq('STABLE') & frame.forecast.eq(0) & frame.net.eq(0)
    return add_axes(frame, x), x


def population(available: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """手動勝敗115試合の時刻範囲で切り、表示欠落を補間しない。"""
    lr = joblib.load('models/exchange_event_v1/G_fe.joblib')
    hgb = joblib.load('models/exchange_event_v1_r1_hgb/G_fe.joblib')
    frames, arrays = [], []
    for source in SOURCES:
        if not (OUT / f'{source}_replay_done.json').exists():
            if available:
                continue
            raise FileNotFoundError(source)
        f, x = source_frame(source, lr, hgb)
        label_path = EVAL / ('set2/labels.json' if source.startswith('s') else 'labels.json')
        set_name = 'set2' if source.startswith('s') else 'set1'
        for game in json.loads(label_path.read_text())['games']:
            mask = (f.t >= game['start']) & (f.t < game['end'])
            g = f[mask].copy()
            if g.empty:
                continue
            g['game'], g['set'], g['game_no'] = f'{set_name}:{game["game"]}', set_name, game['game']
            g['y'], g['elapsed'] = int(game['winner'] == '1P'), g.t - game['start']
            g['dt'] = np.minimum(FRAME_SEC, game['end'] - g.t)
            frames.append(g)
            arrays.append(x[mask])
    result = pd.concat(frames, ignore_index=True)
    assert not result[['set', 't']].duplicated().any()
    x = np.concatenate(arrays)
    order = result.sort_values(['set', 't']).index.to_numpy()
    return result.loc[order].reset_index(drop=True), x[order]


def weight_static(f: pd.DataFrame) -> pd.DataFrame:
    """G_fe表示秒を各試合で正規化し、各試合の総重みを同一にする。"""
    g = f[f.source == 'G_fe'].copy()
    duration = g.groupby('game').dt.transform('sum')
    g['weight'] = g.dt / duration / g.game.nunique()
    return g


def slices(g: pd.DataFrame) -> None:
    """量差の大きい互角表示とセット差を明示する。"""
    results = []
    masks = {'all': np.ones(len(g), bool), 'stable_zero': g.stable_zero,
             'large_amount': abs(g.amount_diff) >= LARGE_AMOUNT,
             'large_amount_near50': (abs(g.amount_diff) >= LARGE_AMOUNT) & (abs(g.R1s - .5) <= NEAR50),
             'feedback5': (g['set'] == 'set2') & g.t.between(4240, 4247, inclusive='left')}
    for name, mask in masks.items():
        if np.any(mask):
            results.append(dict(slice=name, **summary(g[mask])))
    results += [dict(slice=name, **summary(part)) for name, part in g.groupby('set')]
    for amount in (4, 8, 12, 18):
        mask = abs(g.amount_diff) >= amount
        results.append(dict(slice=f'amount_at_least_{amount}', **summary(g[mask])))
    pd.DataFrame(results).to_csv(OUT / 'display_slices.csv', index=False)
    per_game = [dict(game=name, **summary(part)) for name, part in g.groupby('game')]
    pd.DataFrame(per_game).to_csv(OUT / 'display_games.csv', index=False)
    deltas = np.array([r['delta_LL'] for r in per_game])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.mean(rng.choice(deltas, (BOOTSTRAPS, len(deltas))), axis=1)
    save_json('display_uncertainty.json', dict(delta_LL=deltas.mean(),
        game_bootstrap_95=np.quantile(draws, [.025, .975]).tolist(),
        caveat='同一対戦者・同一動画内の試合相関、評価セットでのモデル選択の楽観は除去しない。'))


def episodes(f: pd.DataFrame, mask: Any, name: str) -> pd.DataFrame:
    """連続区間を試合・再生断点で分離し、全場面の時刻を残す。"""
    work = f.copy()
    work['flag'] = np.asarray(mask)
    split = work.game.ne(work.game.shift()) | work.game_idx.ne(work.game_idx.shift()) | work.t.diff().gt(FRAME_SEC + TOLERANCE) | work.flag.ne(work.flag.shift())
    work['segment'] = split.cumsum()
    rows = []
    for _, g in work[work.flag].groupby('segment'):
        rows.append(dict(game=g.game.iloc[0], start=g.t.iloc[0], end=g.t.iloc[-1] + g.dt.iloc[-1],
            seconds=g.dt.sum(), frames=len(g), R1s=g.R1s.mean(), LR=g.LR.mean(), outcome=g.y.iloc[0],
            amount_diff=g.amount_diff.mean(), chain_diff=g.chain_diff.mean(),
            stable_share=g.stable_zero.mean()))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f'{name}.csv', index=False)
    return result


def motion(f: pd.DataFrame, x: np.ndarray) -> None:
    """1秒差・隣接フレーム切替・同値保持と未評価保持を区別する。"""
    split = f.game.ne(f.game.shift()) | f.game_idx.ne(f.game_idx.shift()) | f.source.ne(f.source.shift()) | f.t.diff().gt(FRAME_SEC + TOLERANCE)
    f['segment'] = split.cumsum()
    changes, jumps, frozen = [], [], []
    for _, g in f.groupby('segment'):
        if g.source.iloc[0] != 'G_fe':
            continue
        ids = np.searchsorted(g.t.to_numpy(), g.t.to_numpy() - 1 + TOLERANCE, side='right') - 1
        mask = ids >= 0
        d = g[mask].copy()
        quiet = g.forecast.eq(0) & g.net.eq(0) & ~g.state1.isin(['CHAIN', 'OJAMA_FALL']) & ~g.state2.isin(['CHAIN', 'OJAMA_FALL'])
        cumulative = np.r_[0, np.cumsum(~quiet.to_numpy())]
        d['quiet_entire_second'] = (cumulative[np.flatnonzero(mask) + 1] - cumulative[ids[mask]]) == 0
        for col in ('R1s', 'display_adv'):
            d[col + '_delta_1s'] = g[col].to_numpy()[mask] - g[col].to_numpy()[ids[mask]]
        changed = abs(x[d.index] - x[g.index.to_numpy()[ids[mask]]]) > TOLERANCE
        d['changed_D'] = changed[:, :len(D_COLUMNS)].sum(axis=1)
        d['changed_M0'] = changed[:, M0_INDEX]
        d['changed_phase'] = changed[:, PHASE_START:PHASE_STOP].any(axis=1)
        d['reason_class'] = np.where(d.changed_D > 0, 'D_changed',
            np.where(d.changed_M0, 'M0_only', np.where(d.changed_phase, 'phase_only', 'identical_input')))
        changes.append(d)
        groups = g.R1s.diff().abs().gt(TOLERANCE).cumsum()
        for _, h in g.groupby(groups):
            frozen.append(dict(game=h.game.iloc[0], start=h.t.iloc[0], seconds=h.dt.sum(),
                updates=h.updated_t.nunique(), max_age=h.seconds_since_update.max(),
                stable_share=h.stable_zero.mean()))
    save_movement(pd.concat(changes))
    for idx in np.flatnonzero(split.to_numpy())[1:]:
        a, b = f.iloc[idx - 1], f.iloc[idx]
        if a.game == b.game and b.t - a.t < FRAME_SEC + TOLERANCE and 'G_fe' in (a.source, b.source):
            jumps.append(dict(game=b.game, t=b.t, before=a.source, after=b.source,
                raw_delta=b.R1s - a.R1s, adv_delta=b.display_adv - a.display_adv,
                reset=a.game_idx != b.game_idx or a.source_record != b.source_record))
    pd.DataFrame(jumps).to_csv(OUT / 'switch_jumps.csv', index=False)
    pd.DataFrame(frozen).to_csv(OUT / 'freezes.csv', index=False)
    episodes(f, f.source.eq('G_fe') & f.seconds_since_update.ge(1), 'stale_episodes')


def save_movement(d: pd.DataFrame) -> None:
    """変動の分布と理由別原表をまとめて保存する。"""
    rows = [dict(scope=scope, metric=col, rows=len(part), **dict(zip(map(str, QUANTILES), np.quantile(abs(part[col]), QUANTILES))))
            for scope, part in [('all_G_fe', d), ('no_chain_no_forecast', d[d.quiet_entire_second])]
            for col in ('R1s_delta_1s', 'display_adv_delta_1s') if len(part)]
    pd.DataFrame(rows).to_csv(OUT / 'motion_quantiles.csv', index=False)
    d[abs(d.R1s_delta_1s) >= LARGE_MOVE].to_csv(OUT / 'large_moves.csv', index=False)
    d[d.quiet_entire_second & (abs(d.R1s_delta_1s) >= LARGE_MOVE)].to_csv(OUT / 'large_moves_quiet.csv', index=False)
    reasons = d[abs(d.R1s_delta_1s) >= LARGE_MOVE].groupby('reason_class').agg(
        frames=('t', 'size'), games=('game', 'nunique'), max_change=('R1s_delta_1s', lambda v: abs(v).max()))
    reasons.to_csv(OUT / 'large_move_reasons.csv')
    all_reasons = d.groupby('reason_class').agg(frames=('t', 'size'), games=('game', 'nunique'),
        p95=('R1s_delta_1s', lambda v: abs(v).quantile(.95)), maximum=('R1s_delta_1s', lambda v: abs(v).max()))
    all_reasons.to_csv(OUT / 'motion_reasons_all.csv')
    d[(d.changed_D == 0) & (abs(d.R1s_delta_1s) >= .05)].to_csv(OUT / 'no_D_change_moves.csv', index=False)


def out_of_distribution(g: pd.DataFrame, x: np.ndarray) -> None:
    """59列の経験範囲逸脱を計測。多変量の未知盤面率とは区別する。"""
    b = pd.read_csv(OUT / 'training_bounds.csv')
    assert list(b.column) == list(G_COLUMNS)
    outside = (x < b['min'].to_numpy() - RANGE_TOLERANCE) | (x > b['max'].to_numpy() + RANGE_TOLERANCE)
    tail = (x < b.p01.to_numpy() - RANGE_TOLERANCE) | (x > b.p99.to_numpy() + RANGE_TOLERANCE)
    rows = [dict(column=name, outside_seconds=np.dot(outside[:, i], g.dt),
                 outside_weight=np.dot(outside[:, i], g.weight), tail_weight=np.dot(tail[:, i], g.weight))
            for i, name in enumerate(G_COLUMNS)]
    pd.DataFrame(rows).to_csv(OUT / 'ood_columns.csv', index=False)
    save_json('ood_summary.json', dict(rows=len(g), seconds=g.dt.sum(),
        any_outside_seconds=float(np.dot(outside.any(axis=1), g.dt)),
        any_outside_weight=float(np.dot(outside.any(axis=1), g.weight)),
        nonempty_outside_seconds=float(np.dot(outside.any(axis=1) & (x[:, 1] > 0), g.dt)),
        tolerance=RANGE_TOLERANCE,
        any_tail_weight=float(np.dot(tail.any(axis=1), g.weight))))


def local_sensitivity(g: pd.DataFrame, x: np.ndarray) -> None:
    """近傍200行の各列20/50/80百分位に置換する非加法的局所部分依存。"""
    train = np.load(OUT / 'training_sample.npy')
    scale = np.quantile(train, .75, axis=0) - np.quantile(train, .25, axis=0)
    scale = np.where(scale > TOLERANCE, scale, 1)
    model = joblib.load('models/exchange_event_v1_r1_hgb/G_fe.joblib')
    feedback = g['set'].eq('set2') & g.t.between(4240, 4247, inclusive='left')
    selected = g[feedback | ((abs(g.amount_diff) >= LARGE_AMOUNT) & (abs(g.R1s - .5) <= NEAR50))]
    selected = selected.drop_duplicates('input_id')
    # 指摘5は全入力、その他は試合ごとの先頭1場面に限定して計算範囲を明示する。
    selected = pd.concat([selected[feedback.loc[selected.index]], selected[~feedback.loc[selected.index]].groupby('game').head(1)])
    rows = []
    positions = pd.Series(np.arange(len(g)), index=g.index)
    for idx, row in selected.iterrows():
        values = x[positions.loc[idx]]
        distance = np.mean(((train - values) / scale) ** 2, axis=1)
        neighbors = train[np.argpartition(distance, NEIGHBORS - 1)[:NEIGHBORS]]
        replacements = np.quantile(neighbors, [.2, .5, .8], axis=0)
        probe = np.tile(values, (len(G_COLUMNS) * 3, 1))
        for j in range(len(G_COLUMNS)):
            probe[j * 3:(j + 1) * 3, j] = replacements[:, j]
        predicted = model.predict_proba(probe)[:, 1].reshape(-1, 3)
        for j, name in enumerate(G_COLUMNS):
            rows.append(dict(game=row.game, t=row.t, column=name, actual_value=values[j], actual_p=row.R1s,
                q20=replacements[0, j], q50=replacements[1, j], q80=replacements[2, j],
                p20=predicted[j, 0], p50=predicted[j, 1], p80=predicted[j, 2],
                actual_minus_median=row.R1s - predicted[j, 1], feedback5=bool(feedback.loc[idx])))
    pd.DataFrame(rows).to_csv(OUT / 'local_sensitivity.csv', index=False)


def main() -> None:
    """再生完了を検査して監査の全表を保存する。"""
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--available', action='store_true')
    args = parser.parse_args()
    f, x = population(args.available)
    g = weight_static(f)
    tables(g, 'display')
    slices(g)
    motion(f, x)
    episodes(f, f.source.eq('G_fe') & (abs(f.amount_diff) >= LARGE_AMOUNT) & (abs(f.R1s - .5) <= NEAR50), 'amount_near50_episodes')
    out_of_distribution(g, x[g.index])
    local_sensitivity(g, x[g.index])
    f.to_pickle(OUT / 'display_all.pkl')
    g.to_pickle(OUT / 'display_static.pkl')
    np.save(OUT / 'display_static_x.npy', x[g.index])
    save_json('display_population.json', dict(frames=len(f), games=f.game.nunique(),
        seconds=f.dt.sum(), gfe_frames=len(g), gfe_games=g.game.nunique(), gfe_seconds=g.dt.sum(),
        stable_zero_seconds=g[g.stable_zero].dt.sum(), complete=not args.available,
        old_route_equal_rows=int((f.source == f.old_source).sum()),
        old_lr_equal_gfe_rows=int((abs(g.LR - g.old_probability) <= TOLERANCE).sum())))
    print(summary(g), flush=True)


if __name__ == '__main__':
    main()
