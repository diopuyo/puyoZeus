"""火力監査の集計表を再現する。採否・合否判定は作らない。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from scripts._audit_firepower_1003_inventory import OUT, resources
from scripts._audit_firepower_1003_extract import FPS, NAMES, read_jsonl
from scripts._audit_firepower_1003_training import fire_column, stats

Q = (.5, .9, .95, .99)
PROXY_OJAMA = 30
CONFIDENCE = .8
SCENES = ((1, 2, 210., 218.), (1, 13, 943., 951.), (1, 26, 1725., 1735.),
          (1, 43, 2748., 2763.), (2, 11, 4240., 4247.),
          (2, 11, 4261., 4273.), (2, 41, 6049., 6061.))


def save(name: str, rows: Any) -> pd.DataFrame:
    """全表をUTF-8 CSVで保存する。"""
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / f'{name}.csv', index=False)
    return frame


def errors(chains: pd.DataFrame) -> None:
    """観測値との相違を集計する。OCRを人手真値とみなさない。"""
    rows = []
    corroborated = chains[(chains.first_obs_count == 1) & chains.end_confirmed.fillna(False)
                         & ((chains.display_score-chains.score_before-chains.obs_formula_score).abs() <= 40)]
    for group, part in [('all', chains), ('both_chain_at_fire', chains[chains.both_chain]),
                        ('observed_7plus', chains[chains.obs_count >= 7]),
                        ('formula_display_agree', corroborated)]:
        for name, pred, truth in (
            ('count_vs_observed', 'pred_count', 'obs_count'),
            ('score_vs_formula', 'pred_score', 'formula_total'),
            ('score_vs_raw_formula', 'pred_score', 'obs_formula_score'),
            ('score_vs_finalized_OCR', 'pred_score', 'independent_score'),
            ('end_time_vs_signal', 'predicted_end', 'observed_end')):
            valid = part[[pred, truth]].dropna()
            diff = valid[pred]-valid[truth]
            vals = np.quantile(diff.abs(), Q) if len(diff) else np.full(len(Q), np.nan)
            rows.append(dict(group=group, metric=name, total=len(part), n=len(diff),
                exact=int((diff.abs() < 1e-6).sum()), bias=diff.mean(),
                **{f'abs_p{round(q*100)}': v for q, v in zip(Q, vals)}))
        valid = part[['pred_score', 'formula_total', 'rate']].dropna()
        delta = np.floor(valid.pred_score/valid.rate)-np.floor(valid.formula_total/valid.rate)
        rows.append(dict(group=group, metric='sent_vs_formula', total=len(part), n=len(delta),
            exact=int((delta == 0).sum()), bias=delta.mean(),
            **{f'abs_p{round(q*100)}': v for q, v in zip(Q, np.quantile(delta.abs(), Q) if len(delta) else [np.nan]*len(Q))}))
    save('error_summary', rows)
    ranked = chains.assign(abs_score_error=(chains.pred_score-chains.formula_total).abs())
    save('largest_errors', ranked.nlargest(15, 'abs_score_error'))


def quantiles(frame: pd.DataFrame) -> None:
    """同時連鎖のフレームを差の五分位に分け、実勝者と各確率を比べる。"""
    rows = []
    part = frame[frame.both_chain & frame.winner.notna()]
    for metric in ('pred_send_diff', 'incoming_diff', 'remaining_count_diff', 'remaining_send_diff'):
        valid = part[part[metric].notna()].copy()
        if valid.empty:
            continue
        valid['bin'] = pd.qcut(valid[metric], 5, duplicates='drop')
        for bucket, values in valid.groupby('bin', observed=True):
            for model, col in (('S3_base', 'base_p1'), ('G_fe_landing', 'landing_gfe'),
                               ('output_raw', 'display_p1'), ('output_shown', 'shown_p1')):
                own = values[values[col].notna()]
                rows.append(dict(metric=metric, bin=str(bucket), model=model, n=len(own),
                    seconds=len(own)/FPS, games=len(own[['set', 'match']].drop_duplicates()),
                    observed_win=own.winner.mean(), model_p1=own[col].mean()))
    save('winrate_quantiles', rows)


def weak_probability(frame: pd.DataFrame) -> None:
    """火力差代理と、探索で確認した返し不足を分ける。"""
    rows, reasons = [], []
    base = frame[frame.both_chain & frame.winner.notna()].copy()
    definitions = {
        'predicted_gap_ge30': (base.pred_send_diff.abs() >= PROXY_OJAMA, base.pred_send_diff > 0),
        'pending_gap_ge30': (base.incoming_diff.abs() >= PROXY_OJAMA, base.incoming_diff > 0),
        'searched_deficit': ((base.deficit1 > 0) ^ (base.deficit2 > 0), base.deficit2 > 0),
    }
    for name, (mask, favorable) in definitions.items():
        selected = base[mask].copy()
        selected['favored_p'] = np.where(favorable[mask], selected.shown_p1, 1-selected.shown_p1)
        selected['weak'] = selected.favored_p < CONFIDENCE
        for source, group in [('ALL', selected), *list(selected.groupby('source'))]:
            rows.append(dict(definition=name, source=source, n=len(group), seconds=len(group)/FPS,
                known_shown=int(group.favored_p.notna().sum()), weak_n=int(group.weak.sum()),
                weak_seconds=group.weak.sum()/FPS, weak_fraction=group.weak.mean()))
        weak = selected[selected.weak]
        for reason, group in weak.groupby(['reason', 'bound_reason'], dropna=False):
            reasons.append(dict(definition=name, reason=reason[0], bound_reason=reason[1],
                                n=len(group), seconds=len(group)/FPS))
    save('weak_probability', rows)
    save('weak_reasons', pd.DataFrame(reasons, columns=['definition', 'reason', 'bound_reason', 'n', 'seconds']).sort_values('n', ascending=False))


def scenes(frame: pd.DataFrame, chains: pd.DataFrame) -> None:
    """user指摘7窓の元動画時刻をそのまま集計する。"""
    rows, chain_samples = [], []
    for number, (setno, match, start, end) in enumerate(SCENES, 1):
        part = frame[(frame['set'] == setno) & frame.t_sec.between(start, end, inclusive='left')]
        for source, subset in part.groupby('source'):
            row = dict(scene=number, set=setno, match=match, start=start, end=end,
                       source=source, n=len(subset), seconds=len(subset)/FPS)
            for col in ('display_p1', 'shown_p1', 'base_p1', 'landing_gfe', 'pred_send_diff', 'incoming_diff'):
                row[col+'_min'], row[col+'_max'] = subset[col].min(), subset[col].max()
            rows.append(row)
        subset = chains[(chains['set'] == setno) & (chains.match == match)
                        & chains.trigger.between(start-25, end)]
        chain_samples.append(subset.assign(scene=number))
    save('scenes', rows)
    save('scene_chains', pd.concat(chain_samples))


def features(names: list[str]) -> None:
    """分布は推論呼出単位。隠し段候補の複数呼出も含むことを明示する。"""
    manifest = json.loads(Path('models/exchange_event_v3/manifest.json').read_text())['models']
    gcols = json.loads(Path('models/exchange_event_v1_r1_hgb/manifest.json').read_text())['models']['G_fe']['columns']
    output, populations = [], []
    for model in ('G_fe_R1s', 'G_fe_landing', 'S1', 'S3'):
        paths = [OUT / 'replay' / n / f'input_{model}.npy' for n in names]
        paths = [p for p in paths if p.exists()]
        if not paths:
            populations.append(dict(model=model, calls=0))
            continue
        x = np.concatenate([np.load(p) for p in paths])
        population = pd.concat([pd.read_csv(p.with_suffix('.csv')) for p in paths], ignore_index=True)
        known = pd.concat([pd.read_csv(p.parent / f'stats_{model}.csv') for p in paths])
        cols = gcols if model.startswith('G_fe') else manifest[model + '_prime_light']['columns']
        populations.append(dict(model=model, calls=len(x), both_chain_calls=int(population.both_chain.sum()),
                                columns=len(cols)))
        for i, col in enumerate(cols):
            if fire_column(col):
                same = known[known.column == col]
                output.append(dict(model=model, column=col, **stats(x[:, i]),
                    impute_known_n=same.impute_known_n.sum(), imputed_zero=same.imputed_zero.sum()))
        if model == 'S3':
            sensitivity(x, population, cols)
    save('production_stats', output)
    save('model_populations', populations)


def sensitivity(x: np.ndarray, metadata: pd.DataFrame, cols: list[str]) -> None:
    """動画外部標本の列別permutation。因果効果とは解釈しない。"""
    model = joblib.load('models/exchange_event_v3/S3_prime_light.joblib')
    eligible = metadata.winner.notna().to_numpy()
    ids = np.flatnonzero(eligible)
    # 同一時刻の隠し段候補・再呼出の偏りを抑え、最後の1秒1入力を採る。
    dedup = metadata.iloc[ids].assign(second=np.floor(metadata.iloc[ids].t)).drop_duplicates('second', keep='last')
    ids = dedup.index.to_numpy()
    x, y = x[ids].copy(), metadata.winner.to_numpy()[ids]
    baseline = model.predict_proba(x)[:, 1]
    loss = lambda p: float(np.mean(-y*np.log(np.clip(p, 1e-7, 1-1e-7))-(1-y)*np.log(np.clip(1-p, 1e-7, 1-1e-7))))
    rng, rows = np.random.default_rng(1003), []
    groups = [(col, [i]) for i, col in enumerate(cols) if fire_column(col)]
    groups += [('ALL_FIRE', [i for i, col in enumerate(cols) if fire_column(col)])]
    for name, selected in groups:
        scores, movements = [], []
        for _ in range(3):
            changed = x.copy()
            changed[:, selected] = x[rng.permutation(len(x))][:, selected]
            p = model.predict_proba(changed)[:, 1]
            scores.append(loss(p)-loss(baseline))
            movements.append(float(np.mean(abs(p-baseline))))
        rows.append(dict(column=name, n=len(x), base_logloss=loss(baseline),
            delta_logloss=np.mean(scores), mean_abs_delta_p=np.mean(movements),
            delta_min=min(scores), delta_max=max(scores)))
    save('s3_permutation', rows)


def population(frame: pd.DataFrame, chains: pd.DataFrame) -> None:
    """対象母数・欠測・実時間を明示する。"""
    rows = []
    for setno, part in frame.groupby('set'):
        own = chains[chains['set'] == setno]
        rows.append(dict(set=setno, start=part.t_sec.min(), end=part.t_sec.max(),
            frames=len(part), seconds=len(part)/FPS, games=part.match.nunique(), chains=len(own),
            formula_count_known=int(own.obs_count.notna().sum()),
            formula_score_known=int(own.formula_total.notna().sum()),
            independent_score_known=int(own.independent_score.notna().sum()),
            both_chain_frames=int(part.both_chain.sum()),
            both_prediction_frames=int((part.both_chain & part.pred_send_diff.notna()).sum()),
            first_observed_midchain=int((own.first_obs_count > 1).sum())))
    save('population', rows)


def audit_reasons(names: list[str], chains: pd.DataFrame) -> None:
    """起点盤面・隠し段・途中観測の監査記録を集計する。"""
    rows, totals = [], []
    for name in names:
        folder = OUT / 'replay' / name
        values = json.loads((folder / 'prefire_audit.json').read_text())['rows']
        own = chains[chains.segment == name]
        keys = set(zip(own.game_idx, own.side, own.chain_id))
        for v in values:
            if (v['game'], v['side'], v['chain_id']) not in keys:
                continue
            board = v.get('snapshot', {}).get('board')
            unknown_hidden = bool(board and 10 in board[0])
            unknown_visible = bool(board and any(10 in r for r in board[1:]))
            rows.append(dict(segment=name, game_idx=v['game'], side=v['side'], chain_id=v['chain_id'],
                trigger=v['trigger_sec'], reason=v['reason'] or 'accepted', accepted=v['accepted'],
                used=v['used'], withdrawn=v['withdrawn'], hidden_unknown=unknown_hidden,
                visible_unknown=unknown_visible, predicted_score=v['predicted_score']))
        mid = json.loads((folder / 'midchain_audit.json').read_text())
        totals.append(dict(segment=name, **{k: v for k, v in mid.items() if isinstance(v, (int, float))}))
    raw = save('prefire_rows', rows)
    save('prefire_reasons', raw.groupby('reason', dropna=False).agg(n=('reason', 'size'),
        hidden_unknown=('hidden_unknown', 'sum'), visible_unknown=('visible_unknown', 'sum'),
        used=('used', 'sum')).reset_index())
    save('midchain_totals', totals)


def dynamic_accuracy(names: list[str], chains: pd.DataFrame) -> None:
    """同時連鎖中の各側予測を、その連鎖の最終観測と照合する。"""
    rows = []
    for name in names:
        folder = OUT / 'replay' / name
        frame = pd.read_csv(folder / 'frames.csv.gz', usecols=['t_sec'])
        low, high = frame.t_sec.min(), frame.t_sec.max()
        selected = chains[chains.segment == name]
        truth = {(r.game_idx, r.side, r.chain_id): r for r in selected.itertuples()}
        for row in read_jsonl(folder / 'observations.jsonl.gz'):
            if row['states'] != ['CHAIN', 'CHAIN'] or not low <= row['t'] <= high:
                continue
            for side in ('1P', '2P'):
                c = next((c for c in reversed(row['chains']) if c['side'] == side), None)
                target = truth.get((row['game'], side, c['chain_id'])) if c else None
                rows.append(dict(segment=name, t=row['t'], side=side,
                    predicted_score=c['predicted_final_score'] if c else np.nan,
                    predicted_count=c['predicted_chain_count'] if c else np.nan,
                    final_formula=target.formula_total if target else np.nan,
                    final_count=target.obs_count if target else np.nan))
    raw = save('dynamic_predictions', rows)
    output = []
    for pred, actual in [('predicted_score', 'final_formula'), ('predicted_count', 'final_count')]:
        paired = raw[[pred, actual]].dropna()
        delta = paired[pred]-paired[actual]
        output.append(dict(metric=pred, opportunities=len(raw), prediction_available=int(raw[pred].notna().sum()),
            n=len(paired), exact=int((delta.abs() < 1e-6).sum()), bias=delta.mean(),
            **{f'abs_p{round(q*100)}': v for q, v in zip(Q, np.quantile(delta.abs(), Q))}))
    save('dynamic_error_summary', output)


def main() -> None:
    """完了した区間だけを集計し、完了数を必ず保存する。"""
    resources()
    names = [n for n in NAMES if (OUT / 'replay' / n / 'EXTRACTED.json').exists()]
    frame = pd.concat([pd.read_csv(OUT / 'replay' / n / 'frames.csv.gz') for n in names], ignore_index=True)
    chains = pd.concat([pd.read_csv(OUT / 'replay' / n / 'chains.csv') for n in names], ignore_index=True)
    assert not frame.t_sec.duplicated().any(), '表示区間の重複'
    save('all_chains', chains)
    population(frame, chains)
    errors(chains)
    quantiles(frame)
    weak_probability(frame)
    scenes(frame, chains)
    audit_reasons(names, chains)
    dynamic_accuracy(names, chains)
    features(names)
    (OUT / 'AGGREGATE.json').write_text(json.dumps(dict(segments=names, complete=len(names) == len(NAMES))))
    print('aggregate', names, flush=True)


if __name__ == '__main__':
    main()
