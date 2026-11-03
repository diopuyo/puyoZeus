"""火力監査と同じ初回予測・両者CHAIN側フレームの誤差を集計する。"""
from __future__ import annotations

import json
from typing import Any
import numpy as np
import pandas as pd

from scripts.a1_completion_replay import OUT
from scripts._audit_firepower_1003_extract import (
    NAMES, FPS, games_for, label, process_observation, chain_rows, read_jsonl,
)
from scripts._audit_firepower_1003_report import SCENES, Q

AUDIT = OUT.parents[2]/'review_2026-10-03/fire_audit/all_chains.csv'
AUDIT_ABSOLUTE_TOLERANCE = 1e-9
EXACT_ERROR_TOLERANCE = 1e-6


def reference_check(chains: pd.DataFrame) -> None:
    """既存監査との母数・初回予測・照合値の一致を別票に残す。"""
    reference = pd.read_csv(AUDIT)
    keys = ['segment', 'game_idx', 'side', 'chain_id', 'exchange']
    joined = reference.merge(chains, on=keys, suffixes=('_audit', '_current'), how='outer', indicator=True)
    columns = {}
    for name in ('pred_count', 'pred_score', 'obs_count', 'formula_total'):
        left, right = joined[name+'_audit'], joined[name+'_current']
        same = np.isclose(left, right, rtol=0, atol=AUDIT_ABSOLUTE_TOLERANCE, equal_nan=True)
        columns[name] = dict(equal=int(same.sum()), total=len(same))
    value = dict(audit_rows=len(reference), current_rows=len(chains),
        keys=joined['_merge'].value_counts().to_dict(), columns=columns)
    (OUT/'AUDIT_DEFINITION_CHECK.json').write_text(json.dumps(value, indent=1), encoding='utf-8')


def extract(mode: str, name: str) -> tuple[pd.DataFrame, list[dict]]:
    """初回値を凍結し、観測・得点帰属の事後情報は採点専用に分離する。"""
    folder = OUT/'replay'/mode/name
    games = games_for(name)
    with np.load(folder/'display.npz') as data:
        times = data['t_sec'][[label(t, games)['match'] >= 0 for t in data['t_sec']]]
    initial, observed = {}, {}
    for row in read_jsonl(folder/'observations.jsonl.gz'):
        process_observation(row, initial, observed)
    chains = pd.DataFrame(chain_rows(name, folder, initial, observed, times, games))
    truth = {(r.game_idx, r.side, r.chain_id): r for r in chains.itertuples()}
    dynamic = []
    for row in read_jsonl(folder/'observations.jsonl.gz'):
        if row['states'] != ['CHAIN', 'CHAIN'] or not times[0] <= row['t'] <= times[-1]:
            continue
        for side in ('1P', '2P'):
            chain = next((c for c in reversed(row['chains']) if c['side'] == side), None)
            target = truth.get((row['game'], side, chain['chain_id'])) if chain else None
            dynamic.append(dict(segment=name, t=row['t'], side=side,
                predicted_score=chain['predicted_final_score'] if chain else np.nan,
                predicted_count=chain['predicted_chain_count'] if chain else np.nan,
                final_formula=target.formula_total if target else np.nan,
                final_count=target.obs_count if target else np.nan))
    return chains, dynamic


def errors(mode: str, chains: pd.DataFrame, dynamic: pd.DataFrame) -> list[dict]:
    """監査の同じ欠測除外・同じ分位点を使う。0予測は除外しない。"""
    rows = []
    groups = [('all', chains), ('observed_7plus', chains[chains.obs_count >= 7]),
              ('both_chain_at_fire', chains[chains.both_chain])]
    for group, part in groups:
        for metric, pred, truth in (('score', 'pred_score', 'formula_total'),
                                    ('count', 'pred_count', 'obs_count')):
            rows.append(error_row(mode, group, metric, part, pred, truth))
    for metric, pred, truth in (('score', 'predicted_score', 'final_formula'),
                                ('count', 'predicted_count', 'final_count')):
        rows.append(error_row(mode, 'both_CHAIN_dynamic', metric, dynamic, pred, truth))
    return rows


def error_row(mode: str, group: str, metric: str, part: pd.DataFrame, pred: str, truth: str) -> dict:
    """予測と帰属済み式の両方がある標本だけを対にする。"""
    valid = part[[pred, truth]].dropna()
    delta = valid[pred]-valid[truth]
    quantiles = np.quantile(delta.abs(), Q) if len(delta) else [np.nan]*len(Q)
    return dict(mode=mode, group=group, metric=metric, total=len(part), n=len(delta),
        exact=int((delta.abs() < EXACT_ERROR_TOLERANCE).sum()), bias=delta.mean(),
        **{f'abs_p{round(q*100)}': v for q, v in zip(Q, quantiles)})


def scenes(mode: str) -> list[dict]:
    """レビュー7窓の採用確率と表示有利不利を、別の尺度として記録する。"""
    frames = []
    for name in NAMES:
        with np.load(OUT/'replay'/mode/name/'display.npz') as data:
            frames.append(pd.DataFrame({k: data[k] for k in ('t_sec', 'source', 'display_p1', 'display_adv')})
                          .assign(set=2 if name.startswith('s') else 1))
    frame, rows = pd.concat(frames), []
    for number, (setno, match, start, end) in enumerate(SCENES, 1):
        selected = frame[(frame['set'] == setno) & frame.t_sec.between(start, end, inclusive='left')]
        for source, part in [('ALL', selected), *list(selected.groupby('source'))]:
            rows.append(dict(mode=mode, scene=number, set=setno, match=match, start=start, end=end,
                source=source, frames=len(part),
                seconds=len(part)/FPS, raw_min=part.display_p1.min(), raw_max=part.display_p1.max(),
                raw_mean=part.display_p1.mean(), adv_min=part.display_adv.min(),
                adv_max=part.display_adv.max(), adv_mean=part.display_adv.mean()))
    return rows


def main() -> None:
    """両構成の全発火CSVと動的側フレームCSVを保存する。"""
    summary, scene_rows = [], []
    for mode in ('prod', 'a1'):
        chains, dynamic = [], []
        for name in NAMES:
            part, frames = extract(mode, name)
            chains.append(part)
            dynamic.extend(frames)
            print(mode, name, len(part), flush=True)
        combined, frames = pd.concat(chains), pd.DataFrame(dynamic)
        if mode == 'prod':
            reference_check(combined)
        combined.to_csv(OUT/f'{mode}_all_chains.csv', index=False)
        frames.to_csv(OUT/f'{mode}_dynamic_predictions.csv', index=False)
        summary.extend(errors(mode, combined, frames))
        scene_rows.extend(scenes(mode))
    pd.DataFrame(summary).to_csv(OUT/'error_summary.csv', index=False)
    pd.DataFrame(scene_rows).to_csv(OUT/'scenes.csv', index=False)


if __name__ == '__main__':
    main()
