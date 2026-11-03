"""E33bの対応条件を満たさない行を除外せず、固定母数の計測不成立を記録する。"""
from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from scripts.e33b_score_trace import PREDICTION_SOURCES
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json

VARIANTS = ('off', 'on')


def correspondence(target: dict, row: dict | None) -> tuple[float | None, str | None]:
    """E33bと同じ表示出所・時刻・連鎖対応を要求し、代替得点は作らない。"""
    if row is None:
        return None, 'missing_display_row'
    if (row['t_sec'], row['game']) != (target['t_sec'], target['game']):
        return None, 'time_or_game_mismatch'
    if row['source'] not in PREDICTION_SOURCES:
        return None, 'nonprediction_display'
    values = [v['score'] for v in row['scores']
              if (v['side'], v['chain_id']) == (target['side'], target['chain_id'])]
    if len(values) != 1:
        return None, 'missing_or_ambiguous_chain_score'
    if row['value_sec'] is None or row['value_sec'] > row['t_sec']:
        return None, 'invalid_value_time'
    if not np.isfinite(values[0]):
        return None, 'nonfinite_score'
    return float(values[0]), None


def measured_summary(values: list[float], truth: list[float]) -> dict:
    """全行が対応した場合だけ平均を出し、対応不能行を母数から引かない。"""
    predicted = np.asarray(values)
    valid = np.isfinite(predicted)
    complete = bool(valid.all()) and bool(len(values))
    error = np.abs(predicted-np.asarray(truth))
    return dict(n=len(values), measured_n=int(valid.sum()), unmeasurable_n=int((~valid).sum()),
        complete=complete, mean=float(error.mean()) if complete else None,
        p50=float(np.percentile(error, 50)) if complete else None,
        p95=float(np.percentile(error, 95)) if complete else None)


def audit_scores(out: Path) -> dict:
    """最初の不一致で停止せず、固定した全連鎖行の対応可否を列挙する。"""
    predicted: dict[str, list] = {v: [] for v in VARIANTS}
    sources: dict[str, Counter] = {v: Counter() for v in VARIANTS}
    truth, keys, failures = [], [], []
    for source in SOURCES:
        cohort = json.loads((out/'cohort'/f'{source}.json').read_text())
        truth.extend(r['actual'] for r in cohort)
        keys.extend(f"{source}:{r['frame']}:{r['side']}:{r['chain_id']}" for r in cohort)
        wanted = {r['frame'] for r in cohort}
        for variant in VARIANTS:
            with gzip.open(out/variant/'renders'/source/'on/score_rows.jsonl.gz', 'rt') as stream:
                records = (json.loads(line) for line in stream)
                rows = {r['frame']: r for r in records if r['frame'] in wanted}
            for target in cohort:
                row = rows.get(target['frame'])
                score, reason = correspondence(target, row)
                predicted[variant].append(score if reason is None else np.nan)
                sources[variant][row['source'] if row else 'missing'] += 1
                if reason is not None:
                    failures.append(dict(variant=variant, source=source, target=target,
                                         reason=reason, observed=row))
    result = {v: measured_summary(values, truth) for v, values in predicted.items()}
    np.savez_compressed(out/'SCORE_ROWS.npz', keys=keys, actual=truth, **predicted)
    save_json(out/'SCORES.json', result)
    save_json(out/'SCORE_UNMEASURABLE.json', dict(rows=failures,
        reasons={v: Counter(r['reason'] for r in failures if r['variant'] == v) for v in VARIANTS}))
    save_json(out/'DISPLAY_SOURCES.json', sources)
    return result
