"""Phase 5 の遅延監査と参考指標。参考指標を合否条件へ混ぜない。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import aggregate_e3_exchange_eval_20260926 as outcomes
from scripts import prefire_oracle_ceiling_20260930 as oracle
from scripts.prefire_bestplay_gate_20261001 import trace
from scripts.run_prefire_replay_20260930 import BASELINE_DIRS, OUT
from src.prefire_v5_search import OK, MISSING, FIRING, INVALID

EPS = 1e-7
LEAD = 0.5
WINDOW = 3.0
FPS = 30
SIMULATED_LATENCY_MS = 300.0
MIN_EXPERIMENT_SAMPLES = 100


def winner_labels(source: str, times: np.ndarray) -> np.ndarray:
    """既存独立ラベル・公式勝者・認識参照を区別し、ラベルなしをNaNで残す。"""
    if source == 'zenchi':
        windows = json.loads(oracle.ZENCHI_GAMES.read_text())
    elif source in oracle.LABELED:
        windows = [dict(start=w['start']/FPS, end=w['end']/FPS, winner=w['winner'])
                   for w in outcomes.outcomes(source)[0]]
    else:
        windows = []
    labels = np.full(len(times), np.nan)
    for window in windows:
        labels[(times >= window['start']) & (times < window['end'])] = float(window['winner'] == '1P')
    return labels


def audit(table: dict) -> dict:
    """使用だけでなく計算値を参照した行も完了時刻で監査する。"""
    referenced = np.isfinite(table['v_1p']) | np.isfinite(table['v_2p'])
    early = table['t_sec'] < table['ready_sec']
    return dict(rows=len(early), used=int(table['used'].sum()), compute_ms=timing(table),
        early_referenced=int((referenced & early).sum()), early_used=int((table['used'].astype(bool) & early).sum()),
        held=int(table['held'].sum()),
        missing_any=int(((table['status_1p'] == MISSING) | (table['status_2p'] == MISSING)).sum()),
        firing_any=int(((table['status_1p'] == FIRING) | (table['status_2p'] == FIRING)).sum()),
        not_predicted=int((table['used'] == 0).sum()),
        statuses={name: [int((table[f'status_{side}'] == code).sum()) for side in ('1p', '2p')]
                  for name, code in [('ok', OK), ('missing', MISSING), ('firing', FIRING), ('invalid', INVALID), ('waiting', 4)]})


def timing(table: dict) -> dict:
    """模擬0.3秒と実計算所要は別。実時間で動作可能という誤解を避けて記録する。"""
    costs = table.get('_computes_ms', np.array([]))
    return dict(count=len(costs), p50=float(np.percentile(costs, 50)) if len(costs) else None,
        p95=float(np.percentile(costs, 95)) if len(costs) else None,
        maximum=float(costs.max()) if len(costs) else None,
        over_simulated_latency=int((costs > SIMULATED_LATENCY_MS).sum()))


def agreement(table: dict, rows: list[dict]) -> dict:
    """直前と0.5秒前の両方で採点可能な同じ撃ち合いだけ比較する。"""
    last_hits, early_hits, count = 0, 0, 0
    coverage = dict(exchanges=len(rows), with_both_endpoints=0, last_scored=0, half_second_scored=0, both_scored=0)
    for row in rows:
        eligible = np.flatnonzero((table['game_idx'] == row['game']) &
            (table['t_sec'] >= max(row['prev_close'], row['trigger']-WINDOW)) & (table['t_sec'] < row['trigger']))
        earlier = eligible[table['t_sec'][eligible] <= row['trigger']-LEAD]
        if not len(eligible) or not len(earlier):
            continue
        last, early = int(eligible[-1]), int(earlier[-1])
        coverage['with_both_endpoints'] += 1
        coverage['last_scored'] += int(bool(table['used'][last]))
        coverage['half_second_scored'] += int(bool(table['used'][early]))
        if not all(table['used'][i] for i in (last, early)):
            continue
        coverage['both_scored'] += 1
        totals = [row['target']-table['p_current'][i] for i in (last, early)]
        shifts = [table['p_shown'][i]-table['p_current'][i] for i in (last, early)]
        if any(abs(v) < oracle.MIN_TOTAL_MOVE for v in totals) or any(v == 0 for v in shifts):
            continue
        count += 1
        last_hits += np.sign(totals[0]) == np.sign(shifts[0])
        early_hits += np.sign(totals[1]) == np.sign(shifts[1])
    return dict(denominator=count, last_hits=int(last_hits), half_second_hits=int(early_hits), **coverage)


def loss(table: dict, rows: list[dict], source: str) -> dict:
    """発火前3秒を固定し、勝者ラベルがある行だけLLを計算する。"""
    prefire = np.zeros(len(table['t_sec']), dtype=bool)
    for row in rows:
        prefire |= ((table['game_idx'] == row['game']) & (table['t_sec'] < row['trigger']) &
                    (table['t_sec'] >= max(row['prev_close'], row['trigger']-WINDOW)))
    labels = winner_labels(source, table['t_sec'])
    selected = prefire & np.isfinite(labels)
    scores = {}
    for name, mask in [('all_with_fallback', selected), ('predicted_only', selected & (table['used'] > 0))]:
        result = dict(rows=int(mask.sum()), games=int(len(np.unique(table['game_idx'][mask]))))
        for column in ('p_current', 'p_shown'):
            p = np.clip(table[column][mask], EPS, 1-EPS)
            losses = -(labels[mask]*np.log(p)+(1-labels[mask])*np.log1p(-p))
            games = table['game_idx'][mask]
            result[column] = float(np.mean(losses)) if len(losses) else None
            result[column+'_game_mean'] = float(np.mean([np.mean(losses[games == g]) for g in np.unique(games)])) if len(losses) else None
        scores[name] = result
    return dict(prefire_rows=int(prefire.sum()), missing_labels=int((prefire & ~np.isfinite(labels)).sum()), **scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short', action='store_true')
    args = parser.parse_args()
    if args.short:
        print(json.dumps(audit(trace(Path('logs/prefire_prediction/v5_short')))), flush=True)
        return
    result = {}
    for source in BASELINE_DIRS:
        table = trace(OUT/'v5_L03'/source)
        display, events = oracle.load(source)
        rows = oracle.exchange_rows(display, events)
        result[source] = dict(audit=audit(table), agreement=agreement(table, rows), loss=loss(table, rows, source),
                              display_loss=loss(display_table(source, display, table), rows, source))
    dest = Path('logs/prefire_prediction/v5_experiment/replay_report.json')
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = final_summary(result)
    (dest.parent/'final_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


def final_summary(sources: dict) -> dict:
    """事前の3条件だけで判定する。参考LL・向き一致は合否へ入れない。"""
    root = Path('logs/prefire_prediction/v5_experiment')
    experiment = json.loads((root/'ledger/summary.json').read_text())
    audits = [s['audit'] for s in sources.values()]
    early = sum(a['early_referenced'] for a in audits)
    gates = dict(a=experiment['samples'] >= MIN_EXPERIMENT_SAMPLES and experiment['mismatches'] == 0,
                 b=experiment['samples'] >= MIN_EXPERIMENT_SAMPLES and experiment['missed'] == 0,
                 c=len(sources) == len(BASELINE_DIRS) and early == 0)
    agreements = [s['agreement'] for s in sources.values()]
    return dict(experiment=experiment, gates=gates, passed=all(gates.values()),
        trace_rows=sum(a['rows'] for a in audits), predicted_rows=sum(a['used'] for a in audits),
        early_referenced=early, missing_rows=sum(a['missing_any'] for a in audits),
        firing_rows=sum(a['firing_any'] for a in audits), held_rows=sum(a['held'] for a in audits),
        agreement={key: sum(a[key] for a in agreements) for key in agreements[0]})


def display_table(source: str, baseline: dict, table: dict) -> dict:
    """参考LLは表示全行にも計算し、静止traceだけの母数と区別する。"""
    with np.load(OUT/'v5_L03'/source/'display.npz') as data:
        result = {name: data[name].copy() for name in ('t_sec', 'game_idx', 'display_p1')}
    if not np.array_equal(result['t_sec'], baseline['t_sec']):
        raise ValueError(f'表示の時刻対応が不一致: {source}')
    used = dict(zip(zip(table['t_sec'], table['game_idx']), table['used']))
    result['used'] = np.array([used.get(key, 0) for key in zip(result['t_sec'], result['game_idx'])])
    result['p_current'], result['p_shown'] = baseline['display_p1'], result.pop('display_p1')
    return result


if __name__ == '__main__':
    main()
