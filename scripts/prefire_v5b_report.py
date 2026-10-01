"""5Aと同じ3条件と、5Bの欠測・速度・参考LLを分離して集計する。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import prefire_v5_report as report
from scripts import prefire_oracle_ceiling_20260930 as oracle
from scripts.prefire_bestplay_gate_20261001 import trace
from scripts.run_prefire_replay_20260930 import BASELINE_DIRS

OUT = Path('logs/prefire_prediction/v5b')
LABELS = Path('/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/labels.json')


def set1_labels(times: np.ndarray) -> np.ndarray:
    """セット1の独立勝敗ラベルを動画の絶対時刻で対応付ける。"""
    labels = np.full(len(times), np.nan)
    for game in json.loads(LABELS.read_text())['games']:
        labels[(times >= game['start']) & (times < game['end'])] = float(game['winner'] == '1P')
    return labels


def score(source: str) -> dict:
    """既存5記録は5Aと同じ交換母集団を使用する。"""
    root = OUT/'replay'/source
    done = json.loads((root/'DONE.json').read_text())
    table = trace(root)
    if source in BASELINE_DIRS:
        display, events = oracle.load(source)
    else:
        with np.load(root/'display.npz') as data:
            display = {key: data[key].copy() for key in data.files}
        events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    rows = oracle.exchange_rows(display, events)
    original = report.winner_labels
    if source not in BASELINE_DIRS:
        report.winner_labels = lambda name, times: set1_labels(times)
    try:
        losses = report.loss(table, rows, source)
    finally:
        report.winner_labels = original
    audit = report.audit(table)
    audit['replay_bookkeeping_or_completed_only_ms'] = audit.pop('compute_ms')
    return dict(audit=audit, deferred=done['deferred'], agreement=report.agreement(table, rows), loss=losses)


def aggregate(sources: dict) -> dict:
    """通常評価へ戻った行を予測適用と数えない。"""
    experiment = json.loads((OUT/'final/summary.json').read_text())['after']
    audits = [sources[source]['audit'] for source in BASELINE_DIRS]
    gates = dict(a=experiment['samples'] == 100 and experiment['mismatches'] == 0,
                 b=experiment['samples'] == 100 and experiment['missed'] == 0,
                 c=len(audits) == 5 and sum(a['early_referenced'] for a in audits) == 0)
    names = ('rows', 'used', 'early_referenced', 'early_used', 'held', 'missing_any', 'firing_any')
    return dict(gates=gates, passed=all(gates.values()), experiment=experiment,
                **{name: sum(a[name] for a in audits) for name in names})


def combined_set1(sources: dict) -> dict:
    """6記録のLLを行数で加重平均し、欠測ラベルと適用行を併記する。"""
    losses = [sources[f'c{i}']['loss'] for i in range(1, 7)]
    rows = sum(loss['all_with_fallback']['rows'] for loss in losses)
    result = dict(rows=rows, predicted_rows=sum(loss['predicted_only']['rows'] for loss in losses),
                  missing_labels=sum(loss['missing_labels'] for loss in losses))
    for column in ('p_current', 'p_shown'):
        result[column] = sum(loss['all_with_fallback']['rows'] * (loss['all_with_fallback'][column] or 0)
                             for loss in losses) / rows if rows else None
    return result


def main() -> None:
    """全記録の完了を必須にして、欠けた結果を合格に読み替えない。"""
    sources = {source: score(source) for source in (*BASELINE_DIRS, *(f'c{i}' for i in range(1, 7)))}
    result = dict(summary=aggregate(sources), latency=json.loads((OUT/'latency.json').read_text()),
                  sources=sources, zenchi_set1=combined_set1(sources))
    (OUT/'replay_report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result['summary']), flush=True)
    print(json.dumps(result['zenchi_set1']), flush=True)


if __name__ == '__main__':
    main()
