"""E33をE32と同じ行・採用母数で採点し、除外と因果性を集約する。"""
from __future__ import annotations
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
from scripts import run_e17_ablation_20260928 as prior
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from scripts.run_e30 import quantiles
from scripts.run_e33 import OUT, CUTOFF, CUTOFF_SOURCE

LOSS_MAX = .509977
AGREEMENT_MIN = .9167
ERROR_P95_MAX = 1162.3984220907228
TOP_ERRORS = 5


def read(path: Path) -> Any:
    """UTF-8の固定検証原票を読む。"""
    return json.loads(path.read_text(encoding='utf-8'))


def directory(variant: str, source: str) -> Path:
    """集計器と同じ配置を副作用なしで参照する。"""
    suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
    return OUT/variant/suffix


def identity(row: dict) -> tuple:
    """試合と側を含めて同一発火を照合する。"""
    return row['source'], row['game'], row['side'], row['chain_id']


def score_report() -> dict:
    """事前登録の初回誤差73件を保持し、後続予測の誤差も並べる。"""
    rows = {v: [dict(r, source=s) for s in SOURCES
        for r in read(directory(v, s)/'snapshot_final_audit.json')['rows']]
        for v in ('off', 'on')}
    old = {identity(r): r for r in rows['off'] if r['accepted'] and r['error'] is not None}
    new = {identity(r): r for r in rows['on'] if r['accepted'] and r['error'] is not None}
    assert old.keys() == new.keys() and len(old) == 73
    saved = read(Path('logs/e32/SNAPSHOT_COUNTS.json'))['three_videos']['score_absolute_error']
    first = {v: quantiles([abs(r['error']) for r in rs if r['accepted'] and r['error'] is not None])
             for v, rs in rows.items()}
    assert first['off'] == saved
    latest = {v: quantiles([abs(r['last_error']) for r in rs
        if identity(r) in old and r['last_error'] is not None]) for v, rs in rows.items()}
    assert all(v['n'] == 73 for v in latest.values())
    top = []
    for key in sorted(old, key=lambda k: abs(old[k]['error']), reverse=True)[:TOP_ERRORS]:
        a, b = old[key], new[key]
        top.append(dict(zip(('source', 'game', 'side', 'chain_id'), key),
            trigger_sec=a['trigger_sec'], final_score=a['final_score'],
            e32_initial_error=a['error'], e33_initial_error=b['error'],
            e32_last_error=a['last_error'], e33_last_error=b['last_error'],
            exclusions=b.get('stage_exclusions', [])))
    return dict(initial=first, latest=latest, top5=top)


def exclusion_report() -> dict:
    """重複発動と候補数を区別し、記録別・理由別に集計する。"""
    sources, details = {}, []
    for source in prior.ALL_SOURCES:
        rows = read(directory('on', source)/'snapshot_final_audit.json')['rows']
        records = [json.loads(line) for line in
                   (directory('on', source)/'events.jsonl').read_text().splitlines()]
        signals = {(r['game_idx'], c['side'], c['chain_id']): c['end_signals']
                   for r in records for c in r['chains']}
        changed = [r for r in rows if r.get('stage_exclusions')]
        for row in changed:
            for event in row['stage_exclusions']:
                # 取消時刻は事後監査専用であり、予測器へ戻さない。
                event['later_revoked_sec'] = next((s.get('revoked_sec')
                    for s in signals.get((row['game'], row['side'], row['chain_id']), [])
                    if s['t_sec'] == event['t_sec'] and s['reason'] == event['reason']), None)
        events = [e for r in changed for e in r['stage_exclusions']]
        sources[source] = dict(fires=len(rows), accepted=sum(r['accepted'] for r in rows),
            chains=len(changed), operations=len(events), removed=sum(e['removed'] for e in events),
            reasons=dict(Counter(e['reason'] for e in events)),
            emptied=sum(e['remaining'] == 0 for e in events),
            later_revoked=sum(e['later_revoked_sec'] is not None for e in events))
        details.extend(dict(source=source, game=r['game'], side=r['side'], chain_id=r['chain_id'],
            trigger_sec=r['trigger_sec'], events=r['stage_exclusions']) for r in changed)
    pooled = {k: sum(sources[s][k] for s in SOURCES)
              for k in ('fires', 'accepted', 'chains', 'operations', 'removed', 'emptied', 'later_revoked')}
    return dict(three_videos=pooled, sources=sources, details=details)


def prefix_report() -> dict:
    """打ち切り以前の全列と全時点のイベント状態をビット照合する。"""
    full, short = [directory(v, CUTOFF_SOURCE) for v in ('on', 'prefix')]
    a, b = np.load(full/'display.npz'), np.load(short/'display.npz')
    mask = a['t_sec'] <= CUTOFF
    assert a.files == b.files
    for name in a.files:
        left = a[name][mask] if a[name].ndim and a[name].shape[0] == len(mask) else a[name]
        assert left.dtype == b[name].dtype and left.shape == b[name].shape, name
        assert left.tobytes() == b[name].tobytes(), name
    states, prefix = read(full/'state_hashes.json'), read(short/'state_hashes.json')
    assert states == prefix and states[-1]['t'] == CUTOFF
    assert (full/'checkpoint.jsonl').read_bytes() == (short/'events.jsonl').read_bytes()
    assert read(full/'checkpoint.diagnostics.json') == read(short/'events.diagnostics.json')
    return dict(cutoff=CUTOFF, source=CUTOFF_SOURCE, display_rows=int(mask.sum()),
        columns=len(a.files), event_states=len(states), all_identical=True,
        input_sha256=hashlib.sha256((OUT/'input_prefix.jsonl.gz').read_bytes()).hexdigest())


def matching_rows() -> dict:
    """評価対象の時刻・試合・認識値の列が全条件で一致することを確認する。"""
    result = {}
    for source in prior.ALL_SOURCES:
        a, b = [np.load(directory(v, source)/'display.npz') for v in ('off', 'on')]
        for name in ('t_sec', 'game_idx', 'state1', 'state2', 'score1', 'score2'):
            assert a[name].tobytes() == b[name].tobytes(), (source, name)
        result[source] = len(a['t_sec'])
    return result


def measured_metrics() -> dict:
    """全ON記録の完了後に一度だけ、既存定義で集計する。"""
    prior.OUT, prior.LOSS_MAX, prior.AGREEMENT_MIN = OUT, LOSS_MAX, AGREEMENT_MIN
    prior.report_base.death_metrics.OUT = OUT/'on'
    path = OUT/'on/METRICS.json'
    if path.exists():
        return read(path)
    assert all((directory('on', s)/'DONE.json').exists() for s in prior.ALL_SOURCES)
    return prior.report('on')


def report() -> dict:
    """結果から閾値を動かさず、全必須ゲートの論理積を取る。"""
    metrics = measured_metrics()
    scores = score_report()
    gates = dict(q=metrics['q']['log_loss'] <= LOSS_MAX,
        zenchi=metrics['zenchi']['agreement'] >= AGREEMENT_MIN,
        deaths=metrics['deaths']['unlabelled'] == 0 and
            metrics['deaths']['false']/max(1, metrics['deaths']['total']) <= 1/28,
        initial_score_p95=scores['initial']['on']['p95'] < ERROR_P95_MAX)
    metrics.update(gates=gates, candidate=all(gates.values()))
    save_json(OUT/'on/METRICS.json', metrics)
    assert metrics['q']['frames'] == 6526 and metrics['zenchi']['frames'] == 8333
    result = dict(metrics=metrics, scores=scores, exclusions=exclusion_report(),
        prefix=prefix_report(), same_rows=matching_rows(),
        off={s: read(OUT/f'OFF_{s}.json') for s in prior.ALL_SOURCES},
        gates=gates, passed=all(gates.values()))
    save_json(OUT/'SUMMARY.json', result)
    save_json(OUT/'DEATH_AUDIT.json', prior.report_base.death_metrics.deaths())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(measured_metrics() if args.metrics_only else report(), ensure_ascii=False, indent=2))
