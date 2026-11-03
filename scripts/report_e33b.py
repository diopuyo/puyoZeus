"""E33bを固定表示行で採点し、二機構の原因と合否を保存する。"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from typing import Any
import numpy as np
from scripts import run_e17_ablation_20260928 as prior
from scripts.audit_e33b import attribute
from scripts.e33b_score_trace import OUT, PREDICTION_SOURCES
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json

LOSS_MAX = .509977
AGREEMENT_MIN = .9167
VARIANTS = ('off', 'e33', 'on')


def read(path: Path) -> Any:
    """日本語原票をUTF-8で読む。"""
    return json.loads(path.read_text(encoding='utf-8'))


def directory(variant: str, source: str) -> Path:
    """既存評価器と同じディレクトリ配置を使う。"""
    suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
    return OUT/variant/suffix


def predictions(variant: str, source: str, cohort: list[dict],
                sources: Counter | None = None) -> list[float]:
    """欠測や撤回を黙って除かず、事前固定した全行の予測を照合する。"""
    if not cohort:
        return []
    wanted = {r['frame'] for r in cohort}
    rows = {}
    with gzip.open(directory(variant, source)/'score_rows.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row['frame'] in wanted:
                rows[row['frame']] = row
    result = []
    for target in cohort:
        row = rows[target['frame']]
        assert (row['t_sec'], row['game']) == (target['t_sec'], target['game'])
        assert row['source'] in PREDICTION_SOURCES, (variant, source, target, row)
        if sources is not None:
            sources[row['source']] += 1
        found = [s['score'] for s in row['scores']
                 if (s['side'], s['chain_id']) == (target['side'], target['chain_id'])]
        assert len(found) == 1, (variant, source, target, row)
        assert row['value_sec'] <= row['t_sec']
        result.append(found[0])
    return result


def scores() -> dict:
    """連鎖×表示行を等重みとし、三条件で同じ真値と母数を使う。"""
    actual, keys = [], []
    predicted = {v: [] for v in VARIANTS}
    sources = {v: Counter() for v in VARIANTS}
    for source in SOURCES:
        cohort = read(OUT/'cohort'/f'{source}.json')
        actual.extend(r['actual'] for r in cohort)
        keys.extend(f"{source}:{r['frame']}:{r['side']}:{r['chain_id']}" for r in cohort)
        for variant in VARIANTS:
            predicted[variant].extend(predictions(variant, source, cohort, sources[variant]))
    truth = np.asarray(actual, dtype=float)
    result = {}
    for variant, values in predicted.items():
        error = np.abs(np.asarray(values)-truth)
        assert len(error) == len(keys) and np.isfinite(error).all()
        result[variant] = dict(n=len(error), p50=float(np.percentile(error, 50)),
            p95=float(np.percentile(error, 95)), mean=float(error.mean()))
    np.savez_compressed(OUT/'SCORE_ROWS.npz', keys=keys, actual=truth, **predicted)
    save_json(OUT/'SCORES.json', result)
    save_json(OUT/'DISPLAY_SOURCES.json', sources)
    return result


def measured_metrics() -> dict:
    """新条件を既存のq・公式勝敗・誤発火の定義で集計する。"""
    prior.OUT, prior.LOSS_MAX, prior.AGREEMENT_MIN = OUT, LOSS_MAX, AGREEMENT_MIN
    prior.report_base.death_metrics.OUT = OUT/'on'
    path = OUT/'on/METRICS.json'
    if path.exists():
        return read(path)
    assert all((directory('on', s)/'DONE.json').exists() for s in prior.ALL_SOURCES)
    value = prior.report('on')
    save_json(OUT/'DEATH_AUDIT.json', prior.report_base.death_metrics.deaths())
    return value


def comparisons() -> dict:
    """全列OFF互換と三条件の同一入力行を確認する。"""
    result = {}
    for source in prior.ALL_SOURCES:
        a, b = [np.load(directory(v, source)/'display.npz') for v in ('off', 'on')]
        for name in ('t_sec', 'game_idx', 'state1', 'state2', 'score1', 'score2'):
            assert a[name].tobytes() == b[name].tobytes(), (source, name)
        check = read(OUT/f'OFF_{source}.json')
        assert check == dict(display='byte_identical', events='byte_identical', diagnostics='identical')
        result[source] = dict(rows=len(a['t_sec']), columns=len(a.files), **check)
    for source in SOURCES:
        assert read(OUT/f'E33_{source}.json')['display'] == 'byte_identical'
    return result


def timeout_audit() -> dict:
    """新条件の候補除外が時間切れだけであることを全入力で照合する。"""
    result = {}
    for source in prior.ALL_SOURCES:
        rows = read(directory('on', source)/'snapshot_final_audit.json')['rows']
        changed = [r for r in rows if r.get('stage_exclusions')]
        events = [event for row in changed for event in row['stage_exclusions']]
        assert all(e['reason'] == 'stage_timeout' for e in events)
        result[source] = dict(chains=len(changed), operations=len(events),
            emptied=sum(e['remaining'] == 0 for e in events))
    return result


def report() -> dict:
    """事前登録された五条件を全て満たした場合だけ合格にする。"""
    score = scores()
    metrics = dict(off=read(Path('logs/e32/on/METRICS.json')),
        e33=read(Path('logs/e33/on/METRICS.json')), on=measured_metrics())
    checks, gates = comparisons(), {}
    for variant, value in metrics.items():
        assert value['q']['frames'] == 6526 and value['zenchi']['frames'] == 8333
        gates[variant] = dict(q=value['q']['log_loss'] <= LOSS_MAX,
            zenchi=value['zenchi']['agreement'] >= AGREEMENT_MIN,
            deaths=value['deaths']['unlabelled'] == 0 and
                value['deaths']['false']/max(1, value['deaths']['total']) <= 1/28,
            score_mean=score[variant]['mean'] < score['off']['mean'],
            score_p95=score[variant]['p95'] < score['off']['p95'])
    metrics['on'].update(gates=gates['on'], candidate=all(gates['on'].values()))
    save_json(OUT/'on/METRICS.json', metrics['on'])
    metrics = {v: {k: m[k] for k in ('q', 'zenchi', 'deaths')} for v,m in metrics.items()}
    result = dict(metrics=metrics, scores=score, attribution=attribute(),
        comparisons=checks, cohort=read(OUT/'COHORT.json'), gates=gates,
        exclusions=timeout_audit(), passed=all(gates['on'].values()))
    save_json(OUT/'SUMMARY.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--scores-only', action='store_true')
    modes.add_argument('--metrics-only', action='store_true')
    args = parser.parse_args()
    value = scores() if args.scores_only else measured_metrics() if args.metrics_only else report()
    print(json.dumps(value, ensure_ascii=False, indent=2))
