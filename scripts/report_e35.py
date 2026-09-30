"""E35を固定母数で採点し、全5記録の誤った負け確定を照合する。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from scripts import run_e17_ablation_20260928 as prior
from scripts.run_e35 import OUT, SOURCES
from scripts.run_e30 import quantiles
from scripts.run_e3_exchange_eval_20260926 import save_json
from scripts.measure_e23_display_20260928 import displayed, first
from src.exchange_event_record import read_records

Q_MAX = .509977
ZENCHI_MIN = .9167
SCENE_DEADLINE = 2766.0
SCENE_STAMP = 2759.05  # 既存レビューの3:00行（判定器には渡さない）。


def read(path: Path) -> Any:
    """保存したUTF-8原票を読む。"""
    return json.loads(path.read_text(encoding='utf-8'))


def directory(source: str) -> Path:
    """既存集計器の出力配置を保持する。"""
    return OUT/'on'/(source if source in ('review', 'zenchi') else f'renders/{source}/on')


def events(source: str) -> list[dict]:
    """終端観測とモデル予測を混同せず、イベント原票を読む。"""
    return [json.loads(line) for line in (directory(source)/'events.jsonl').read_text().splitlines()]


def deaths(source: str, existing: list[dict]) -> list[dict]:
    """試合・側単位の初回断定を、実際の死亡信号と固定勝敗へ照合する。"""
    observed, predicted = {}, {}
    for item in read_records(Path('logs/e31/records')/f'{source}.jsonl.gz'):
        if item['kind'] != 'update':
            continue
        result, _, _, stamp, game, *_ = item['args']
        for side in getattr(result, 'confirmed_dead_sides', ()):
            observed.setdefault((game, side), stamp)
    for event in events(source):
        for value in event['values']:
            if value['source'] != 'unavoidable_death':
                continue
            for side in value['dead_sides']:
                key = (event['game_idx'], side)
                if key not in predicted or value['t_sec'] < predicted[key]['first_sec']:
                    predicted[key] = dict(source=source, game=key[0], side=side, first_sec=value['t_sec'],
                        bound=any(v.get('dead') and v['side'] == side for v in value.get('post_counter_bound', ())))
    cases = []
    for (game, side), row in predicted.items():
        labels = {r['winner'] for r in existing if r['source']==source and r['game_idx']==game and r['winner']}
        own, other = observed.get((game, side)), observed.get((game, '2P' if side=='1P' else '1P'))
        survived = side in labels or (own is None and other is not None)
        known = bool(labels) or own is not None or other is not None
        cases.append(dict(row, observed_death_sec=own, opponent_death_sec=other,
            winners=sorted(labels), false=survived, unresolved=not known,
            lead_sec=None if own is None else own-row['first_sec']))
    return cases


def audit_all(existing: list[dict]) -> dict:
    """重複するreviewも指定どおり別記録として全件検査する。"""
    helpers = prior.report_base.helpers
    games = read(helpers.ZENCHI/'official_games.json')
    existing = existing + helpers.death_rows(directory('review'), 'review', games)
    cases = [r for source in SOURCES for r in deaths(source, existing)]
    baseline_false = {(r['source'], r['game_idx'], r['side'])
                      for r in read(Path('logs/e32/DEATH_AUDIT.json')) if r['false_positive'] is True}
    bound_keys = {(s, r['game'], r['side']) for s in SOURCES
                  for r in read(directory(s)/'post_counter_bound_audit.json')['rows'] if r['dead']}
    for row in cases:
        key = (row['source'], row['game'], row['side'])
        row.update(baseline_false=key in baseline_false, bound_claim=key in bound_keys)
    result = dict(records=len(SOURCES), cases=len(cases), false=sum(r['false'] for r in cases),
        unresolved=sum(r['unresolved'] for r in cases),
        baseline_false=sum(r['false'] and r['baseline_false'] for r in cases),
        bound_false=sum(r['false'] and r['bound_claim'] for r in cases),
        bound_cases=sum(r['bound_claim'] for r in cases),
        lead_sec=quantiles([r['lead_sec'] for r in cases]), rows=cases)
    save_json(OUT/'DEATH_AUDIT_ALL.json', result)
    return result


def fallback_counts() -> dict:
    """非確定から実際に旧証明器へ委譲した回数と、旧入力ゲートの拒否を分ける。"""
    counts: Counter = Counter()
    for source in SOURCES:
        for event in events(source):
            for value in event['values']:
                for row in value.get('post_counter_bound', ()):
                    if row['dead'] or 'proofs' not in row:
                        continue
                    legacy = value['multi_landing'][int(row['side']=='2P')]
                    counts['search' if 'nodes' in legacy or 'proofs' in legacy else 'blocked'] += 1
    return dict(counts)


def bound_statistics() -> dict:
    """呼出し・キャッシュを除く計算・連鎖件数を別母数で保存する。"""
    rows = [dict(source=s, **r) for s in SOURCES
            for r in read(directory(s)/'post_counter_bound_audit.json')['rows']]
    groups = {(r['source'], r['game'], r['side'], r['chain_id']) for r in rows}
    proved = {(r['source'], r['game'], r['side'], r['chain_id']) for r in rows if r['dead']}
    fallbacks = fallback_counts()
    result = dict(calls=len(rows), bound=sum(r['dead'] for r in rows),
        undecided=sum(not r['dead'] for r in rows), fallback=fallbacks.get('search', 0),
        fallback_blocked=fallbacks.get('blocked', 0), chains=len(groups), proven_chains=len(proved),
        compute_sec=quantiles([r['elapsed_sec'] for r in rows if not r['cached']]),
        all_calls_sec=quantiles([r['elapsed_sec'] for r in rows]))
    save_json(OUT/'BOUND_STATISTICS.json', result)
    return result


def scene() -> dict:
    """3:00と初到達を全評価と同じ表示列から取り出す。"""
    times, display = displayed(directory('review')/'display.npz')
    index = int(np.argmin(abs(times-SCENE_STAMP)))
    data = np.load(directory('review')/'display.npz')
    rows = [v for e in events('review') for v in e['values'] if v['t_sec'] <= times[index]]
    latest = max(rows, key=lambda r:r['t_sec'])
    result = dict(first_sec=first(times, display), stamp=float(times[index]),
        p2_display=1-float(display[index]), p2_selected=1-float(data['display_p1'][index]),
        source=str(data['source'][index]), latest=latest)
    save_json(OUT/'SCENE.json', result)
    return result


def report() -> dict:
    """事前登録した全ゲートが合格したときだけ動画生成を許可する。"""
    if not (OUT/'OFF_review.json').exists():
        with (OUT/'off_review.log').open('a') as log:
            subprocess.run([sys.executable, '-B', '-m', 'scripts.run_e35', '--source', 'review', '--control'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    prior.OUT, prior.LOSS_MAX, prior.AGREEMENT_MIN = OUT, Q_MAX, ZENCHI_MIN
    value = prior.report('on')
    assert value['q']['frames'] == 6526 and value['zenchi']['frames'] == 8333
    existing = prior.report_base.death_metrics.deaths()
    audit, stats, target = audit_all(existing), bound_statistics(), scene()
    for source in SOURCES:
        suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
        before, after = np.load(Path('logs/e32/on')/suffix/'display.npz'), np.load(directory(source)/'display.npz')
        for key in ('t_sec', 'game_idx', 'state1', 'state2', 'score1', 'score2'):
            np.testing.assert_array_equal(before[key], after[key])
    value['gates'].update(false_certainty=audit['false']==0 and audit['unresolved']==0,
        scene=target['first_sec'] is not None and target['first_sec'] <= SCENE_DEADLINE)
    value.update(death_audit=audit, bound_statistics=stats, scene=target,
                 passed=all(value['gates'].values()), candidate=all(value['gates'].values()))
    save_json(OUT/'on/METRICS.json', {k:v for k,v in value.items() if k not in ('death_audit', 'bound_statistics')})
    save_json(OUT/'SUMMARY.json', value)
    return value


if __name__ == '__main__':
    print(json.dumps(report(), ensure_ascii=False), flush=True)
