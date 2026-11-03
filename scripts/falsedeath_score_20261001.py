"""D1〜D4事前登録の固定条件を、同じ採点器・固定勝敗で判定する。"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any
import numpy as np

from scripts.falsedeath_collect_20261001 import OUT, ROOT, EXEV, SOURCES
from scripts.falsedeath_replay_20261001 import MODES, input_record
from scripts._diag_set2_falsedeath_audit import scan, difference, events, read
from scripts import eval_set_score_20261001 as scorer, eval_set_noise_20261001 as noise

SETS = dict(set1=([f'c{i}' for i in range(1, 7)] + ['zenchi'], ROOT/'logs/eval_set/labels.json'),
            set2=([f's{i}' for i in range(7)], ROOT/'logs/eval_set/set2/labels.json'))
BOOT_N = 10_000
SEEDS = dict(set1=20261001, set2=0)
M_TIME_UPPER_MAX = .002
M_AGREE_LOWER_MIN = -.01


def save(name: str, value: Any) -> None:
    """数値原票を画像と分離して保存する。"""
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')


def paths(mode: str, parts: list[str]) -> list[Path]:
    """完了済み記録だけを採点する。"""
    result = [OUT/'replay'/mode/part for part in parts]
    assert all((p/'status.json').exists() and read(p/'status.json').get('state') == 'completed'
               for p in result), (mode, parts)
    return result


def terminal_rows(mode: str, part: str) -> list[dict]:
    """観測死亡を元の記録から取り、予測死亡による補完をしない。"""
    from src.exchange_event_record import read_records
    rows, seen = [], set()
    for item in read_records(input_record(mode, part)):
        if item['kind'] != 'update':
            continue
        result, _, _, stamp, game, *_ = item['args']
        for side in getattr(result, 'confirmed_dead_sides', ()):
            if (game, side) not in seen:
                seen.add((game, side))
                rows.append(dict(game=game, side=side, stamp=stamp))
    return rows


def distribution(values: list[float]) -> dict:
    """未観測をゼロ秒にしない。観測できた遅れの分位を返す。"""
    return dict(n=len(values), quantiles=dict(zip(('min', 'p25', 'p50', 'p75', 'p95', 'max'),
        map(float, np.percentile(values, (0, 25, 50, 75, 95, 100)))))) if values else dict(n=0)


def delays(change: dict, terminals: list[dict], games: list[dict] | None = None) -> dict:
    """消失・保持を基準側の事例キーで固定し、観測死亡までの遅れも列挙する。"""
    rows = []
    for old in change['correct_lost']:
        game = next((g for g in games or () if g['game'] == old['game']), None)
        stamps = [r['stamp'] for r in terminals if r['side'] == old['side'] and
                  (game['start'] <= r['stamp'] < game['end'] if game else r['game'] == old['game'])]
        observed = min(stamps) if stamps else None
        # 元の事例は監査由来の observed_death_sec を既に持つため、修正後の観測死亡は別名で残す。
        rows.append(dict(**old, candidate_observed_death_sec=observed,
                         delay_sec=None if observed is None else observed-old['first_sec']))
    return dict(lost=rows, lost_count=len(rows), missing_observed=sum(r['delay_sec'] is None for r in rows),
                lost_delay=distribution([r['delay_sec'] for r in rows if r['delay_sec'] is not None]),
                retained_delay=distribution([r['seconds'] for r in change['delays']]))


def score_set(name: str, mode: str, terminals: dict) -> dict:
    """セット別に同じ試合の対差を復元抽出する。時刻列の同一性も必須。"""
    parts, labels = SETS[name]
    games = read(labels)['games']
    source = {v: paths(v, parts) for v in ('prod', mode)}
    scores = {v: scorer.score([p/'display.npz' for p in pp], labels) for v, pp in source.items()}
    displays = {v: scorer.load_display([p/'display.npz' for p in pp]) for v, pp in source.items()}
    np.testing.assert_array_equal(displays['prod']['t_sec'], displays[mode]['t_sec'])
    index = np.random.default_rng(SEEDS[name]).integers(0, len(games), (BOOT_N, len(games)))
    metrics = {}
    for metric in ('m_time', 'm_agree'):
        values = {v: noise.per_game(s['rows'], metric) for v, s in scores.items()}
        metrics[metric] = dict(paired=noise.paired(values[mode], values['prod'], index),
                              **{v: float(noise.aggregate(a)) for v, a in values.items()})
    death = {v: scan(pp, games) for v, pp in source.items()}
    change = difference(death['prod'], death[mode])
    terminal = [r for part in parts for r in terminals[part]]
    return dict(games=len(games), metrics=metrics, deaths=change,
                correct_delay=delays(change, terminal, games), bootstrap=dict(n=BOOT_N, seed=SEEDS[name]),
                summaries={v: s['summary'] for v, s in scores.items()}, rows={v:s['rows'] for v,s in scores.items()})


def existing(mode: str, terminals: dict) -> dict:
    """既存5記録の固定ラベル・同じ監査器を維持する。"""
    from scripts import eval_set2_regression_20261001 as regression, report_e35 as auditor
    from src.exchange_event_record import read_records
    fixed = regression.fixed_labels()
    result = {}
    for variant in ('prod', mode):
        paths(variant, list(SOURCES))
        auditor.events = lambda source: events(OUT/'replay'/variant/source/'events.jsonl')
        auditor.read_records = lambda path: read_records(input_record(variant, path.name.removesuffix('.jsonl.gz')))
        result[variant] = [r for s in SOURCES for r in auditor.deaths(s, fixed)]
    changes = {}
    for source in SOURCES:
        change = difference(dict(rows=[r for r in result['prod'] if r['source']==source]),
                            dict(rows=[r for r in result[mode] if r['source']==source]))
        changes[source] = dict(change=change, correct_delay=delays(change, terminals[source]))
    return dict(sources=changes, before=len(result['prod']), after=len(result[mode]),
                false=sum(r['false'] for r in result[mode]), unresolved=sum(r['unresolved'] for r in result[mode]),
                rows=result)


def regression(mode: str) -> dict:
    """事前登録と同じ3:00の既存指標・q第14試合を採点する。"""
    from scripts import eval_set2_regression_20261001 as prior, report_e36b
    original = prior.directory
    prior.directory = lambda source: OUT/'replay'/mode/source
    try:
        scene = prior.scene()
        game14 = report_e36b.game14_false_rows(events(prior.directory('q_7gc4TgFig')/'events.jsonl'))
    finally:
        prior.directory = original
    return dict(scene=scene, game14_false_times=game14, passed=scene['passed'] and not game14)


def main() -> None:
    """全修正だけで合否を判定。一つ抜きは参考値として独立に保存する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('legacy', 'v2', 'v2b', 'v2c', 'v2d', 'v2e', 'v2f', 'v2g', 'v2h', 'v2h2', 'v2h2b'), default='legacy')
    mode = parser.parse_args().mode
    if mode in ('v2', 'v2b', 'v2c', 'v2d', 'v2e', 'v2f', 'v2g', 'v2h', 'v2h2', 'v2h2b'):
        from scripts.falsedeath_v2_score_20261002 import main as score_v2
        score_v2(mode)
        return
    from scripts._diag_set2_falsedeath_batch import records
    terminal = {mode: {part: terminal_rows(mode, part) for part in records()} for mode in ('prod', 'all')}
    save('TERMINALS.json', terminal)
    save('TERMINAL_IDENTITY.json', {part: terminal['prod'][part] == terminal['all'][part] for part in records()})
    results = {}
    for mode in (m for m in MODES[1:] if m not in ('v2', 'v2b', 'v2c', 'v2d', 'v2e', 'v2f', 'v2g', 'v2h', 'v2h2', 'v2h2b')):
        observed = terminal['prod' if mode == 'no_d4' else 'all']
        sets = {name: score_set(name, mode, observed) for name in SETS}
        old, target = existing(mode, observed), regression(mode)
        gates = dict(false_death=old['false'] == 0 and old['unresolved'] == 0 and all(
            s['deaths']['after']['false'] == s['deaths']['after']['false_frames'] == 0 for s in sets.values()),
            m_time=all(s['metrics']['m_time']['paired']['ci95'][1] < M_TIME_UPPER_MAX for s in sets.values()),
            m_agree=all(s['metrics']['m_agree']['paired']['ci95'][0] > M_AGREE_LOWER_MIN for s in sets.values()),
            regression=target['passed'])
        results[mode] = dict(sets=sets, existing5=old, regression=target, gates=gates, passed=all(gates.values()))
        save(f'RESULT_{mode}.json', results[mode])
    save('RESULT.json', results)
    print(json.dumps(dict(passed=results['all']['passed'], gates=results['all']['gates']), ensure_ascii=False))


if __name__ == '__main__':
    main()
