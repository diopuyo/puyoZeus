"""既存原票から消失例の判定系列を抽出する。元成果物は読み取り専用。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path('/mnt/d/puyo_analyzer/wt_evalset')
BASE = ROOT / 'logs/eval_set/falsedeath_fix'
OUT = BASE / 'diag_dv2d_loss_2026-10-02'
MIN_MEMORY = 6 * 1024**3
EPS = 0.04


def read(path: Path) -> Any:
    """UTF-8原票を読む。"""
    return json.loads(path.read_text(encoding='utf-8'))


def save(name: str, value: Any) -> None:
    """新規ファイルだけを作る。"""
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=1)


def resource() -> dict:
    """指定された優先度と空きメモリを実測する。"""
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0])*1024
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    assert available >= MIN_MEMORY and nice == 10, (available, nice)
    return dict(available_gib=available/1024**3, nice=nice)


def events(source: str, mode: str = 'v2d') -> list[dict]:
    """イベントを時系列に平坦化する。"""
    path = BASE / 'replay' / mode / source / 'events.jsonl'
    rows = []
    for line in path.open(encoding='utf-8'):
        event = json.loads(line)
        rows.extend(dict(game=event['game_idx'], **v) for v in event['values'])
    return sorted(rows, key=lambda r: r['t_sec'])


def targets() -> list[dict]:
    """55件と、外れ値比較用の重複しない既存記録を分けて読む。"""
    result = read(BASE / 'v2d/RESULT_v2d.json')
    groups = {name: r['correct_delay']['lost'] for name, r in result['sets'].items()}
    groups.update({name: result['existing5']['sources'][name]['correct_delay']['lost']
                   for name in ('q_7gc4TgFig', 'fcXG83vInDY')})
    rows = [dict(group=group, **r) for group, rr in groups.items() for r in rr]
    assert sum(r['group'] in ('set1', 'set2') for r in rows) == 55
    return rows


def enrich(row: dict, cache: dict) -> dict:
    """時刻に対応する入力記録と実際の判定系列を確定する。"""
    parts = ([f'c{i}' for i in range(1, 7)]+['zenchi'] if row['group'] == 'set1'
             else [f's{i}' for i in range(7)] if row['group'] == 'set2' else [row['group']])
    t, end, side = row['first_sec'], row['candidate_observed_death_sec'], row['side']
    candidates = []
    for part in parts:
        old, new = cache[part]
        hits = [r for r in old if abs(r['t_sec']-t) < EPS and r['source'] == 'unavoidable_death'
                and side in r.get('dead_sides', [])]
        if hits:
            candidates.append((part, hits[0], new))
    assert candidates, row
    part, original, new = min(candidates, key=lambda v: abs(v[1]['t_sec']-t))
    near = [r for r in new if r['t_sec'] <= t+EPS and 'post_counter_bound' in r]
    first = near[-1] if near else None
    series = [r for r in new if t-EPS <= r['t_sec'] <= end+EPS]
    return dict(**row, record=part, local_game=original['game'], original=original,
                at_first=first, series=series)


def main() -> None:
    """再現に必要な既存判定を保存し、標準出力は小さく保つ。"""
    OUT.mkdir(parents=True, exist_ok=True)
    save('RESOURCE_inventory.json', resource())
    rows = targets()
    parts = [f'c{i}' for i in range(1, 7)]+[f's{i}' for i in range(7)]
    parts += ['zenchi', 'q_7gc4TgFig', 'fcXG83vInDY']
    cache = {p: (events(p, 'prod'), events(p)) for p in parts}
    enriched = [enrich(r, cache) for r in rows]
    save('EVENTS.json', enriched)
    for r in enriched:
        a = r['at_first'] or {}
        print(json.dumps({k:r[k] for k in ('group','game','side','record','local_game','delay_sec')}
              | dict(first=a), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
