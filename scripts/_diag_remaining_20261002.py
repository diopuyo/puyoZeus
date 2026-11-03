"""既存成果物を変更せず、残存事例と外し比較の原票を抽出する。"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path('/mnt/d/puyo_analyzer/wt_evalset')
BASE = ROOT / 'logs/eval_set/falsedeath_fix'
OUT = BASE / 'diag_remaining_2026-10-02'
MIN_MEMORY = 6 * 1024**3
MODES = ('prod', 'all', 'no_d1', 'no_d2', 'no_d3', 'no_d4')
WINDOW = 12.0


def read(path: Path) -> Any:
    """既存JSONを読む。"""
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(name: str, value: Any) -> None:
    """新規の原票のみを保存する。"""
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=1)


def guard() -> None:
    """優先度と使用可能メモリの指定条件を検査する。"""
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0]) * 1024
    assert available >= MIN_MEMORY, available
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    OUT.mkdir(exist_ok=True)
    save('resources.json', dict(available_bytes=available, nice=19))


def extract_events(source: str, stamp: float, tag: str) -> None:
    """前後の判定系列と該当交換の最終得点を抽出する。"""
    for mode in MODES:
        directory = BASE / 'replay' / mode / source
        selected = []
        for line in (directory / 'events.jsonl').open():
            event = json.loads(line)
            values = [v for v in event['values'] if stamp-WINDOW <= v['t_sec'] <= stamp+WINDOW]
            if values:
                selected.append(dict(event, values=values))
        with np.load(directory / 'display.npz') as display:
            index = int(np.argmin(abs(display['t_sec']-stamp)))
            near = {k: display[k][index].item() for k in display.files if display[k].ndim == 1}
        save(f'{tag}_{mode}_events.json', dict(events=selected, nearest_display=near,
             diagnostics=read(directory / 'events.diagnostics.json')))


def losses(result: dict) -> None:
    """全修正で消えた事例に、各一つ抜きでの復活時刻を付ける。"""
    rows = []
    for name, data in result['all']['sets'].items():
        for old in data['correct_delay']['lost']:
            row = dict(set=name, **old)
            row['restored'] = {}
            for mode in MODES[2:]:
                found = next((r for r in result[mode]['sets'][name]['deaths']['after']['rows']
                              if (r['game'], r['side']) == (old['game'], old['side'])), None)
                row['restored'][mode] = found['first_sec'] if found else None
            rows.append(row)
    save('lost_attribution.json', rows)
    print('LOST', json.dumps({s: result['all']['sets'][s]['correct_delay'] for s in ('set1', 'set2')}))


def main() -> None:
    """対象イベントと消失事例を既存原票から取り出す。"""
    guard()
    result = read(BASE / 'RESULT.json')
    false = [dict(set='set1', **r) for r in result['all']['sets']['set1']['deaths']['after']['rows'] if r['false']]
    false += [dict(set='existing5', **r) for r in result['all']['existing5']['rows']['all'] if r['false']]
    for row in false:
        if 'source' not in row:
            for directory in (BASE / 'replay/all').glob('c*'):
                for line in (directory / 'events.jsonl').open():
                    if any(abs(v['t_sec']-row['first_sec']) < 1e-6 and row['side'] in v.get('dead_sides', [])
                           for v in json.loads(line)['values']):
                        row['source'] = directory.name
        extract_events(row['source'], row['first_sec'], f"{row['source']}_g{row['game']}")
    save('false_cases.json', false)
    extract_events('zenchi', 2766.0, 'scene')
    losses(result)
    print('FALSE', json.dumps(false))


if __name__ == '__main__':
    main()
