"""D3の保存記録を読み、観測可能な合図と書込みの原票を抽出する。"""
from __future__ import annotations

from collections import Counter
import gzip
import json
import os
from pathlib import Path
from typing import Any

OUT = Path('logs/d3')
SOURCES = ('q_7gc4TgFig', 'fcXG83vInDY', 'mia8KCjr52g', 'zenchi', 'review')
NICE = 19


def decode(value: Any) -> Any:
    """原票の格納型だけを展開する。"""
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    for key in ('board', 'array', 'namespace', 'tuple'):
        if key in value:
            return decode(value[key])
    return {k: decode(v) for k, v in value.items()}


def records(source: str) -> Any:
    """更新列を逐次読み、認識・評価コードを実行しない。"""
    with gzip.open(f'logs/e31/records/{source}.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] == 'update':
                yield decode(row)['args']


def run(source: str) -> dict:
    """全更新の盤面・合図を欠落なく縮約保存する。"""
    counts: Counter = Counter()
    first, last, previous = None, None, None
    with gzip.open(OUT/f'{source}_records.jsonl.gz', 'wt') as stream:
        for args in records(source):
            result, snapshot, final, stamp, game, totals, scores, formula = args
            first = stamp if first is None else first
            if previous is not None:
                counts[f'dt_{stamp-previous:.6f}'] += 1
            row = dict(t=stamp, game=game, formula=formula, sides=[result[k] for k in ('p1', 'p2')])
            stream.write(json.dumps(row, separators=(',', ':'))+'\n')
            for side in row['sides']:
                counts['state_'+side['state']['state']] += 1
                counts['next_'+str(side['next_slide_motion'])] += 1
            counts['rows'] += 1
            last, previous = stamp, stamp
    return dict(first=first, last=last, counts=counts, sample=row)


def main() -> None:
    """資源優先度を下げ、5記録を直列で抽出する。"""
    os.nice(NICE)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for source in SOURCES:
        result[source] = run(source)
        print(source, json.dumps(result[source], ensure_ascii=False), flush=True)
    (OUT/'inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
