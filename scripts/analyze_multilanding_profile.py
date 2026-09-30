"""profile_multilanding_speed の JSONL を、関数別・出典別の分布と遅い呼出し上位に集計する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PERCENTILES = (50, 95, 99)
SLOW_MS = 100.0
SLOWER_SEC = 1.0
TOP_N = 20


def load(root: Path) -> dict[str, list[dict]]:
    """出典名 -> 行リスト。"""
    return {p.stem.removeprefix('profile_'): [json.loads(l) for l in p.read_text().splitlines()]
            for p in sorted(root.glob('profile_*.jsonl'))}


def distribution(rows: list[dict]) -> dict:
    """秒の分布 (母数付き)。"""
    sec = np.array([r['sec'] for r in rows]) if rows else np.zeros(0)
    if not len(sec):
        return dict(n=0)
    out = dict(n=int(len(sec)), total_sec=float(sec.sum()), max_ms=float(sec.max()*1e3),
               over_100ms=int((sec > SLOW_MS/1e3).sum()), over_1s=int((sec > SLOWER_SEC).sum()))
    out.update({f'p{p}_ms': float(np.percentile(sec, p)*1e3) for p in PERCENTILES})
    return out


def summarize(data: dict[str, list[dict]]) -> dict:
    """関数別 x 出典別 (と全体) の分布。"""
    names = sorted({r['fn'] for rows in data.values() for r in rows})
    result = {}
    for name in names:
        per = {src: distribution([r for r in rows if r['fn'] == name]) for src, rows in data.items()}
        allrows = [r for rows in data.values() for r in rows if r['fn'] == name]
        result[name] = dict(per_source=per, all=distribution(allrows))
    return result


def top_slow(data: dict[str, list[dict]], fn: str, n: int = TOP_N) -> list[dict]:
    """指定関数の遅い呼出し上位。"""
    rows = [dict(source=s, **r) for s, rs in data.items() for r in rs if r['fn'] == fn]
    return sorted(rows, key=lambda r: -r['sec'])[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--top', default='projection_evaluate')
    args = parser.parse_args()
    data = load(args.root)
    summary = summarize(data)
    (args.root / 'DISTRIBUTION.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    for name, value in summary.items():
        print(name, json.dumps(value['all'], ensure_ascii=False))
    for row in top_slow(data, args.top):
        print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
