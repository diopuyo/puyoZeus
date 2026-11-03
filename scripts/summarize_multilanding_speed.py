"""profile_*.jsonl (基準 / 変更後) の通知 1 件あたり所要を、出典別・関数別に母数つきで比較して表にする。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SOURCES = ('q_7gc4TgFig', 'review', 'fcXG83vInDY', 'mia8KCjr52g', 'zenchi',
           'live_b18_stall', 'live_b18_run5', 'live_b20_on')
PERCENTILES = (50, 95, 99)
THRESHOLDS = (0.1, 0.2, 1.0, 5.0)
FUNCTIONS = ('projection_evaluate', 'prove_multilanding', 'evaluate_multilanding',
             'post_counter_evaluate', 'weighted_landing', 'future_send_miss')


def load(root: Path, source: str) -> list[dict]:
    path = root / f'profile_{source}.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def stats(seconds: list[float]) -> dict:
    """母数・分位・最大・しきい値超えの件数。"""
    a = np.asarray(seconds)
    if not len(a):
        return dict(n=0)
    row = dict(n=len(a), total=float(a.sum()), max_ms=float(a.max() * 1e3))
    row.update({f'p{p}_ms': float(np.percentile(a, p) * 1e3) for p in PERCENTILES})
    row.update({f'over_{int(t * 1e3)}ms': int((a > t).sum()) for t in THRESHOLDS})
    return row


def table(root: Path) -> dict:
    result = {}
    for source in SOURCES:
        rows = load(root, source)
        if rows:
            result[source] = {fn: stats([r['sec'] for r in rows if r['fn'] == fn]) for fn in FUNCTIONS}
    return result


def main() -> None:
    before, after = table(Path(sys.argv[1])), table(Path(sys.argv[2]))
    out = dict(before=before, after=after)
    Path(sys.argv[3]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for source in SOURCES:
        if source not in before or source not in after:
            continue
        for fn in ('projection_evaluate',):
            b, a = before[source][fn], after[source][fn]
            print(f"{source:16s} {fn:20s} n={b['n']:5d} | 前 P50 {b['p50_ms']:7.1f} P95 {b['p95_ms']:7.1f} P99 {b['p99_ms']:7.1f} max {b['max_ms']:8.1f} "
                  f">0.1s {b['over_100ms']:3d} >1s {b['over_1000ms']:3d} 合計 {b['total']:6.1f}s | 後 P50 {a['p50_ms']:7.1f} P95 {a['p95_ms']:7.1f} "
                  f"P99 {a['p99_ms']:7.1f} max {a['max_ms']:8.1f} >0.1s {a['over_100ms']:3d} >1s {a['over_1000ms']:3d} 合計 {a['total']:6.1f}s")


if __name__ == '__main__':
    main()
