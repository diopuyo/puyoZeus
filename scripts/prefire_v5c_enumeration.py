"""固定100局面の候補列挙と重複統合を、価値評価から分けて測る。"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src import prefire_v5_search as base
from src import prefire_v5b_search as exact

OUT = Path('logs/prefire_prediction/v5c/enumeration.json')
SAMPLES = Path('logs/prefire_prediction/v5_experiment/samples_ledger.json')


def main() -> None:
    """全既知3手の列挙・Python状態生成・重複統合を含める。"""
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = []
    for index, row in enumerate(json.loads(SAMPLES.read_text())):
        states = tuple(base.Position(bytes(g), tuple(k), p)
                       for g, k, p in zip(row['boards'], row['known'], row['pending']))
        exact.candidates.cache_clear()
        start = perf_counter()
        options = tuple(exact.candidates(p, side=i) for i, p in enumerate(states))
        enumerated = perf_counter()
        unique = tuple(exact.unique(items) for items in options)
        finished = perf_counter()
        rows.append(dict(index=index, candidates=[len(v) for v in options],
                         unique=[len(v) for v in unique], enumeration_ms=(enumerated-start)*1000,
                         deduplication_ms=(finished-enumerated)*1000))
    elapsed = [r['enumeration_ms']+r['deduplication_ms'] for r in rows]
    result = dict(samples=len(rows), p95_ms=float(np.percentile(elapsed, 95)), rows=rows)
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(dict(samples=len(rows), p95_ms=result['p95_ms'], first=rows[0]), flush=True)


if __name__ == '__main__':
    main()
