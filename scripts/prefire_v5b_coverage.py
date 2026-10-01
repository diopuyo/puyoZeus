"""同一5A診断行での修正後欠測と、100局面のnative候補数を測る。"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.prefire_v5b_missing import OLD, RECORDS, BASELINE_DIRS
from src.exchange_event_record import read_records
from src.prefire_v5b_queue import SideQueueV5B
from src import prefire_v5_search as base
from src import prefire_v5b_search as exact

OUT = Path('logs/prefire_prediction/v5b')


def coverage(source: str) -> dict:
    """適用行数ではなく、同一母数での入力欠測を測る。予測精度とは扱わない。"""
    saved = np.load(OLD/source/'prefire_trace.npz')
    columns, rows = list(saved['columns']), saved['values']
    times = {round(float(r[0]), 6): r for r in rows}
    queues, boards, game = [SideQueueV5B(), SideQueueV5B()], [None, None], None
    counts, missing_any, eligible, matched = [Counter(), Counter()], 0, 0, 0
    for row in read_records(RECORDS/f'{source}.jsonl.gz'):
        if row['kind'] != 'update':
            continue
        result, _, _, stamp, current = row['args'][:5]
        if game != current:
            queues, boards, game = [SideQueueV5B(), SideQueueV5B()], [None, None], current
        for side, data in enumerate((result.p1, result.p2)):
            if data.state.value == 'stable' and data.confirmed_board is not None:
                grid = getattr(data.confirmed_board, '_grid', data.confirmed_board)
                boards[side] = np.asarray(grid, dtype=np.int8).tobytes()
                queues[side].observe(stamp, boards[side], np.array([
                    *(data.next_pair or (0, 0)), *(data.dnext_pair or (0, 0))]))
        if round(float(stamp), 6) not in times:
            continue
        matched += 1
        statuses = [base.status(b, q.known()) if b is not None else base.MISSING
                    for b, q in zip(boards, queues)]
        missing_any += base.MISSING in statuses
        eligible += statuses == [base.OK, base.OK]
        for side, queue in enumerate(queues):
            counts[side][f'status_{statuses[side]}'] += 1
            counts[side][f'known_hands_{exact.known_depth(queue.known())}'] += 1
    return dict(denominator=len(rows), matched=matched, missing_any=missing_any,
                eligible_both=eligible, by_side=counts)


def census() -> dict:
    """native列挙時間と、価値評価の直積の大きさを分けて保存する。"""
    samples = json.loads(Path('logs/prefire_prediction/v5_experiment/samples_ledger.json').read_text())
    results = []
    for index, row in enumerate(samples):
        states = [base.Position(bytes(g), tuple(k), p)
                  for g, k, p in zip(row['boards'], row['known'], row['pending'])]
        start = perf_counter()
        counts = [len(exact.candidates(p, side=i)) for i, p in enumerate(states)]
        results.append(dict(index=index, count=counts, matrix_size=counts[0]*counts[1],
                            milliseconds=(perf_counter()-start)*1000))
    return dict(samples=len(results), results=results,
                p95_ms=float(np.percentile([r['milliseconds'] for r in results], 95)))


def main() -> None:
    """母数が同じ診断比較を出し、採点済みという表示は付けない。"""
    result = {source: coverage(source) for source in BASELINE_DIRS}
    (OUT/'coverage.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)
    measured = census()
    (OUT/'enumeration_timing.json').write_text(json.dumps(measured, indent=2), encoding='utf-8')
    print('enumeration_p95_ms', measured['p95_ms'], flush=True)


if __name__ == '__main__':
    main()
