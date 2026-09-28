"""保存入力の同一通知・公開時刻で、障害隔離のまとめ更新を切り分ける。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import time

import numpy as np

from scripts.measure_live_b6 import save
from src.exchange_event_record import read_records, encode

ROOT = Path('logs/live_b17')
SOURCE = Path('logs/live_b16/realtime/inputs.jsonl.gz')
START, SECONDS, MS = 2580.566, 300.0, 1000


def extract(destination: Path, start: float, end: float) -> None:
    """区間開始を含む試合の冒頭から保持し、静止特徴も保存する。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    header, pending, frames, began = None, [], 0, False
    game = None
    with gzip.open(destination, 'wt') as output:
        for row in read_records(SOURCE):
            if row['kind'] == 'header':
                header = row
                output.write(json.dumps(encode(header))+'\n')
                continue
            if row['kind'] == 'update':
                sec, current = row['args'][3:5]
                if sec >= end:
                    break
                if not began and current != game:
                    pending.clear()
                game = current
                began |= sec >= start
            if row['kind'] == 'complete':
                continue
            pending.append(row)
            if began:
                for item in pending:
                    frames += item['kind'] == 'update'
                    output.write(json.dumps(encode(item))+'\n')
                pending.clear()
        output.write(json.dumps(dict(kind='complete', frames=frames))+'\n')


def replay(path: Path, mode: str, output: Path, start: float) -> dict:
    from scripts.replay_live_b16 import make_overlay
    from src.phase_j.live_cache import bounded_chain_caches
    overlay = make_overlay(path, True, output, mode != 'direct')
    if mode in ('batched', 'incremental'):
        from scripts.measure_live_b17 import batched_update
        from types import MethodType
        overlay.update = MethodType(batched_update, overlay)
    rows = list(read_records(path))
    attempts = {r['t_sec'] for r in rows if r['kind'] == 'display'}
    measurements, values = [], []
    with ExitStack() as cleanup, bounded_chain_caches():
        if mode != 'direct':
            cleanup.callback(overlay.close)
        for row in rows:
            if row['kind'] != 'update':
                continue
            inputs = row['args']
            before = time.perf_counter()
            overlay.update(*inputs)
            if mode == 'incremental':
                overlay.connection.send(dict(op='updates', commands=overlay.pending))
                reply = overlay.receive()
                if reply['kind'] == 'error':
                    raise RuntimeError(reply)
                overlay.archive.extend(reply['sealed'])
                overlay.diagnostics.extend(reply['diagnostics'])
                overlay.pending.clear()
            updated = time.perf_counter()
            if inputs[3] in attempts:
                overlay.calculate()
                finished = time.perf_counter()
                values.append(dict(t_sec=inputs[3], source=overlay.tracker.source,
                                   p1=overlay.tracker.probability))
                if inputs[3] >= start:
                    measurements.append(dict(t_sec=inputs[3], update_ms=(updated-before)*MS,
                        calculation_ms=(finished-updated)*MS, total_ms=(finished-before)*MS))
    result = dict(mode=mode, measurements=measurements, values=values,
        percentiles={key: np.percentile([r[key] for r in measurements], [50, 95, 99]).tolist()
                     for key in ('update_ms', 'calculation_ms', 'total_ms')})
    save(output, result)
    return result['percentiles']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('direct', 'batched', 'incremental', 'fixed'), required=True)
    parser.add_argument('--start', type=float, default=START)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    record = args.output/'inputs.jsonl.gz'
    if not record.exists():
        extract(record, args.start, args.start+SECONDS)
    print(json.dumps(replay(record, args.mode, args.output/(args.mode+'.json'), args.start)))


if __name__ == '__main__':
    main()
