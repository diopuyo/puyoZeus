"""再生入力の全更新を同期版へビット照合する。性能ベンチではない。"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from queue import Empty
import struct
from time import monotonic

import numpy as np

from src.b3_async import AsyncB3
from src.b3_realtime import RealtimeRows
from src.b3_realtime_replay import replay_inputs
from src.b3_source import resolve_attempt, sha256_file
from src.b3_timing import TimingTable
from src.b3_train_model import load_model

DRAIN_TIMEOUT_SEC = 1800.
PROGRESS_INTERVAL = 500


def bits(row: dict) -> str:
    """dtypeに依らず値の浮動小数点ビットを比較。NaNもビットを保持する。"""
    digest = sha256()
    for key, value in sorted(row.items()):
        digest.update(key.encode()+b'\0')
        if isinstance(value, (float, np.floating)):
            digest.update(b'f'+struct.pack('!d', float(value)))
        else:
            digest.update(json.dumps(value, default=int, ensure_ascii=False).encode())
        digest.update(b'\0')
    return digest.hexdigest()


def synchronous(items: list, table: TimingTable, model: object, video: str) -> dict:
    engine = RealtimeRows(table, video, measure_timing=False)
    expected = {}
    for item in items:
        row = engine.update(*item)
        if row is not None:
            p = model.predict_row(row)
            expected[row['frame_idx']] = (bits(row), struct.pack('!d', p).hex())
            if len(expected) % PROGRESS_INTERVAL == 0:
                print(f'同期: {len(expected)}更新', flush=True)
    return expected


def asynchronous(items: list, args: argparse.Namespace, table: TimingTable,
                 expected: dict, drop: bool) -> dict:
    engine = AsyncB3(args.model, table, args.video, drop=drop, measure=False, include_rows=True)
    seen, differences, failures = set(), [], []
    counts = None
    try:
        for item in items:
            engine.submit(*item, epoch=1)
        engine.finish()
        deadline = monotonic()+DRAIN_TIMEOUT_SEC
        while counts is None:
            if monotonic() > deadline:
                raise TimeoutError('一致確認の末尾待ちが期限を超えました')
            try:
                result = engine.outputs.get(timeout=1.)
            except Empty:
                if not engine.process.is_alive():
                    raise RuntimeError('末尾通知なしで状態processが終了しました')
                continue
            if 'fatal' in result:
                raise RuntimeError(result['fatal'])
            if 'done' in result:
                counts = result['done']
                continue
            key = result['frame']
            if result['error'] is not None:
                failures.append(dict(frame=key, error=result['error']))
            elif (bits(result['row']), struct.pack('!d', result['raw_probability']).hex()) != expected[key]:
                differences.append(key)
            if key in seen:
                raise AssertionError(f'更新の重複: {key}')
            seen.add(key)
            if len(seen) % PROGRESS_INTERVAL == 0:
                print(f'非同期 drop={drop}: {len(seen)}更新', flush=True)
    finally:
        engine.close()
    return dict(drop=drop, counts=counts, computed=len(seen), differences=differences,
                failures=failures, missing=len(set(expected)-seen))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'model', 'timing-table', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--video', default='b3_zenchi_s1')
    parser.add_argument('--max-frame', type=int, default=36000)
    args = parser.parse_args()
    record = resolve_attempt(args.video, args.root).record
    items = list(replay_inputs(record, args.max_frame))
    table = TimingTable.from_json(args.timing_table.read_text(encoding='utf-8'))
    expected = synchronous(items, table, load_model(args.model), args.video)
    runs = [asynchronous(items, args, table, expected, drop) for drop in (False, True)]
    passed = all(not r['differences'] and not r['failures'] and r['computed'] > 0
                 and r['counts']['inputs'] == len(items) for r in runs) and runs[0]['missing'] == 0
    result = dict(passed=passed, video=args.video, max_frame=args.max_frame,
        inputs=len(items), synchronous_updates=len(expected), timing_measured=False,
        model_sha256=sha256_file(args.model), table_sha256=sha256_file(args.timing_table), runs=runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
