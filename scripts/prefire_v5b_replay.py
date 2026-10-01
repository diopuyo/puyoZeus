"""通知時間からLを固定し、5記録とzenchiセット1を同じ遅れで再生する。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.run_prefire_replay_20260930 import RECORDS, BASELINE_DIRS, options, model_directory
from scripts.prefire_v5b_deferred import DeferredComputations

OUT = Path('logs/prefire_prediction/v5b')
SET1 = Path('/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/collect/records')
TIMING_INDICES = (0, 25, 44, 71, 77)
MILLISECONDS_PER_TENTH = 100.0


def freeze_latency() -> dict:
    """5A系の規則どおり、通知のP95を0.1秒単位で切り上げ、採点前に保存する。"""
    path = OUT/'latency.json'
    if path.exists():
        return json.loads(path.read_text())
    paths = [OUT/'notifications'/f'benchmark_{i:03d}.json' for i in TIMING_INDICES]
    costs = [json.loads(p.read_text())['milliseconds'] for p in paths]
    p95 = float(np.percentile(costs, 95))
    result = dict(notifications=len(costs), indices=list(TIMING_INDICES), milliseconds=costs, p95_ms=p95,
                  latency_sec=float(np.ceil(p95/MILLISECONDS_PER_TENTH)/10),
                  within_200ms=p95 <= 200, sampling='5A固定100局面の各記録の先頭1件・cold cache',
                  limitation='5件の事前診断。全通知分布のP95を保証する標本ではない。',
                  sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def run(source: str) -> dict:
    """固定Lで仮想再生する。取消しは適用成功にも計算完了にも数えない。"""
    from scripts import d5_runtime
    from scripts.replay_exchange_event_20260926 import replay
    latency = freeze_latency()
    dest = OUT/'replay'/source
    if (dest/'DONE.json').exists():
        raise FileExistsError('完了済みの再生を上書きしない')
    dest.mkdir(parents=True, exist_ok=True)
    d5_runtime.OUT = dest/'runtime'
    d5_runtime.install()
    adapter = DeferredComputations()
    adapter.install()
    record = (SET1 if source.startswith('c') else RECORDS)/f'{source}.jsonl.gz'
    if source == 'short':
        record = Path('logs/prefire_prediction/v5_short/input.jsonl.gz')
    opts = options('v5', latency['latency_sec'])
    try:
        result = replay(record, dest, model_directory(opts), True, None, **opts)
    finally:
        adapter.restore()
    result.update(options=opts, deferred=adapter.counts(), input=str(record),
                  latency_measurement=str(OUT/'latency.json'))
    (dest/'DONE.json').write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
    return result


def main() -> None:
    """短区間を先に通し、全記録は記録別に起動する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=(*BASELINE_DIRS, 'short', *(f'c{i}' for i in range(1, 7))))
    args = parser.parse_args()
    print(json.dumps(run(args.source), default=str), flush=True)


if __name__ == '__main__':
    main()
