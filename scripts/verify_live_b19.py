"""B19の実記録再構築と盤面コピーコストを再現可能な原票へ保存する。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from time import perf_counter_ns
from typing import Any
from unittest.mock import patch

import numpy as np

from scripts.verify_live_b18a import compare_rows
from scripts.verify_live_b18b import save
from scripts.run_live_pipeline_20260928 import compare_arrays
from src.phase_j.live_eval_supervisor import SupervisedOverlay, EvaluationError

OUT = Path('logs/live_b19')
WARMUP, SAMPLES, NS_PER_MS = 100, 2000, 1_000_000


def copy_cost() -> None:
    """1080pの両盤面領域をコピーする追加処理だけを同じ入力で測定する。"""
    from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
    frame = np.ones((1080, 1920, 3), dtype=np.uint8)
    crops = [frame[r.y:r.y+r.height, r.x:r.x+r.width]
             for r in (DEFAULT_P1_REGION, DEFAULT_P2_REGION)]
    timings = []
    for index in range(WARMUP+SAMPLES):
        begin = perf_counter_ns()
        owned = [crop.copy() for crop in crops]
        elapsed = (perf_counter_ns()-begin)/NS_PER_MS
        if index >= WARMUP:
            timings.append(elapsed)
    report = dict(samples=SAMPLES, bytes_per_frame=sum(c.nbytes for c in owned),
                  p50_ms=float(np.percentile(timings, 50)), p95_ms=float(np.percentile(timings, 95)))
    save(OUT/'copy-cost.json', report)
    assert report['p95_ms'] <= .3, report


def traced(dest: Path, boundary: int, stage: str, instances: list, rows: list) -> type:
    class Measured(SupervisedOverlay):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs, directory=dest/'supervisor')
            self.auto_acknowledge = False
            self.seen_games: list[int] = []
            self.inject_next = False
            instances.append(self)

        def request(self, request: dict) -> dict:
            if self.inject_next and request['op'] == stage:
                self.inject_next = False
                request = dict(request, fault='B19境界故障注入')
            return super().request(request)

        def update(self, *args: Any, **kwargs: Any) -> None:
            if args[4] not in self.seen_games:
                self.seen_games.append(args[4])
                self.inject_next = len(self.seen_games) == boundary
            try:
                super().update(*args, **kwargs)
                self.calculate()
            except EvaluationError:
                self.calculate()
            rows.append(dict(t_sec=args[3], game=args[4], p1=self.tracker.probability,
                             source=self.tracker.source, display=list(self.display or (0., .5))))
            self.succeeded()
    return Measured


def replay(boundary: int, stage: str) -> None:
    from scripts import replay_exchange_event_20260926 as cli
    from src.phase_j.live_notification_eval import latest_display
    dest = OUT/f'fault-{boundary}-{stage}'
    instances, rows = [], []
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(cli, 'ExchangeEventOverlay',
                                             traced(dest, boundary, stage, instances, rows)))
            stack.enter_context(patch('scripts.visualize_advantage_overlay._exchange_display', latest_display))
            stack.enter_context(patch.object(sys, 'argv', ['replay', 'logs/e31/records/review.jsonl.gz',
                '--out', str(dest), '--production-exchange-event']))
            cli.main()
        expected = Path('logs/live_b18c/saved/review/offline')
        events = [json.loads(line) for line in (dest/'events.jsonl').read_text().splitlines()]
        ids = [row['exchange_id'] for row in events]
        report = dict(events=len(events), ids=ids,
            events_identical=(dest/'events.jsonl').read_bytes() == (expected/'events.jsonl').read_bytes(),
            probabilities=compare_rows(json.loads((expected/'probabilities.json').read_text()), rows),
            display=compare_arrays(expected/'display.npz', dest/'display.npz'),
            errors=instances[0].error_count, restarts=instances[0].restarts)
        save(dest/'report.json', report)
        assert report['events_identical'] and ids == list(range(1, 14)), report
        assert report['probabilities']['equal'] == report['probabilities']['total'], report
    finally:
        for instance in instances:
            instance.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boundary', type=int, choices=(2, 3), default=2)
    parser.add_argument('--stage', choices=('advance', 'batch'), default='advance')
    parser.add_argument('--copy-cost', action='store_true')
    args = parser.parse_args()
    if args.copy_cost:
        copy_cost()
    else:
        replay(args.boundary, args.stage)


if __name__ == '__main__':
    main()
