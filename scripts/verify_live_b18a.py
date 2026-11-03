"""B18a: 保存入力だけで本番CLIとライブ評価の同値性・相対時間を測る。"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from src.exchange_event_record import read_records
from src.phase_j.live_evaluation import SplitExchangeOverlay

MILLISECONDS = 1000.0
PERCENTILES = (50, 95)
DEFAULT_RECORD = Path('logs/e31/records/review.jsonl.gz')
DEFAULT_OUTPUT = Path('logs/live_b18a')
PROGRESS_FRAMES = 10000


def audit_input(path: Path) -> dict:
    """欠けた画像由来観測を陰性として補完せず、存在件数だけを数える。"""
    counts: Counter = Counter()
    for row in read_records(path):
        counts[row['kind']] += 1
        if row['kind'] != 'update':
            continue
        result = row['args'][0]
        counts['terminal_evidence'] += bool(getattr(result, 'terminal_evidence_available', False))
        for side in (result.p1, result.p2):
            for name in ('prefire_snapshot', 'midchain_board'):
                counts[name] += getattr(side, name, None) is not None
    return dict(path=str(path), **counts)


def quantiles(samples: list[float]) -> dict:
    """入力復号・モデル読込を除いた評価一回のwall time。"""
    return dict(count=len(samples), **{f'P{p}_ms': float(np.percentile(samples, p))*MILLISECONDS
                                     for p in PERCENTILES})


def replay_cli(record: Path, output: Path, live: bool) -> dict:
    """実際の--production-exchange-event解析を使い、評価器だけを交換する。"""
    from scripts import replay_exchange_event_20260926 as replay
    from scripts.visualize_advantage_overlay import _ExchangeDisplayEMA
    from src.phase_j.live_evaluation import sampled_ema
    from types import SimpleNamespace
    output.mkdir(parents=True, exist_ok=True)
    timings: list[float] = []
    probabilities: list[dict] = []
    base = SplitExchangeOverlay if live else replay.ExchangeEventOverlay
    bridge = SimpleNamespace(notification_count=0)

    class Measured(base):
        def update(self, *args: Any, **kwargs: Any) -> None:
            started = time.perf_counter()
            super().update(*args, **kwargs)
            if live:
                self.calculate()
            timings.append(time.perf_counter()-started)
            bridge.notification_count += 1
            probabilities.append(dict(t_sec=args[3], game=args[4],
                                      p1=self.tracker.probability, source=self.tracker.source))

    with ExitStack() as stack:
        stack.enter_context(patch.object(replay, 'ExchangeEventOverlay', Measured))
        stack.enter_context(patch.object(sys, 'argv', ['replay', str(record), '--out', str(output),
                                                      '--production-exchange-event']))
        if live:
            stack.enter_context(patch('scripts.visualize_advantage_overlay._ExchangeDisplayEMA',
                                      sampled_ema(_ExchangeDisplayEMA, bridge)))
        replay.main()
    (output/'probabilities.json').write_text(json.dumps(probabilities), encoding='utf-8')
    return dict(timing=quantiles(timings), probabilities=probabilities)


def compare_rows(left: list[dict], right: list[dict]) -> dict:
    """数値と由来の完全一致を行単位で数える。"""
    pairs = list(zip(left, right))
    first = next((dict(index=i, offline=a, realtime=b) for i, (a, b) in enumerate(pairs)
                  if a != b), None)
    return dict(equal=sum(a == b for a, b in pairs), total=max(len(left), len(right)),
                first_mismatch=first)


def benchmark_pair(record: Path) -> list:
    """本番フラグ取得関数と記録当時のモデルから二つの評価器を作る。"""
    from scripts.replay_exchange_event_20260926 import static_builder
    from scripts.visualize_advantage_overlay import _ExchangeEventEndSignals
    from src.exchange_event_evaluator import FileExchangeModels
    from src.exchange_event_m0 import FileM0Predictor
    from src import production_config
    import shlex
    header = next(read_records(record))
    tokens = shlex.split(production_config.exchange_event_flags())
    model_index = tokens.index('--exchange-event-model-dir')
    directory = Path(tokens[model_index+1])
    enabled = [f for f in tokens if f.startswith('--') and f not in
               ('--exchange-event-update', '--exchange-event-model-dir')]
    options = {f.removeprefix('--').removeprefix('exchange-event-').replace('-', '_'): True
               for f in enabled}
    pairs = [(Path(header['model_dir']), {}), (directory, options)]
    return [SplitExchangeOverlay(FileExchangeModels.load(d, lightweight=True),
        static_builder(record), _ExchangeEventEndSignals, FileM0Predictor(d/'M0'),
        per_side_settled=header['per_side_settled'], **kw) for d, kw in pairs]


def benchmark(record: Path) -> dict:
    """B16全通知を送り、記録displayと同じ時点だけ数値計算する。"""
    overlays = benchmark_pair(record)
    timings: list[list[float]] = [[], []]
    updates: list[list[float]] = [[], []]
    calculations: list[list[float]] = [[], []]
    pending = [0., 0.]
    for row in read_records(record):
        if row['kind'] not in ('update', 'display'):
            continue
        order = (0, 1) if len(updates[0]) % 2 == 0 else (1, 0)
        for idx in order:
            started = time.perf_counter()
            if row['kind'] == 'update':
                overlays[idx].update(*row['args'])
                duration = time.perf_counter()-started
                updates[idx].append(duration)
                pending[idx] += duration
            else:
                overlays[idx].calculate()
                duration = time.perf_counter()-started
                calculations[idx].append(duration)
                timings[idx].append(pending[idx]+duration)
                pending[idx] = 0.
        if row['kind'] == 'update' and len(updates[0]) % PROGRESS_FRAMES == 0:
            print(json.dumps(dict(benchmark_frames=len(updates[0]))), flush=True)
    return dict(**{name: dict(quantiles(timings[idx]), update=quantiles(updates[idx]),
                             calculate=quantiles(calculations[idx]))
                   for idx, name in enumerate(('E14', 'production'))},
                nice=os.nice(0), scope='保存displayごと、直前までのupdate合計+calculate、R1並走下の相対比較')


def main() -> None:
    """動画・実時間入力を開かず、原票と結果を保存する。"""
    from scripts.run_live_pipeline_20260928 import compare_arrays
    from src.phase_j.live_cpu import configure_environment, apply_runtime
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, default=DEFAULT_RECORD)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--benchmark-record', type=Path,
                        default=Path('logs/live_b16/realtime/inputs.jsonl.gz'))
    parser.add_argument('--comparison-only', action='store_true')
    parser.add_argument('--benchmark-only', action='store_true')
    options = parser.parse_args()
    configure_environment(1, 10)
    apply_runtime('evaluation')
    options.out.mkdir(parents=True, exist_ok=True)
    if options.benchmark_only:
        report = json.loads((options.out/'comparison.json').read_text(encoding='utf-8'))
        report['benchmark'] = benchmark(options.benchmark_record)
        (options.out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return
    audit = audit_input(options.benchmark_record)
    report = dict(input=audit_input(options.record), b16_input=audit)
    offline = replay_cli(options.record, options.out/'offline', False)
    live = replay_cli(options.record, options.out/'realtime', True)
    report['probabilities'] = compare_rows(offline['probabilities'], live['probabilities'])
    report['display'] = compare_arrays(options.out/'offline/display.npz', options.out/'realtime/display.npz')
    (options.out/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not options.comparison_only:
        report['benchmark'] = benchmark(options.benchmark_record)
    (options.out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
