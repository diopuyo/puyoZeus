"""現行本番を無変更で再生し、成功した静止評価入力だけ観測する。"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from typing import Any
from scripts._audit_static_1003_extract import OUT, BASE, SOURCES, guard, save_json
import numpy as np


def capture(source: str) -> None:
    """追跡器の戻り値も状態も変更せず静止評価結果を複写する。"""
    from src.exchange_event_tracker import ExchangeEventTracker
    from src.exchange_event_evaluator import build_features
    from scripts import replay_exchange_event_20260926 as module
    original = ExchangeEventTracker._evaluate
    traces: list[tuple] = []

    def observe(tracker: Any, event: Any, route: str, t_sec: float) -> bool:
        result = original(tracker, event, route, t_sec)
        if route == 'G_fe' and result:
            x = build_features(event, tracker.models.elapsed_thresholds)[1]
            traces.append((t_sec, tracker.probability, x))
        if len(traces) % 1000 == 0:
            guard()
        return result

    ExchangeEventTracker._evaluate = observe
    dest = OUT / 'replay' / source
    sys.argv = [module.__file__, str(BASE / f'collect/records/{source}.jsonl.gz'),
                '--out', str(dest), '--production-exchange-event']
    module.main()
    np.savez_compressed(dest / 'static_inputs.npz', t=[r[0] for r in traces],
                        p=[r[1] for r in traces], x=[r[2] for r in traces])
    save_json(f'{source}_replay_done.json', dict(calls=len(traces)))


def main() -> None:
    """指定された一分割または全分割を一プロセスずつ実行する。"""
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=SOURCES)
    args = parser.parse_args()
    if args.source:
        capture(args.source)
        return
    for source in SOURCES:
        if (OUT / f'{source}_replay_done.json').exists():
            continue
        print('START', source, flush=True)
        subprocess.run([sys.executable, '-B', '-m', 'scripts._audit_static_1003_replay',
                        '--source', source], check=True)
        print('DONE', source, flush=True)


if __name__ == '__main__':
    main()
