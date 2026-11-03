"""B15の未完gzipから完全な記録行だけを救出し、動画なしで障害を再生する。"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import time
import traceback
from typing import Iterator
import zlib

from src.exchange_event_record import decode, encode, read_records

BLOCK = 65536
ROOT = Path('logs/live_b16')


def recover(path: Path) -> Iterator[dict]:
    inflater, pending = zlib.decompressobj(16+zlib.MAX_WBITS), b''
    with path.open('rb') as stream:
        while block := stream.read(BLOCK):
            pending += inflater.decompress(block)
            parts = pending.split(b'\n')
            pending = parts.pop()
            for line in parts:
                yield decode(json.loads(line))


def extract(source: Path, destination: Path) -> dict:
    rows = list(recover(source))
    updates = [row for row in rows if row['kind'] == 'update']
    game = updates[-1]['args'][4]
    selected = [r for r in rows if r['kind'] not in ('complete', 'update', 'display') or
                (r['kind'] == 'update' and r['args'][4] >= game-1) or
                (r['kind'] == 'display' and r['game_idx'] >= game-1)]
    frames = sum(r['kind'] == 'update' for r in selected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, 'wt') as stream:
        for row in [*selected, dict(kind='complete', frames=frames)]:
            stream.write(json.dumps(encode(row))+'\n')
    result = dict(source=str(source), frames=frames, all_frames=len(updates), game=game,
                  first_sec=next(r['args'][3] for r in selected if r['kind'] == 'update'),
                  last_sec=updates[-1]['args'][3], saved=str(destination))
    (destination.parent/'extraction.json').write_text(json.dumps(result, indent=2))
    return result


def make_overlay(path: Path, split: bool, output: Path, supervised: bool) -> object:
    from scripts.replay_exchange_event_20260926 import static_builder
    from scripts.visualize_advantage_overlay import _ExchangeEventEndSignals
    from src.exchange_event_overlay import ExchangeEventOverlay
    from src.exchange_event_evaluator import FileExchangeModels
    from src.exchange_event_m0 import FileM0Predictor
    from src.phase_j.live_evaluation import SplitExchangeOverlay
    from src.phase_j.live_cpu import configure_environment, apply_runtime
    configure_environment(1, 0)
    apply_runtime('evaluation', lower_priority=False)
    rows = list(read_records(path))
    model = Path(rows[0]['model_dir'])
    overlay_type = SplitExchangeOverlay if split else ExchangeEventOverlay
    options = {}
    if supervised:
        from src.phase_j.live_eval_supervisor import SupervisedOverlay
        overlay_type = SupervisedOverlay
        options['directory'] = output.parent/'replay-worker'
    overlay = overlay_type(
        FileExchangeModels.load(model, lightweight=True), static_builder(path),
        _ExchangeEventEndSignals, FileM0Predictor(model/'M0'),
        per_side_settled=rows[0]['per_side_settled'], **options)
    apply_runtime('evaluation', lower_priority=False)
    return overlay


def failure_row(overlay: object, error: Exception, t_sec: float) -> dict:
    record = getattr(overlay.tracker, 'current', None)
    return dict(t_sec=t_sec, kind=type(error).__name__, message=str(error),
        stack=traceback.format_exc(), pending=repr(getattr(overlay.tracker, 'pending', None)),
        values=record.values if record else None,
        diagnostics=getattr(overlay.tracker, 'diagnostics', [])[-10:],
        current_id=getattr(record, 'exchange_id', None))


def replay(path: Path, split: bool, output: Path, supervised: bool = False) -> dict:
    overlay = make_overlay(path, split, output, supervised)
    rows = list(read_records(path))
    attempts = {r['t_sec'] for r in rows if r['kind'] == 'display'}
    updates = [r['args'] for r in rows if r['kind'] == 'update']
    attempts.add(updates[-1][3])
    start, values, failure = time.perf_counter(), [], None
    for inputs in updates:
        try:
            overlay.update(*inputs)
            if split and inputs[3] in attempts:
                overlay.calculate()
            values.append(dict(t_sec=inputs[3], source=overlay.tracker.source, p1=overlay.tracker.probability))
        except Exception as error:
            failure = failure_row(overlay, error, inputs[3])
            break
    result = dict(split=split, frames=len(values), attempts=len(attempts), failure=failure,
                  seconds=time.perf_counter()-start, values=values)
    output.write_text(json.dumps(result, indent=2))
    if supervised:
        overlay.close()
    return {k: v for k, v in result.items() if k != 'values'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--supervised', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT/'replay.json')
    args = parser.parse_args()
    record = ROOT/'recovered.jsonl.gz'
    if args.extract:
        print(json.dumps(extract(Path('logs/live_b15/realtime/inputs.jsonl.gz'), record)))
    else:
        print(json.dumps(replay(record, not args.offline, args.out, args.supervised), ensure_ascii=False))
