"""B18b: 保存入力・動画通知・本番再生を照合する。実時間遅延は測定しない。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch
import numpy as np

from scripts.verify_live_b18a import compare_rows, quantiles
from src.phase_j.live_notification_eval import NotificationExchangeOverlay, latest_display

OUT = Path('logs/live_b18b')
EVAL_ROOT = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer')
PROGRESS_FRAMES = 1000
CPU_THREADS, NICE_LEVEL = 1, 10


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def replay(record: Path, dest: Path, live: bool, legacy_event_schema: bool = False) -> list:
    from scripts import replay_exchange_event_20260926 as cli
    probabilities = []
    base = NotificationExchangeOverlay if live else cli.ExchangeEventOverlay
    class Measured(base):
        def update(self, *args: Any, **kwargs: Any) -> None:
            if legacy_event_schema:
                for side in (args[0].p1, args[0].p2):
                    if side.chain_event is not None and hasattr(side.chain_event, 'before_board'):
                        del side.chain_event.before_board
            super().update(*args, **kwargs)
            from src.phase_j.live_notification_eval import _ExchangeDisplayEMA, _exchange_display
            if not hasattr(self, 'verification_smoothing'):
                self.verification_smoothing = _ExchangeDisplayEMA()
            display = (latest_display(self, 0., .5) if live else
                _exchange_display(self, 0., .5, self.verification_smoothing, args[3]))
            probabilities.append(dict(t_sec=args[3], game=args[4],
                p1=self.tracker.probability, source=self.tracker.source, display=list(display)))
    with ExitStack() as stack:
        stack.enter_context(patch.object(cli, 'ExchangeEventOverlay', Measured))
        if live:
            stack.enter_context(patch('scripts.visualize_advantage_overlay._exchange_display', latest_display))
        stack.enter_context(patch.object(sys, 'argv', ['replay', str(record), '--out', str(dest),
                                                      '--production-exchange-event']))
        cli.main()
    save(dest/'probabilities.json', probabilities)
    return probabilities


def compare(record: Path, dest: Path) -> None:
    from scripts.run_live_pipeline_20260928 import compare_arrays
    offline = replay(record, dest/'offline', False)
    live = replay(record, dest/'realtime', True)
    save(dest/'comparison.json', dict(probabilities=compare_rows(offline, live),
         display=compare_arrays(dest/'offline/display.npz', dest/'realtime/display.npz')))


def command(source: str, dest: Path) -> list[str]:
    from scripts.measure_realtime_breakdown_20260928 import PATH_FLAGS
    status = EVAL_ROOT/('logs/review_zenchi_g41_43_e14/status.json' if source == 'review'
                        else 'logs/review_zenchi_part3/on_e10c/status.json')
    args = json.loads(status.read_text())['command'][3:]
    args = [v for v in args if v not in ('--worker', '--review-data-panel')]
    for flag, filename in PATH_FLAGS.items():
        if flag in args:
            args[args.index(flag)+1] = str(dest/filename)
    for flag in ('--no-render', '--production-exchange-event'):
        if flag not in args:
            args.append(flag)
    return args


def capture(source: str, output: Path | None = None) -> None:
    """動画供給・認識・差分通知・評価は実装と共通、待機とSSE送出だけ省略する。"""
    from scripts import visualize_advantage_overlay as render
    from scripts.run_e3_exchange_eval_20260926 import worker
    from src.phase_j.live_bridge import RecognitionBridge, build_live_generate, recognize
    from src.phase_j.live_process import NoticeDeltaCodec
    from src.phase_j.live_source import VideoFileSource
    from time import perf_counter
    from src.phase_j.live_cpu import apply_runtime
    apply_runtime('recognition')
    dest, timings = output or OUT/'video'/source, []
    dest.mkdir(parents=True, exist_ok=True)
    class Bridge(RecognitionBridge):
        def observations(self, pipe: Any, cap: Any, fps: float,
                         start: int, end: int, stride: int) -> Any:
            encoder, decoder = NoticeDeltaCodec(), NoticeDeltaCodec()
            for frame in VideoFileSource(cap, fps, start, end, stride):
                started = perf_counter()
                notice = recognize(pipe, frame)
                cost = dict(pipe._live_snapshot_inputs.cost, recognition_sec=perf_counter()-started)
                timings.append(cost)
                yield from self.timed_notice(decoder.decode(encoder.encode(notice)))
                if len(timings) % PROGRESS_FRAMES == 0:
                    print(json.dumps(dict(source=source, frames=len(timings), t_sec=frame.media_sec)), flush=True)
    bridge = Bridge(False, lambda *args: None)
    bridge.split_evaluation = True
    args = command(source, dest)
    save(dest/'command.json', args)
    created, probabilities = [], []
    with ExitStack() as stack:
        stack.enter_context(patch.object(render, 'ExchangeEventOverlay', traced_overlay(created, probabilities)))
        stack.enter_context(patch.object(render, '_exchange_display', latest_display))
        stack.enter_context(patch.object(render, 'generate', build_live_generate(render, bridge)))
        stack.enter_context(patch.object(sys, 'argv', ['capture', '--worker', *args]))
        worker()
    save(dest/'probabilities.json', probabilities)
    save(dest/'prefire_audit.json', created[0]._prefire.summary())
    np.savez_compressed(dest/'timing.npz', **{key: [r[key] for r in timings] for key in timings[0]})
    report = {key: quantiles([r[key] for r in timings]) for key in timings[0]}
    report['retention_sec'] = quantiles([r['snapshot_sec']-r['hsv_sec'] for r in timings])
    save(dest/'cost.json', report)
    compare(dest/'inputs.jsonl.gz', dest/'replay')
    compare_video(dest)


def traced_overlay(created: list, probabilities: list) -> type:
    class Traced(NotificationExchangeOverlay):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

        def update(self, *args: Any, **kwargs: Any) -> None:
            super().update(*args, **kwargs)
            probabilities.append(dict(t_sec=args[3], game=args[4],
                p1=self.tracker.probability, source=self.tracker.source,
                display=list(latest_display(self, 0., .5))))
    return Traced


def compare_video(dest: Path) -> None:
    from scripts.run_live_pipeline_20260928 import compare_arrays
    reference = json.loads((dest/'replay/offline/probabilities.json').read_text())
    actual = json.loads((dest/'probabilities.json').read_text())
    if actual and 'display' not in actual[0]:
        reference = [{k: v for k, v in row.items() if k != 'display'} for row in reference]
    save(dest/'comparison.json', dict(probabilities=compare_rows(reference, actual),
        display=compare_arrays(dest/'replay/offline/display.npz', dest/'display.npz')))


def main() -> None:
    from src.phase_j.live_cpu import configure_environment, apply_runtime
    configure_environment(CPU_THREADS, NICE_LEVEL)
    apply_runtime('recognition')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('review', 'zenchi'), default='review')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--record', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--legacy-event-schema', action='store_true',
                        help='原因切り分け専用：旧zenchi原票で未記録のbefore_boardを除外')
    args = parser.parse_args()
    if args.legacy_event_schema:
        if args.record is None or args.out is None or args.video:
            parser.error('原因切り分けには--recordと--outが必要、--videoは併用不可')
        replay(args.record, args.out, False, legacy_event_schema=True)
    elif args.video:
        capture(args.source, args.out)
    else:
        compare(args.record or Path('logs/e31/records')/f'{args.source}.jsonl.gz',
                args.out or OUT/'saved'/args.source)


if __name__ == '__main__':
    main()
