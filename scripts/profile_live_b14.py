"""B14: 同一入力の認識をCPU一スレッドで計測・保存する。"""
from __future__ import annotations

import argparse
import cProfile
import json
import os
import importlib
from contextlib import ExitStack
from pathlib import Path
from threading import Event
from typing import Any

from src.phase_j.live_cpu import configure_environment

WARMUP_SECONDS, RECOGNITION_HZ = 1.0, 30


def video_bounds(fps: float, start: float, end: float) -> tuple[float, int, int, int]:
    """本番generateと同じ切捨てで、絶対フレームの偶奇を保つ。"""
    return fps, int((start-WARMUP_SECONDS)*fps), int(end*fps), round(fps/RECOGNITION_HZ)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=float, default=2580.566)
    parser.add_argument('--end', type=float, default=2700.566)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--stages', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--measure-start', type=float)
    parser.add_argument('--original-templates', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    configure_environment(1, 0)
    restore_baseline(args)
    run(args)


def restore_baseline(args: argparse.Namespace) -> None:
    names = ('background_fingerprint', 'ui_mask', 'image_reader', 'patch_classifier',
             'hybrid_classifier', 'match_end_detector', 'telop_detector')
    selected = names if args.baseline else (
        ('ui_mask', 'match_end_detector', 'telop_detector') if args.original_templates else ())
    for name in selected:
        module = importlib.import_module('src.'+name)
        source = Path('logs/live_b14/baseline_sources')/(name+'.py')
        exec(compile(source.read_text(), str(source), 'exec'), vars(module))


def save_audit_once(original: Any) -> Any:
    """計測終了後だけ監査を一度読み込み、列ごとのディスク再走査を避ける。"""
    def save(audit: Any, runtime: dict, source: Any) -> None:
        rows = audit.rows
        try:
            audit.rows = list(rows)
            original(audit, runtime, source)
        finally:
            audit.rows = rows
    return save


def run(args: argparse.Namespace) -> None:
    import cv2
    from unittest.mock import patch
    from src.phase_j.live_audit import RecognitionAudit
    from scripts.analyze_live_b14 import source_hashes
    from scripts.bench_live_b8_recognition import Sink
    from scripts.diagnose_live_b8 import case_command
    from scripts.measure_realtime_breakdown_20260928 import DEFAULT_VIDEO
    from src.phase_j.live_process import recognition_worker
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'manifest.json').write_text(json.dumps(dict(source_hashes=source_hashes(),
        baseline=args.baseline, original_templates=args.original_templates,
        start=args.start, end=args.end, measure_start=args.measure_start, loadavg=os.getloadavg()), indent=2))
    case_command(('probe', False, args.start, args.end, 1, 0), args.output)
    config = json.loads(Path('logs/live_b13/reference/recognition_config.json').read_text())
    capture = cv2.VideoCapture(str(DEFAULT_VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    # 本番generateと同じ切捨て。roundでは偶奇が変わり、ROI走査の位相も変わる。
    bounds = video_bounds(fps, args.start, args.end)
    (args.output/'bounds.json').write_text(json.dumps(dict(fps=fps, start=bounds[1],
                                                        end=bounds[2], stride=bounds[3]), indent=2))
    profiler, sink = cProfile.Profile(), Sink()
    if args.profile:
        profiler.enable()
    with ExitStack() as stack:
        stack.enter_context(patch.object(RecognitionAudit, 'save', save_audit_once(RecognitionAudit.save)))
        if args.stages:
            from scripts.live_b14_meter import StageMeter, install
            meter = StageMeter()
            install(stack, meter)
        recognition_worker(sink, Event(), config, str(DEFAULT_VIDEO), bounds, False, None,
            lifecycle=True, live_config=str(args.output/'probe_config.json'),
            audit_path=str(args.output/'recognition.npz'),
            retention_path=str(args.output/'spool'))
        if args.stages:
            meter.save(args.output, args.measure_start or args.start)
    if args.profile:
        profiler.disable()
        profiler.dump_stats(str(args.output/'profile.pstats'))
    (args.output/'summary.json').write_text(json.dumps(sink.summary, indent=2))


if __name__ == '__main__':
    main()
