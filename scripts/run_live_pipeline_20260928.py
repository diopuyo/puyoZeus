"""B2: 動画入力を認識30Hz・評価バッチ/SSE最大2Hzへ分離する新経路。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from threading import Event, Thread
import time
from typing import Any
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np

from scripts.measure_realtime_breakdown_20260928 import DEFAULT_STATUS, DEFAULT_VIDEO, build_command
from src.phase_j.live_bridge import RecognitionBridge, RecognitionNotice, build_live_generate, notice_digest
from src.phase_j.live_publish import LivePublisher

ROOT = Path(__file__).resolve().parents[1]
START_SEC, END_SEC, WARMUP_SEC = 2600.0, 2640.0, 1.0
DEFAULT_PORT = 8765
MILLISECONDS = 1000.0
PERCENTILES = (50, 95, 99)
PROBE_TIMEOUT_SEC = 60
PROBE_READY_SEC = 10
PROBE_JOIN_SEC = 2


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def asset_hashes() -> dict[str, str]:
    """実行設定とモデル資産をDTOの再現用IDへ固定する。"""
    groups = dict(app_build_id=['scripts/run_live_pipeline_20260928.py',
        'scripts/visualize_advantage_overlay.py', 'src/phase_j/live_bridge.py'],
        recognition_model_hash=['models/cnn_phase_b_large_v2.pt', 'models/cnn_global_best.pt',
                                'models/cnn_best.pt'],
        recognition_config_hash=['src/production_config.py'],
        prediction_model_hash=['models/exchange_event_v2/manifest.json',
                               'models/exchange_event_v1/manifest.json',
                               'models/exchange_event_v1/M0/manifest.json'],
        calibration_hash=['models/calibration_video01.json'])
    result = {}
    for name, files in groups.items():
        digest = hashlib.sha256()
        for filename in files:
            digest.update(filename.encode())
            digest.update((ROOT / filename).read_bytes())
        result[name] = 'sha256:' + digest.hexdigest()
    return result


class SSEProbe:
    """localhostの実購読を維持し、サーバが実際にSSE送出した遅延を測れるようにする。"""

    def __init__(self, address: tuple[str, int]) -> None:
        self.url = 'http://%s:%s/events' % address
        self.ready = Event()
        self.error: BaseException | None = None
        self.count = 0
        self.thread = Thread(target=self._read, name='sse-smoke-subscriber', daemon=True)

    def _read(self) -> None:
        try:
            with urlopen(self.url, timeout=PROBE_TIMEOUT_SEC) as response:
                self.ready.set()
                for line in response:
                    if line.startswith(b'data: '):
                        self.count += 1
        except BaseException as error:
            self.error = error
            self.ready.set()

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(PROBE_READY_SEC) or self.error is not None:
            raise RuntimeError('SSE購読を開始できません') from self.error


class ResultSink:
    """評価履歴は全フレーム保持し、配信側には最新候補だけ渡す。"""

    def __init__(self, publisher: LivePublisher) -> None:
        self.publisher = publisher
        self.rows: list[dict[str, Any]] = []

    def observe(self, notice: RecognitionNotice, probability: float, advantage: float,
                overlay: Any, result: Any, game: int, queue_depth: int, evaluated_at: float) -> None:
        source = overlay.tracker.source
        stable = any(side.state.name == 'STABLE' for side in (result.p1, result.p2))
        row = dict(frame=notice.frame, t_sec=notice.t_sec, game=game,
            captured_at=notice.captured_at, recognized_at=notice.recognized_at,
            evaluated_at=evaluated_at, probability=float(probability), advantage=float(advantage),
            raw_probability=overlay.tracker.probability, source=source,
            state1=result.p1.state.name, state2=result.p2.state.name,
            score1=result.p1.score, score2=result.p2.score, queue_depth=queue_depth,
            digest=notice_digest(notice), capture_gap=notice.dropped_before > 0,
            hold=not stable and source == 'G_fe')
        self.rows.append(row)
        self.publisher.offer(row)


def percentile(values: list[float]) -> dict[str, Any]:
    """欠測はnullとし、母数を必ず保存する。"""
    return dict(count=len(values), **{f'P{p}': float(np.percentile(values, p)) if values else None
                                     for p in PERCENTILES})


def metrics(bridge: RecognitionBridge, sink: ResultSink, options: argparse.Namespace,
            command: list[str]) -> dict[str, Any]:
    """認識全件・評価全件・実送信の各母数を区別する。"""
    recognition = [r for r in bridge.recognition_rows if r['t_sec'] >= options.start_sec]
    fps, start, end, stride = bridge.frame_bounds
    first = max(start, start + math.ceil((options.start_sec*fps-start)/stride)*stride)
    expected = len(range(first, end, stride))
    sent = list(sink.publisher.server.sent.values())
    stages = {'capture_to_recognition': [(r['recognized_at']-r['captured_at'])*MILLISECONDS for r in recognition],
        'recognition_to_evaluation': [(r['evaluated_at']-r['recognized_at'])*MILLISECONDS for r in sink.rows],
        'capture_to_evaluation': [(r['evaluated_at']-r['captured_at'])*MILLISECONDS for r in sink.rows],
        'evaluation_to_sse': [(r['sent_at']-r['evaluated_monotonic_sec'])*MILLISECONDS for r in sent],
        'capture_to_sse': [(r['sent_at']-r['capture_monotonic_sec'])*MILLISECONDS for r in sent]}
    return dict(realtime=options.realtime, frames=len(sink.rows), dropped_frames=bridge.source.dropped,
        dropped_frames_in_measured_window=expected-len(recognition), expected_frames=expected,
        recognition_frames_including_warmup=len(bridge.recognition_rows),
        latency_ms={name: percentile(values) for name, values in stages.items()},
        evaluation_queue=dict(capacity=bridge.queue.maxsize, maximum=bridge.queue_max,
                             maximum_including_batch=bridge.pending_max,
                             end=bridge.queue.qsize(), observed=[r['queue_depth'] for r in sink.rows]),
        batch_starts=bridge.batch_starts, batch_sizes=bridge.batch_sizes,
        publications=sink.publisher.publications, sse_sent=sent, command=command,
        recognition=recognition, nice=os.nice(0),
        timing_basis='realtimeは予定capture壁時計、非realtimeは読出開始。SSEはsocket flush完了。')


def run_live(options: argparse.Namespace) -> dict[str, Any]:
    """認識のみ別スレッドへ移し、評価状態はこの単一workerが所有する。"""
    import torch
    import scripts.visualize_advantage_overlay as overlay
    from scripts.run_e3_exchange_eval_20260926 import SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    options.no_render = True
    command = build_command(options)
    publisher = LivePublisher(asset_hashes(), options.host, options.port)
    sink = ResultSink(publisher)
    bridge = RecognitionBridge(options.realtime, sink.observe)
    publisher.start()
    probe = SSEProbe(publisher.server.address())
    try:
        probe.start()
        with ExitStack() as stack:
            stack.enter_context(patch.object(overlay, 'generate', build_live_generate(overlay, bridge)))
            stack.enter_context(patch.object(sys, 'argv', ['live-pipeline', *command]))
            overlay.main()
    finally:
        try:
            bridge.close()
        finally:
            try:
                publisher.close()
            finally:
                probe.thread.join(PROBE_JOIN_SEC)
    if probe.error is not None or not publisher.server.sent:
        raise RuntimeError('SSE送信の観測がありません') from probe.error
    report = metrics(bridge, sink, options, command)
    report.update(gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                  sse_probe_messages=probe.count, assets=dict(publisher.initial.assets))
    save_json(options.output / 'metrics.json', report)
    save_json(options.output / 'evaluations.json', sink.rows)
    save_json(options.output / 'latest.json', publisher.hub.latest.to_mapping())
    return {key: report[key] for key in ('frames', 'dropped_frames', 'latency_ms')}


def compare_arrays(reference: Path, candidate: Path) -> dict[str, Any]:
    """時刻を含む全列を厳密比較し、不一致件数と最初の場所を保存する。"""
    with np.load(reference) as left, np.load(candidate) as right:
        count_left, count_right = len(left['t_sec']), len(right['t_sec'])
        count = min(count_left, count_right)
        mismatch = np.zeros(count, dtype=bool)
        columns = sorted(set(left.files) | set(right.files))
        first = None
        for name in columns:
            if name == 'video_id':
                continue
            if name not in left or name not in right:
                raise ValueError(f'比較列が欠落: {name}')
            a, b = left[name][:count], right[name][:count]
            equal = a == b
            if a.dtype.kind == 'f':
                equal |= np.isnan(a) & np.isnan(b)
            mismatch |= ~equal
            if not equal.all():
                index = int(np.flatnonzero(~equal)[0])
                if first is None or index < first['index']:
                    first = dict(index=index, column=name, t_sec=float(left['t_sec'][index]),
                                 reference=str(a[index]), candidate=str(b[index]))
        missing = abs(count_left-count_right)
        if first is None and missing:
            first = dict(index=count, column='length', reference=count_left, candidate=count_right)
        return dict(reference_frames=count_left, candidate_frames=count_right,
                    compared_frames=count, mismatched_frames=int(mismatch.sum())+missing,
                    first_mismatch=first, columns=columns)


def run_child(command: list[str], output: Path) -> None:
    """比較の両側を独立プロセス・同一seedで開始する。"""
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'command.json', command)
    with (output / 'run.log').open('w', encoding='utf-8') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def run_comparison(options: argparse.Namespace) -> dict[str, Any]:
    """基準は無改変のrun_e3 worker。新経路と全30Hz観測で照合する。"""
    base = options.output
    options.output, options.no_render = base / 'reference', True
    reference = [sys.executable, '-m', 'scripts.run_e3_exchange_eval_20260926',
                 '--worker', *build_command(options)]
    run_child(reference, options.output)
    candidate = [sys.executable, '-m', 'scripts.run_live_pipeline_20260928',
        '--video', str(options.video), '--status', str(options.status),
        '--start-sec', str(options.start_sec), '--end-sec', str(options.end_sec),
        '--warmup-sec', str(options.warmup_sec), '--port', str(options.port),
        '--output', str(base / 'candidate')]
    run_child(candidate, base / 'candidate')
    report = compare_arrays(base/'reference/display.npz', base/'candidate/display.npz')
    save_json(base / 'comparison.json', report)
    if report['mismatched_frames'] or report['compared_frames'] == 0:
        raise ValueError(f'同値比較不一致: {report}')
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=DEFAULT_VIDEO)
    parser.add_argument('--status', type=Path, default=DEFAULT_STATUS)
    parser.add_argument('--output', type=Path, default=Path('logs/live_pipeline_20260928'))
    parser.add_argument('--start-sec', type=float, default=START_SEC)
    parser.add_argument('--end-sec', type=float, default=END_SEC)
    parser.add_argument('--warmup-sec', type=float, default=WARMUP_SEC)
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    options = parser.parse_args()
    if options.compare and options.realtime:
        parser.error('全フレーム同値比較とrealtimeは別実行です')
    if not 0 <= options.start_sec < options.end_sec or options.warmup_sec < 0:
        parser.error('計測区間が不正です')
    return options


def main() -> None:
    options = parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    result = run_comparison(options) if options.compare else run_live(options)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
