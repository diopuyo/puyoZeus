"""B20: 認識1フレーム費用を fast_terminal OFF/ON の A/B/A/B で、同一フレーム列に対して比較する。

本番の recognition_worker 相当 (1スレッド設定・同一 config・非realtime stride2) で単独process実走。
各パスで死亡確定 (confirmed_dead_sides) を全フレーム記録し、OFF/ON の決定差も数える。
割り算の単位は「認識processが実際に処理した frame」。src/ は変更しない。
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

# 本番 live (cnn_device=cpu) と同条件にするため torch import 前に GPU を隠す (B20_GPU=1 で解除)。
if os.environ.get('B20_GPU') != '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

from src.phase_j.live_cpu import configure_environment
configure_environment(1, 10)  # numpy/torch import前に本番同様のスレッド環境

import cv2
import numpy as np

WINDOWS_VIDEO = 'C:/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/frames/video_zenchi_c0BQoMJwwQU.mp4'
START = float(sys.argv[2]) if len(sys.argv) > 2 else 5880.566
FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
# off=従来 / on=fast_terminal / none=enrich_recognition自体を除去した下限 (何を測っているかの物差し)
ORDER = ('off', 'on', 'none', 'off', 'on', 'none')
WARMUP = 60
CONFIG = Path('logs/live_b18/run5/candidate/recognition_config.json')
OUT = Path(os.environ.get('B20_OUT', 'logs/live_b20/bench_recognition.json'))


def _default_video() -> str:
    from scripts.measure_realtime_breakdown_20260928 import DEFAULT_VIDEO
    return str(DEFAULT_VIDEO)


def png_source(directory: str, fps: float):
    """Windows配布版は動画を開けないため、PNG連番 (stride2抽出済み) から CapturedFrame を作る。"""
    from src.phase_j.live_source import CapturedFrame
    for path in sorted(Path(directory).glob('frame_*.png')):
        index = int(path.stem.split('_')[1])
        yield CapturedFrame(index, index/fps, time.perf_counter(), time.perf_counter(), cv2.imread(str(path)))


def _loadavg() -> str:
    return open('/proc/loadavg').read().strip() if os.name != 'nt' else 'n/a(windows)'


def one_pass(mode: str) -> dict:
    os.environ['PUYO_FAST_TERMINAL'] = '1' if mode == 'on' else '0'
    import contextlib
    from unittest.mock import patch
    import src.phase_j.live_snapshot as snap
    from src.phase_j.live_cpu import apply_runtime
    from src.recognition_pipeline import RecognitionPipeline
    from src.phase_j.live_bridge import recognize
    from src.phase_j.live_source import VideoFileSource
    DEFAULT_VIDEO = WINDOWS_VIDEO if os.name == 'nt' else _default_video()
    apply_runtime('recognition')
    pipe = RecognitionPipeline.load_default(**json.loads(CONFIG.read_text()))
    if os.environ.get('B20_PNG_DIR'):
        source = png_source(os.environ['B20_PNG_DIR'], 60.0)
    else:
        cap = cv2.VideoCapture(str(DEFAULT_VIDEO))
        fps = cap.get(cv2.CAP_PROP_FPS)
        start = int(START*fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        source = VideoFileSource(cap, fps, start, start+FRAMES*2, 2, False)
    times, dead, terminal_ms = [], [], []
    stub = (patch.multiple(snap, enrich_recognition=lambda p, f, r, s: r,
                           prepare_recognition=lambda p: None)
            if mode == 'none' else contextlib.nullcontext())
    with stub:
        for frame in source:
            t = time.perf_counter()
            notice = recognize(pipe, frame)
            times.append((time.perf_counter()-t)*1000)
            dead.append((frame.index, getattr(pickle.loads(notice.result_bytes), 'confirmed_dead_sides', ())))
            inputs = getattr(getattr(pipe, 'pipe', pipe), '_live_snapshot_inputs', None)
            terminal_ms.append(0.0 if inputs is None else inputs.cost.get('total_sec', 0.0)*1000)
    a = np.array(times[WARMUP:])
    inputs = getattr(getattr(pipe, 'pipe', pipe), '_live_snapshot_inputs', None)
    det = inputs.terminal if inputs is not None else None
    return dict(mode=mode, n=len(a), p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), mean=float(a.mean()),
                enrich_total_ms_mean=float(np.mean(terminal_ms[WARMUP:])),
                full_matches=list(det.full_matches) if det else None,
                detector_frames=det.frames if det else None, dead=dead)


def main() -> None:
    print('loadavg_before', _loadavg(), flush=True)
    runs = []
    for mode in ORDER:
        runs.append(one_pass(mode))
        r = runs[-1]
        print(r['mode'], 'n', r['n'], 'p50 %.1f p95 %.1f p99 %.1f mean %.1f enrich %.1f full_matches %s/%s'
              % (r['p50'], r['p95'], r['p99'], r['mean'], r['enrich_total_ms_mean'],
                 r['full_matches'], r['detector_frames']),
              'loadavg', _loadavg().split()[:3], flush=True)
    by = {r['mode']+str(i): r for i, r in enumerate(runs)}
    diffs = sum(a != b for a, b in zip(runs[0]['dead'], runs[1]['dead']))
    print('decision differences off vs on: %d / %d frames' % (diffs, len(runs[0]['dead'])))
    OUT.write_text(json.dumps([{k: v for k, v in r.items() if k != 'dead'} for r in runs] +
                              [dict(decision_differences=diffs, frames=len(runs[0]['dead']),
                                    dead_frames=sum(bool(d) for _, d in runs[0]['dead']))], indent=1))


if __name__ == '__main__':
    main()
