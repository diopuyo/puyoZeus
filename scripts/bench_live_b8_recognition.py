"""評価・MCなしの同じ330フレームで、認識自身のスレッド数の影響を確認する。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from threading import Event
from typing import Any

from scripts.diagnose_live_b8 import case_command, OUTPUT
from src.phase_j.live_cpu import configure_environment

START, END, WARMUP, HZ = 2600., 2610., 1., 30


class Sink:
    def __init__(self) -> None:
        self.summary: dict = {}

    def put(self, message: tuple) -> None:
        kind, value = message
        if kind == 'error':
            raise RuntimeError(value)
        if kind == 'summary':
            self.summary = value


def run(threads: int, output: Path) -> dict[str, Any]:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    configure_environment(threads, 0)
    import cv2
    from src.phase_j.live_process import recognition_worker
    from scripts.measure_realtime_breakdown_20260928 import DEFAULT_VIDEO
    name = f'recognition_only_{threads}'
    case_command((name, False, START, END, threads, 0), output)
    target = output/name
    target.mkdir(parents=True, exist_ok=True)
    pipe_config = json.loads((output/'full/recognition_config.json').read_text())
    capture = cv2.VideoCapture(str(DEFAULT_VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    bounds = (fps, round((START-WARMUP)*fps), round(END*fps), round(fps/HZ))
    sink = Sink()
    recognition_worker(sink, Event(), pipe_config, str(DEFAULT_VIDEO), bounds, False, None,
        lifecycle=True, live_config=str(output/(name+'_config.json')),
        audit_path=str(target/'recognition.npz'))
    return sink.summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-threads', type=int, choices=(0, 1), required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    options = parser.parse_args()
    print(json.dumps(run(options.cpu_threads, options.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
