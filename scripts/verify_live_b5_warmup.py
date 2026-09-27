"""実対戦画面で初回と保存profile再確認の収束までを測る。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import cv2

from scripts.verify_live_b4_degradation import CONFIG, DEFAULT_VIDEO, START, END, FPS
from scripts.run_live_pipeline_20260928 import save_json
from src.phase_j.live_calibration import ColorWarmup
from src.phase_j.live_source import VideoFileSource

OUTPUT = Path('logs/live_b5_warmup')


def measure(device: SimpleNamespace, config: dict) -> dict:
    from src.recognition_pipeline import RecognitionPipeline
    pipe = RecognitionPipeline.load_default(**config)
    warmup = ColorWarmup(device, verification_only=True, calibrator=pipe._online_hsv)
    pipe._online_hsv = None
    capture = cv2.VideoCapture(str(DEFAULT_VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS)
    first = round(START*fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    source = VideoFileSource(capture, fps, first, round(END*fps), round(fps/FPS))
    frames = 0
    try:
        for frame in source:
            result = pipe.update(frame.index, frame.media_sec, frame.image)
            status = warmup.observe(frame.image, result, pipe._reader._classifier)
            frames += 1
            if warmup.ready:
                break
    finally:
        capture.release()
    return dict(frames=frames, media_seconds=frames/FPS, status=status,
                counts=warmup.calibrator.get_sample_counts(), target=warmup.target)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = SimpleNamespace(name='zenchi-replay-validation', index=0,
                             calibration_path=OUTPUT / 'device.json')
    device.calibration_path.unlink(missing_ok=True)
    config = json.loads(CONFIG.read_text())
    report = {}
    for name in ('initial', 'cached'):
        report[name] = measure(device, config)
        save_json(OUTPUT / 'report.json', report)


if __name__ == '__main__':
    main()
