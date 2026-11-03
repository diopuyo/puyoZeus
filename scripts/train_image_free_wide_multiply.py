"""R1の広域×スコアを追加蒸留する。画像は保存せず未採用の重みを作る。"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.compare_image_free_digits import cv2, joblib, np, fit, frame_at
from scripts.obs_priority_guard import ObsPriorityGuard
from src.image_free_digit import wide_multiply_features
from src.placement_signal_images import multiply
from src.score_ocr import ScoreOcr

TRAINING_FRAMES = 1600
TRAINING_FRACTION = 0.6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('video', 'templates', 'model', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    guard = ObsPriorityGuard()
    guard.checkpoint(force=True)
    cv2.setNumThreads(1)
    teacher = ScoreOcr.load_default(args.templates)
    cap = cv2.VideoCapture(str(args.video))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    features, targets = [], []
    for i, stamp in enumerate(np.linspace(1, duration * TRAINING_FRACTION, TRAINING_FRAMES)):
        guard.checkpoint()
        frame = frame_at(cap, float(stamp))
        features.extend(wide_multiply_features(frame))
        targets.extend(multiply(frame, teacher._mult_template_gray))
        if i % 100 == 0:
            print(f'広域×教師 {i}/{TRAINING_FRAMES}', flush=True)
    cap.release()
    model = joblib.load(args.model)
    model['wide_multiply'] = fit(np.asarray(features), np.asarray(targets), True, guard)
    model['wide_training'] = dict(frames=TRAINING_FRAMES, end_sec=duration * TRAINING_FRACTION)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output, compress=3)
    print(f'未採用候補を保存: {args.output}', flush=True)


if __name__ == '__main__':
    main()
