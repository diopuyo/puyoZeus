"""R1広域×の採否を後半の標本で比較する。全体合格には使わない。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.compare_image_free_digits import cv2, np, frame_at
from scripts.obs_priority_guard import ObsPriorityGuard
from src.image_free_digit import ImageFreeScoreOcr
from src.score_ocr import ScoreOcr, FORMULA_MULT_NCC_MIN

SAMPLES = 960
START_FRACTION = 0.6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('video', 'templates', 'model', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    guard = ObsPriorityGuard()
    guard.checkpoint(force=True)
    cv2.setNumThreads(1)
    old = ScoreOcr.load_default(args.templates).placement_multiply_reader()
    new = ImageFreeScoreOcr(args.model).placement_multiply_reader()
    cap = cv2.VideoCapture(str(args.video))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    differences = []
    old_positive = new_positive = equal = 0
    for i, stamp in enumerate(np.linspace(duration * START_FRACTION, duration - 1, SAMPLES)):
        guard.checkpoint()
        frame = frame_at(cap, float(stamp))
        for side, (a, b) in enumerate(zip(old(frame), new(frame))):
            truth, predicted = a >= FORMULA_MULT_NCC_MIN, b >= FORMULA_MULT_NCC_MIN
            old_positive += int(truth)
            new_positive += int(predicted)
            equal += int(truth == predicted)
            if truth != predicted:
                differences.append(dict(t_sec=float(stamp), side=side + 1, old=a, new=b))
        if i % 100 == 0:
            print(f'広域×評価 {i}/{SAMPLES} 差={len(differences)}', flush=True)
    cap.release()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(equal=equal, total=SAMPLES * 2, old_positive=old_positive,
                  new_positive=new_positive, rate=equal / (SAMPLES * 2), gate_passed=False,
                  scope='後半40%の960標本。連続フレームと下流は未検証')
    for name, value in (('comparison', report), ('differences', differences)):
        (args.output / f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
