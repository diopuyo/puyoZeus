"""既知の不一致周辺を追加学習する。回帰試験を独立評価と混同しない。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.compare_image_free_digits import (cv2, joblib, np, samples, fit,
                                               evaluate, TRAIN_FRACTION, SAMPLES)
from scripts.obs_priority_guard import ObsPriorityGuard
from src.image_free_digit import ImageFreeScoreOcr, FEATURE_SIZE, MODEL_SCHEMA
from src.score_ocr import ScoreOcr

NEIGHBOR_SECONDS = (-0.2, -0.1, 0.0, 0.1, 0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--templates', type=Path, required=True)
    parser.add_argument('--differences', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    guard = ObsPriorityGuard()
    guard.checkpoint(force=True)
    cv2.setNumThreads(1)
    teacher = ScoreOcr.load_default(args.templates)
    cap = cv2.VideoCapture(str(args.video))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    known = json.loads(args.differences.read_text(encoding='utf-8'))
    original = np.linspace(1, duration - 1, SAMPLES)[:int(SAMPLES * TRAIN_FRACTION)]
    extra = [row['t_sec'] + dt for row in known for dt in NEIGHBOR_SECONDS]
    times = np.unique(np.concatenate((original, extra)))
    data = samples(cap, teacher, times, guard)
    model = dict(schema=MODEL_SCHEMA, feature_size=FEATURE_SIZE,
                 labels=fit(data[0], data[1], False, guard),
                 confidence=fit(data[0], data[2], True, guard),
                 multiply=fit(data[0], data[3], True, guard))
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / 'digit_student.joblib'
    joblib.dump(model, model_path, compress=3)
    student = ImageFreeScoreOcr(model_path)
    report = evaluate(cap, np.unique([r['t_sec'] for r in known]), teacher,
                      student, args.output / 'differences.json', guard)
    report.update(training_frames=len(times), evaluation_kind='既知不一致の回帰試験・学習内',
                  independent_gate_passed=False, downstream='未測定')
    (args.output / 'comparison.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    cap.release()
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
