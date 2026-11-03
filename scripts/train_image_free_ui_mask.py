"""窒息位置の×印を蒸留し、時間分離した標本で検査する（全体合格ではない）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.compare_image_free_digits import cv2, joblib, np, fit, frame_at
from scripts.obs_priority_guard import ObsPriorityGuard
from src.image_free_digit import digit_features
from src.image_free_ui_mask import MODEL_SCHEMA
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.ui_mask import UiMaskMatcher

SAMPLES = 1600
TRAIN_FRACTION = 0.6
TARGET_ROW, TARGET_COL = 1, 2


def patches(frame: np.ndarray) -> list[np.ndarray]:
    result = []
    for region in (DEFAULT_P1_REGION, DEFAULT_P2_REGION):
        x1, y1, x2, y2 = region.cell_sample_rect(TARGET_ROW, TARGET_COL)
        result.append(frame[y1:y2, x1:x2])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('video', 'templates', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    guard = ObsPriorityGuard()
    guard.checkpoint(force=True)
    cv2.setNumThreads(1)
    teacher = UiMaskMatcher.load_default(args.templates)
    cap = cv2.VideoCapture(str(args.video))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    features, labels, scores = [], [], []
    for i, stamp in enumerate(np.linspace(1, duration - 1, SAMPLES)):
        guard.checkpoint()
        for patch in patches(frame_at(cap, float(stamp))):
            result = teacher.match(patch)
            features.append(digit_features(patch))
            labels.append(result.template_name if result.is_ui else '')
            scores.append(result.score)
        if i % 100 == 0:
            print(f'×印教師 {i}/{SAMPLES}', flush=True)
    cap.release()
    split = int(SAMPLES * TRAIN_FRACTION) * 2
    data = np.asarray(features)
    model = dict(schema=MODEL_SCHEMA, labels=fit(data[:split], np.asarray(labels[:split]), False, guard),
                 confidence=fit(data[:split], np.asarray(scores[:split]), True, guard))
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / 'ui_mask.joblib', compress=3)
    predicted = model['labels'].predict(data[split:])
    truth = np.asarray(labels[split:])
    same = (predicted != '') == (truth != '')
    report = dict(equal=int(same.sum()), total=len(same), rate=float(same.mean()),
                  old_positive=int((truth != '').sum()), new_positive=int((predicted != '').sum()),
                  gate_passed=False, scope='時間順60%学習/40%評価の標本・全フレーム比較未実施')
    (args.output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
