"""×印候補の後半標本を比較し、不一致の時刻と側を保存する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.compare_image_free_digits import cv2, np, frame_at
from scripts.obs_priority_guard import ObsPriorityGuard
from scripts.train_image_free_ui_mask import patches, SAMPLES, TRAIN_FRACTION
from src.image_free_ui_mask import ImageFreeUiMaskMatcher
from src.ui_mask import UiMaskMatcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('video', 'templates', 'model', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    guard = ObsPriorityGuard()
    guard.checkpoint(force=True)
    cv2.setNumThreads(1)
    old = UiMaskMatcher.load_default(args.templates)
    new = ImageFreeUiMaskMatcher(args.model)
    cap = cv2.VideoCapture(str(args.video))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    times = np.linspace(1, duration - 1, SAMPLES)[int(SAMPLES * TRAIN_FRACTION):]
    differences = []
    for i, stamp in enumerate(times):
        guard.checkpoint()
        for side, patch in enumerate(patches(frame_at(cap, float(stamp)))):
            a, b = old.match(patch), new.match(patch)
            if a.is_ui != b.is_ui:
                differences.append(dict(t_sec=float(stamp), side=side + 1,
                    old_is_ui=a.is_ui, new_is_ui=b.is_ui,
                    old_score=a.score, new_score=b.score,
                    old_name=a.template_name, new_name=b.template_name))
        if i % 100 == 0:
            print(f'×印差分 {i}/{len(times)} 差={len(differences)}', flush=True)
    cap.release()
    args.output.write_text(json.dumps(differences, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
