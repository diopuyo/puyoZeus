"""W8-B 補助: v9 と v10 の予測差分を細かく分析。

v05_m55 review labels で v9 と v10 の予測が違うセルだけを抽出し、
truth と組み合わせて v10 の改善 (v9 wrong, v10 correct) と
v10 の劣化 (v9 correct, v10 wrong) を可視化。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console  # noqa: E402
init_console()

import cv2
import numpy as np

from src.board import HIDDEN_ROWS
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.patch_classifier import CnnPatchClassifier
from src.patch_classifier_v2 import CnnPatchClassifierV2

LABEL_TO_CODE = {
    "EM": 0, "RED": 1, "BLUE": 2, "GRN": 3,
    "YEL": 4, "PUR": 5, "OJM": 9,
}
CODE_TO_LABEL = {v: k for k, v in LABEL_TO_CODE.items()}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/verify/phase_w_review/v05_m55_full/labels.csv",
    )
    parser.add_argument("--video", default="data/frames/video_05.mp4")
    args = parser.parse_args()
    csv_path = Path(args.csv)
    video_path = Path(args.video)

    cnn_v9 = CnnPatchClassifier.load(Path("models/cnn_phase_u_v9.pt"))
    cnn_v10 = CnnPatchClassifierV2()
    cnn_v10.load(Path("models/cnn_phase_u_v10.pt"))

    cap = cv2.VideoCapture(str(video_path))

    diffs = []
    v9_better = 0
    v10_better = 0
    both_wrong = 0
    both_right = 0

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            truth_str = (row.get("your_answer") or "").strip()
            if truth_str not in LABEL_TO_CODE:
                continue
            truth = LABEL_TO_CODE[truth_str]
            t = float(row["time"])
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, fr = cap.read()
            if not ok:
                continue
            if fr.shape[:2] != (1080, 1920):
                fr = cv2.resize(fr, (1920, 1080))
            region = (
                DEFAULT_P1_REGION if row["side"] == "1P"
                else DEFAULT_P2_REGION
            )
            r = int(row["row"]) + HIDDEN_ROWS
            c = int(row["col"])
            x1, y1, x2, y2 = region.cell_sample_rect(r, c)
            patch = fr[max(0, y1):min(1080, y2), max(0, x1):min(1920, x2)]
            if patch.size == 0:
                continue
            p9 = cnn_v9.classify(patch)
            p10 = cnn_v10.classify(patch)
            v9_correct = (p9 == truth)
            v10_correct = (p10 == truth)
            if v9_correct and v10_correct:
                both_right += 1
            elif not v9_correct and not v10_correct:
                both_wrong += 1
                diffs.append((row, truth, p9, p10, "BOTH_WRONG"))
            elif v9_correct and not v10_correct:
                v9_better += 1
                diffs.append((row, truth, p9, p10, "V9_BETTER"))
            else:
                v10_better += 1
                diffs.append((row, truth, p9, p10, "V10_BETTER"))

    cap.release()

    print(f"both_right: {both_right}")
    print(f"both_wrong: {both_wrong}")
    print(f"v10_better (v9 wrong, v10 correct): {v10_better}")
    print(f"v9_better (v9 correct, v10 wrong): {v9_better}")
    print(f"net v10 gain: {v10_better - v9_better}")
    print()
    print("--- v10_better cases ---")
    for row, t, p9, p10, tag in diffs:
        if tag == "V10_BETTER":
            print(f"  t={row['time']} {row['side']} r={row['row']} c={row['col']} "
                  f"truth={CODE_TO_LABEL[t]} v9={CODE_TO_LABEL[p9]} v10={CODE_TO_LABEL[p10]}")
    print()
    print("--- v9_better cases ---")
    for row, t, p9, p10, tag in diffs:
        if tag == "V9_BETTER":
            print(f"  t={row['time']} {row['side']} r={row['row']} c={row['col']} "
                  f"truth={CODE_TO_LABEL[t]} v9={CODE_TO_LABEL[p9]} v10={CODE_TO_LABEL[p10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
