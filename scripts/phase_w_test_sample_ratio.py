"""W15-D: CELL_SAMPLE_RATIO の感度テスト。

異なる sample 範囲 (0.5, 0.65, 0.8, 0.9, 1.0) で v16 の accuracy を比較。
ROI 中心位置のずれや背景影響への robustness を確認する。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_test_sample_ratio \
        --csv data/verify/phase_w_review/v18_m03_field2/labels.csv \
        --video data/frames/video_18.mp4
"""
from __future__ import annotations

import argparse
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

from src.board import HIDDEN_ROWS, BOARD_COLS, VISIBLE_ROWS
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.patch_classifier import CnnPatchClassifier

LABEL_TO_CODE = {
    "EM": 0, "RED": 1, "BLUE": 2, "GRN": 3,
    "YEL": 4, "PUR": 5, "OJM": 9,
}
CODE_TO_LABEL = {v: k for k, v in LABEL_TO_CODE.items()}


def custom_cell_rect(
    region, row: int, col: int, ratio: float,
) -> tuple[int, int, int, int]:
    """region 内 cell の指定比率の中央矩形を返す。"""
    cell_w = region.width / BOARD_COLS
    cell_h = region.height / VISIBLE_ROWS
    visible_row = row - HIDDEN_ROWS
    cx = int(region.x + (col + 0.5) * cell_w)
    cy = int(region.y + (visible_row + 0.5) * cell_h)
    half_w = max(1, int(cell_w * ratio / 2))
    half_h = max(1, int(cell_h * ratio / 2))
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def evaluate_at_ratio(
    csv_path: Path, video_path: Path,
    classifier: CnnPatchClassifier,
    ratio: float,
) -> tuple[int, int, dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0, {}
    correct = 0
    total = 0
    cm: dict[tuple[int, int], int] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ans = (row.get("your_answer") or "").strip()
            if not ans or ans not in LABEL_TO_CODE:
                continue
            truth = LABEL_TO_CODE[ans]
            t = float(row["time"])
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            if fr.shape[:2] != (1080, 1920):
                fr = cv2.resize(fr, (1920, 1080))
            region = (
                DEFAULT_P1_REGION if row["side"] == "1P"
                else DEFAULT_P2_REGION
            )
            r = int(row["row"]) + HIDDEN_ROWS
            c = int(row["col"])
            x1, y1, x2, y2 = custom_cell_rect(region, r, c, ratio)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(1920, x2)
            y2 = min(1080, y2)
            patch = fr[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            pred = classifier.classify(patch)
            total += 1
            if pred == truth:
                correct += 1
            else:
                cm[(truth, pred)] = cm.get((truth, pred), 0) + 1
    cap.release()
    return correct, total, cm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--model", default="models/cnn_phase_u_v16.pt",
    )
    parser.add_argument(
        "--ratios", nargs="+", type=float,
        default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    args = parser.parse_args()

    classifier = CnnPatchClassifier.load(Path(args.model))
    print(f"model: {args.model}")
    print(f"csv: {args.csv}")
    print(f"video: {args.video}\n")

    for ratio in args.ratios:
        c, t, cm = evaluate_at_ratio(
            Path(args.csv), Path(args.video), classifier, ratio,
        )
        if t == 0:
            continue
        err = sorted(cm.items(), key=lambda kv: -kv[1])[:5]
        err_str = " ".join(
            f"{CODE_TO_LABEL[k[0]]}->{CODE_TO_LABEL[k[1]]}:{v}"
            for k, v in err
        ) or "(no errors)"
        print(
            f"ratio={ratio:.2f}: {c}/{t} = {c/t:.4f}  {err_str}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
