"""W15-D: cell 中心位置の ±N px offset 感度テスト。

(dx, dy) のシフトを各 -3..+3 px で grid search し、最良シフトを発見。
動画ごとに ROI が微妙にずれている可能性を検証。
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


def cell_rect_with_offset(
    region, row: int, col: int, ratio: float,
    dx: int, dy: int,
) -> tuple[int, int, int, int]:
    cell_w = region.width / BOARD_COLS
    cell_h = region.height / VISIBLE_ROWS
    visible_row = row - HIDDEN_ROWS
    cx = int(region.x + (col + 0.5) * cell_w) + dx
    cy = int(region.y + (visible_row + 0.5) * cell_h) + dy
    half_w = max(1, int(cell_w * ratio / 2))
    half_h = max(1, int(cell_h * ratio / 2))
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def evaluate_at_offset(
    csv_path: Path, video_path: Path,
    classifier: CnnPatchClassifier,
    ratio: float, dx: int, dy: int,
) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0
    correct = 0
    total = 0
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
            x1, y1, x2, y2 = cell_rect_with_offset(
                region, r, c, ratio, dx, dy,
            )
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
    cap.release()
    return correct, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--model", default="models/cnn_phase_u_v16.pt",
    )
    parser.add_argument("--ratio", type=float, default=0.6)
    parser.add_argument("--range", type=int, default=4)
    args = parser.parse_args()

    classifier = CnnPatchClassifier.load(Path(args.model))
    print(f"model: {args.model}, ratio={args.ratio}\n")

    R = args.range
    grid = np.zeros((2 * R + 1, 2 * R + 1), dtype=np.float32)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            c, t = evaluate_at_offset(
                Path(args.csv), Path(args.video), classifier,
                args.ratio, dx, dy,
            )
            acc = c / max(1, t)
            grid[dy + R, dx + R] = acc
            print(
                f"  dx={dx:+d} dy={dy:+d}: {c}/{t} = {acc:.4f}"
            )
    # 最良
    best = np.unravel_index(np.argmax(grid), grid.shape)
    best_dy = best[0] - R
    best_dx = best[1] - R
    print(
        f"\nbest: dx={best_dx:+d} dy={best_dy:+d} "
        f"acc={grid[best]:.4f}"
    )
    print(f"baseline (0,0) acc={grid[R, R]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
