"""W11-A: v7 + ColorRecoveryRefiner の cross-video 評価。

CNN が EM 判定した cell でのみ recovery を試行。saturation + centroid 距離の
両方を満たすときのみ色を割り当て。

cross-video harness で v7 単独 vs v7+recovery を比較。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2
import numpy as np

from src.board import HIDDEN_ROWS, COLOR_EMPTY
from src.centroid_classifier import CentroidClassifier
from src.color_recovery_refiner import ColorRecoveryRefiner
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.patch_classifier import CnnPatchClassifier

LABEL_TO_CODE = {
    "EM": 0, "RED": 1, "BLUE": 2, "GRN": 3,
    "YEL": 4, "PUR": 5, "OJM": 9,
}
CODE_TO_LABEL = {v: k for k, v in LABEL_TO_CODE.items()}


def review_sets() -> list[tuple[str, Path, Path]]:
    base = Path("data/verify/phase_w_review")
    sets: list[tuple[str, Path, Path]] = [
        ("v05_m55_full",
         base / "v05_m55_full" / "labels.csv",
         Path("data/frames/video_05.mp4")),
        ("v12_m54_full",
         base / "v12_m54_full" / "labels.csv",
         Path("data/frames/video_12.mp4")),
    ]
    for name, vid in (
        ("v09_m02_full", "09"), ("v13_m02_full", "13"),
        ("v17_m11_full", "17"), ("v18_m03_full", "18"),
        ("v18_m08_full", "18"), ("v19_m06_full", "19"),
    ):
        sets.append((
            name, base / name / "labels.csv",
            Path(f"data/frames/video_{vid}.mp4"),
        ))
    for n in range(4, 20):
        v = f"v{n:02d}"
        sets.append((
            f"viol_{v}",
            base / "violations_50_bg" / v / "labels.csv",
            Path(f"data/frames/video_{n:02d}.mp4"),
        ))
    return sets


def evaluate(
    name: str, csv_path: Path, video_path: Path,
    v7: CnnPatchClassifier,
    recovery: ColorRecoveryRefiner,
) -> tuple[int, int, int, int, int]:
    """Returns (correct_v7, correct_with_recovery, total, n_recovered, n_helped)."""
    if not csv_path.exists() or not video_path.exists():
        return 0, 0, 0, 0, 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0, 0, 0, 0

    correct_v7 = 0
    correct_rec = 0
    total = 0
    n_recovered = 0
    n_helped = 0  # v7 wrong, recovery correct

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
            x1, y1, x2, y2 = region.cell_sample_rect(r, c)
            patch = fr[max(0, y1):min(1080, y2),
                       max(0, x1):min(1920, x2)]
            if patch.size == 0:
                continue
            v7_pred = v7.classify(patch)
            rec_pred = recovery.refine_cell(v7_pred, patch)
            total += 1
            if v7_pred == truth:
                correct_v7 += 1
            if rec_pred == truth:
                correct_rec += 1
            if rec_pred != v7_pred:
                n_recovered += 1
                if v7_pred != truth and rec_pred == truth:
                    n_helped += 1
    cap.release()
    return correct_v7, correct_rec, total, n_recovered, n_helped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--centroid", default="models/centroid_v2.npz",
    )
    parser.add_argument("--min-sat", type=float, default=70.0)
    parser.add_argument("--max-dist", type=float, default=25.0)
    parser.add_argument(
        "--out-tsv",
        default="data/verify/phase_w_eval_color_recovery.tsv",
    )
    args = parser.parse_args()

    v7 = CnnPatchClassifier.load(Path("models/cnn_phase_u_v7.pt"))
    centroid = CentroidClassifier()
    centroid.load(args.centroid)
    recovery = ColorRecoveryRefiner(
        centroid=centroid,
        min_saturation=args.min_sat,
        max_centroid_distance=args.max_dist,
    )

    rows: list[str] = [
        "set\tcorrect_v7\tcorrect_recovery\ttotal\tn_changed\tn_helped"
    ]
    grand = {"v7": 0, "rec": 0, "tot": 0, "ch": 0, "h": 0}
    for name, cp, vp in review_sets():
        c7, cr, t, nc, nh = evaluate(name, cp, vp, v7, recovery)
        if t == 0:
            continue
        rows.append(f"{name}\t{c7}\t{cr}\t{t}\t{nc}\t{nh}")
        delta = cr - c7
        sign = "+" if delta >= 0 else ""
        print(
            f"  {name}: v7={c7}/{t}={c7/t:.4f} "
            f"+rec={cr}/{t}={cr/t:.4f} ({sign}{delta}, "
            f"changed={nc}, helped={nh})"
        )
        grand["v7"] += c7
        grand["rec"] += cr
        grand["tot"] += t
        grand["ch"] += nc
        grand["h"] += nh

    rows.append(
        f"TOTAL\t{grand['v7']}\t{grand['rec']}\t{grand['tot']}\t"
        f"{grand['ch']}\t{grand['h']}"
    )
    print(
        f"\nTOTAL: v7={grand['v7']}/{grand['tot']}={grand['v7']/grand['tot']:.4f} "
        f"+rec={grand['rec']}/{grand['tot']}={grand['rec']/grand['tot']:.4f} "
        f"(changed={grand['ch']}, helped={grand['h']})"
    )

    out = Path(args.out_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"saved: {to_windows_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
