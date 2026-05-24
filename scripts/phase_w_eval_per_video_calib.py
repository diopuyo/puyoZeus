"""W11-C: PerVideoCalibrator + v7 の cross-video 評価。

各 review set の試合開始秒で BGR shift をキャリブレーション、
それを cell パッチに適用してから v7 で分類、accuracy 比較。
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

from src.board import HIDDEN_ROWS
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.patch_classifier import CnnPatchClassifier
from src.per_video_calibrator import PerVideoCalibrator

LABEL_TO_CODE = {
    "EM": 0, "RED": 1, "BLUE": 2, "GRN": 3,
    "YEL": 4, "PUR": 5, "OJM": 9,
}


def review_sets() -> list[dict]:
    base = Path("data/verify/phase_w_review")
    starts: dict[str, float] = {
        "v05_m55_full": 3252.0, "v12_m54_full": 3450.0,
        "v09_m02_full": 323.0, "v13_m02_full": 276.0,
        "v17_m11_full": 689.0, "v18_m03_full": 251.0,
        "v18_m08_full": 580.0, "v19_m06_full": 532.0,
    }
    sets: list[dict] = []
    for name, vid in (
        ("v05_m55_full", "05"), ("v12_m54_full", "12"),
        ("v09_m02_full", "09"), ("v13_m02_full", "13"),
        ("v17_m11_full", "17"), ("v18_m03_full", "18"),
        ("v18_m08_full", "18"), ("v19_m06_full", "19"),
    ):
        sets.append({
            "name": name,
            "csv": base / name / "labels.csv",
            "video": Path(f"data/frames/video_{vid}.mp4"),
            "start": starts[name],
        })
    for n in range(4, 20):
        v = f"v{n:02d}"
        match_tsv = (
            Path("data/verify/match_boundaries_v5") /
            f"video_{n:02d}" / "matches.tsv"
        )
        if not match_tsv.exists():
            match_tsv = (
                Path("data/verify/match_boundaries_v4") /
                f"video_{n:02d}" / "matches.tsv"
            )
        first_start = 0.0
        if match_tsv.exists():
            with open(match_tsv, encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    try:
                        first_start = float(r["start_sec"])
                        break
                    except (KeyError, ValueError):
                        continue
        sets.append({
            "name": f"viol_{v}",
            "csv": base / "violations_50_bg" / v / "labels.csv",
            "video": Path(f"data/frames/video_{n:02d}.mp4"),
            "start": first_start,
        })
    return sets


def evaluate(s: dict, v7: CnnPatchClassifier) -> tuple[int, int, int, int]:
    cap = cv2.VideoCapture(str(s["video"]))
    if not cap.isOpened():
        return 0, 0, 0, 0

    calib = PerVideoCalibrator()
    n_used = calib.calibrate_from_video(cap, s["start"])

    correct_v7 = 0
    correct_calib = 0
    total = 0
    n_diff = 0

    with open(s["csv"], encoding="utf-8") as f:
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
            pred_v7 = v7.classify(patch)
            patch_calib = calib.apply(patch)
            pred_calib = v7.classify(patch_calib)
            total += 1
            if pred_v7 == truth:
                correct_v7 += 1
            if pred_calib == truth:
                correct_calib += 1
            if pred_calib != pred_v7:
                n_diff += 1
    cap.release()
    return correct_v7, correct_calib, total, n_diff


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-tsv",
        default="data/verify/phase_w_eval_per_video_calib.tsv",
    )
    args = parser.parse_args()

    v7 = CnnPatchClassifier.load(Path("models/cnn_phase_u_v7.pt"))

    rows: list[str] = ["set\tcorrect_v7\tcorrect_calib\ttotal\tn_diff"]
    grand = {"v7": 0, "c": 0, "tot": 0, "diff": 0}
    for s in review_sets():
        c7, cc, t, nd = evaluate(s, v7)
        if t == 0:
            continue
        rows.append(f"{s['name']}\t{c7}\t{cc}\t{t}\t{nd}")
        delta = cc - c7
        sign = "+" if delta >= 0 else ""
        print(
            f"  {s['name']}: v7={c7}/{t}={c7/t:.4f} "
            f"+calib={cc}/{t}={cc/t:.4f} ({sign}{delta}, diff={nd})"
        )
        grand["v7"] += c7
        grand["c"] += cc
        grand["tot"] += t
        grand["diff"] += nd

    rows.append(
        f"TOTAL\t{grand['v7']}\t{grand['c']}\t{grand['tot']}\t{grand['diff']}"
    )
    print(
        f"\nTOTAL: v7={grand['v7']}/{grand['tot']}={grand['v7']/grand['tot']:.4f} "
        f"+calib={grand['c']}/{grand['tot']}={grand['c']/grand['tot']:.4f} "
        f"diff={grand['diff']}"
    )

    out = Path(args.out_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"saved: {to_windows_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
