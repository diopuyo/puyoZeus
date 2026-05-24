"""W9-F: v7 + BgEmptyDetector overlay の cross-video 評価。

戦略:
    1. 各 review set の match start 時刻で BgEmptyDetector をキャリブレーション
    2. v7 (CnnPatchClassifier) で predict
    3. v7 が非 EM 判定したセルでも BgEmptyDetector が EM 判定 → EM に上書き
    4. 通常の v7 vs (v7 + BG-EM) で acc 比較

期待: EM hallucination (BLUE/GRN/RED が背景ノイズに反応) を抑制し
v18_m03 (78%) や v19_m06 (93%) で大幅改善。
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

from src.bg_empty_detector import BgEmptyDetector
from src.board import COLOR_EMPTY, HIDDEN_ROWS
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.patch_classifier import CnnPatchClassifier

LABEL_TO_CODE = {
    "EM": 0, "RED": 1, "BLUE": 2, "GRN": 3,
    "YEL": 4, "PUR": 5, "OJM": 9,
}
CODE_TO_LABEL = {v: k for k, v in LABEL_TO_CODE.items()}


def review_sets() -> list[dict]:
    """各 review set の {name, csv, video, match_start} 情報。"""
    base = Path("data/verify/phase_w_review")
    sets: list[dict] = []
    starts: dict[str, float] = {
        "v05_m55_full": 3252.0,
        "v12_m54_full": 3450.0,
        "v09_m02_full": 323.0,
        "v13_m02_full": 276.0,
        "v17_m11_full": 689.0,
        "v18_m03_full": 251.0,
        "v19_m06_full": 532.0,
    }
    sets.append({
        "name": "v05_m55_full",
        "csv": base / "v05_m55_full" / "labels.csv",
        "video": Path("data/frames/video_05.mp4"),
        "start": starts["v05_m55_full"],
    })
    sets.append({
        "name": "v12_m54_full",
        "csv": base / "v12_m54_full" / "labels.csv",
        "video": Path("data/frames/video_12.mp4"),
        "start": starts["v12_m54_full"],
    })
    for name, vid in (
        ("v09_m02_full", "09"), ("v13_m02_full", "13"),
        ("v17_m11_full", "17"), ("v18_m03_full", "18"),
        ("v19_m06_full", "19"),
    ):
        sets.append({
            "name": name,
            "csv": base / name / "labels.csv",
            "video": Path(f"data/frames/video_{vid}.mp4"),
            "start": starts[name],
        })
    # violations_50_bg は match_boundaries の最初の試合開始秒で代用
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


def load_v7() -> CnnPatchClassifier:
    return CnnPatchClassifier.load(Path("models/cnn_phase_u_v7.pt"))


def evaluate_set(
    s: dict, v7: CnnPatchClassifier, threshold: float,
) -> tuple[int, int, int, int, dict]:
    """
    Returns:
        (correct_v7, correct_with_bg, total, n_overrides, error_dict_with_bg)
    """
    if not s["csv"].exists() or not s["video"].exists():
        return 0, 0, 0, 0, {}
    cap = cv2.VideoCapture(str(s["video"]))
    if not cap.isOpened():
        return 0, 0, 0, 0, {}

    bg = BgEmptyDetector(threshold=threshold)
    bg.calibrate_from_video(cap, s["start"])

    correct_v7 = 0
    correct_with = 0
    total = 0
    n_overrides = 0
    error_with: dict[tuple[int, int], int] = {}

    with open(s["csv"], encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            truth_str = (row.get("your_answer") or "").strip()
            if not truth_str or truth_str not in LABEL_TO_CODE:
                continue
            truth = LABEL_TO_CODE[truth_str]
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
            # BG-EM オーバーレイ: BG が EM 判定すれば EM に上書き
            override = bg.is_empty(
                row["side"], int(row["row"]), int(row["col"]), patch,
            )
            pred_with = COLOR_EMPTY if override else pred_v7
            if override and pred_v7 != COLOR_EMPTY:
                n_overrides += 1

            total += 1
            if pred_v7 == truth:
                correct_v7 += 1
            if pred_with == truth:
                correct_with += 1
            if pred_with != truth:
                error_with[(truth, pred_with)] = (
                    error_with.get((truth, pred_with), 0) + 1
                )
    cap.release()
    return correct_v7, correct_with, total, n_overrides, error_with


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold", type=float, default=18.0,
        help="L2 距離閾値 (BG セルとマッチとみなす上限)",
    )
    parser.add_argument(
        "--out-tsv",
        default="data/verify/phase_w_eval_v7_bgem.tsv",
    )
    args = parser.parse_args()

    v7 = load_v7()

    rows: list[str] = [
        "set\tcorrect_v7\tcorrect_v7+bgem\ttotal\toverrides\ttop_errors_v7+bgem"
    ]
    grand = {"v7": 0, "bg": 0, "tot": 0, "ovr": 0}
    for s in review_sets():
        c7, cb, t, ov, em_err = evaluate_set(s, v7, args.threshold)
        if t == 0:
            continue
        err_pairs = sorted(
            ((k, v) for k, v in em_err.items() if k[0] != k[1]),
            key=lambda kv: -kv[1],
        )[:5]
        err_str = " ".join(
            f"{CODE_TO_LABEL[k[0]]}->{CODE_TO_LABEL[k[1]]}:{v}"
            for k, v in err_pairs
        ) or "(no errors)"
        rows.append(
            f"{s['name']}\t{c7}\t{cb}\t{t}\t{ov}\t{err_str}"
        )
        delta = cb - c7
        sign = "+" if delta >= 0 else ""
        print(
            f"  {s['name']}: v7={c7}/{t}={c7/t:.4f}  "
            f"+bgem={cb}/{t}={cb/t:.4f}  ({sign}{delta}, ovr={ov})  "
            f"{err_str}"
        )
        grand["v7"] += c7
        grand["bg"] += cb
        grand["tot"] += t
        grand["ovr"] += ov

    rows.append(
        f"TOTAL\t{grand['v7']}\t{grand['bg']}\t{grand['tot']}\t{grand['ovr']}\t-"
    )
    print(
        f"\nTOTAL: v7={grand['v7']}/{grand['tot']}={grand['v7']/grand['tot']:.4f} "
        f"v7+bgem={grand['bg']}/{grand['tot']}={grand['bg']/grand['tot']:.4f} "
        f"overrides={grand['ovr']}"
    )

    out = Path(args.out_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"saved: {to_windows_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
