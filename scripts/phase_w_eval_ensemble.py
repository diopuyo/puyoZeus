"""W10-A: Ensemble (v7 + centroid + bg-em) を cross-video で評価。"""
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
from src.board import HIDDEN_ROWS
from src.centroid_classifier import CentroidClassifier
from src.ensemble_classifier import EnsembleClassifier
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.patch_classifier import CnnPatchClassifier

LABEL_TO_CODE = {
    "EM": 0, "RED": 1, "BLUE": 2, "GRN": 3,
    "YEL": 4, "PUR": 5, "OJM": 9,
}
CODE_TO_LABEL = {v: k for k, v in LABEL_TO_CODE.items()}


def review_sets() -> list[dict]:
    base = Path("data/verify/phase_w_review")
    starts: dict[str, float] = {
        "v05_m55_full": 3252.0, "v12_m54_full": 3450.0,
        "v09_m02_full": 323.0, "v13_m02_full": 276.0,
        "v17_m11_full": 689.0, "v18_m03_full": 251.0,
        "v19_m06_full": 532.0,
    }
    sets: list[dict] = []
    for name, vid in (
        ("v05_m55_full", "05"), ("v12_m54_full", "12"),
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


def evaluate_set(s: dict, ens: EnsembleClassifier) -> tuple[int, int, int, int, dict]:
    cap = cv2.VideoCapture(str(s["video"]))
    if not cap.isOpened():
        return 0, 0, 0, 0, {}

    if ens.bg is not None:
        ens.bg.calibrate_from_video(cap, s["start"])

    correct_cnn = 0
    correct_ens = 0
    total = 0
    n_diff = 0
    err_ens: dict[tuple[int, int], int] = {}

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

            cnn_p = ens.cnn.classify(patch)
            ens_p = ens.classify(
                patch, row["side"], int(row["row"]), int(row["col"]),
            )
            total += 1
            if cnn_p == truth:
                correct_cnn += 1
            if ens_p == truth:
                correct_ens += 1
            if ens_p != cnn_p:
                n_diff += 1
            if ens_p != truth:
                err_ens[(truth, ens_p)] = err_ens.get((truth, ens_p), 0) + 1
    cap.release()
    return correct_cnn, correct_ens, total, n_diff, err_ens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-tsv",
        default="data/verify/phase_w_eval_ensemble.tsv",
    )
    args = parser.parse_args()

    cnn = CnnPatchClassifier.load(Path("models/cnn_phase_u_v7.pt"))
    centroid = CentroidClassifier()
    centroid.load("models/centroid_v1.npz")
    bg = BgEmptyDetector(threshold=18.0)
    ens = EnsembleClassifier(cnn=cnn, centroid=centroid, bg=bg)

    rows: list[str] = [
        "set\tcorrect_cnn\tcorrect_ens\ttotal\tn_diff\ttop_errors_ens"
    ]
    grand = {"cnn": 0, "ens": 0, "tot": 0, "diff": 0}
    for s in review_sets():
        c_cnn, c_ens, t, nd, em = evaluate_set(s, ens)
        if t == 0:
            continue
        err_pairs = sorted(
            ((k, v) for k, v in em.items() if k[0] != k[1]),
            key=lambda kv: -kv[1],
        )[:5]
        err_str = " ".join(
            f"{CODE_TO_LABEL[k[0]]}->{CODE_TO_LABEL[k[1]]}:{v}"
            for k, v in err_pairs
        ) or "(no errors)"
        rows.append(
            f"{s['name']}\t{c_cnn}\t{c_ens}\t{t}\t{nd}\t{err_str}"
        )
        delta = c_ens - c_cnn
        sign = "+" if delta >= 0 else ""
        print(
            f"  {s['name']}: cnn={c_cnn}/{t}={c_cnn/t:.4f} "
            f"ens={c_ens}/{t}={c_ens/t:.4f} ({sign}{delta}, diff={nd}) {err_str}"
        )
        grand["cnn"] += c_cnn
        grand["ens"] += c_ens
        grand["tot"] += t
        grand["diff"] += nd

    rows.append(
        f"TOTAL\t{grand['cnn']}\t{grand['ens']}\t{grand['tot']}\t"
        f"{grand['diff']}\t-"
    )
    print(
        f"\nTOTAL: cnn={grand['cnn']}/{grand['tot']}={grand['cnn']/grand['tot']:.4f} "
        f"ens={grand['ens']}/{grand['tot']}={grand['ens']/grand['tot']:.4f} "
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
