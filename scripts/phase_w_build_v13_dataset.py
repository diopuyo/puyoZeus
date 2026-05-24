"""W9-D: v13 訓練データ構築。

manual_plus_strict (451K, pl1-4) +
pseudo_v7_all19 (38K) +
W9-B 弱点動画 5 sheet review (~977 cells = 977 patches) +
v05_m55_full (200) + v12_m54_full (323) +
violations_50_bg/v04..v19 (16×50=800) =
合計 ~493K + ~2300 (review が高品質)
"""
from __future__ import annotations

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

LABEL_TO_CODE = {
    "EM": 0, "RED": 1, "BLUE": 2, "GRN": 3,
    "YEL": 4, "PUR": 5, "OJM": 9,
}
PATCH_SIZE = 16


def collect_review(csv_path: Path, video_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists() or not video_path.exists():
        return np.empty((0, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8), \
               np.empty((0,), dtype=np.int32)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.empty((0, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8), \
               np.empty((0,), dtype=np.int32)
    patches: list[np.ndarray] = []
    labels: list[int] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ans = (row.get("your_answer") or "").strip()
            if not ans or ans not in LABEL_TO_CODE:
                continue
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
            resized = cv2.resize(
                patch, (PATCH_SIZE, PATCH_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            patches.append(resized)
            labels.append(LABEL_TO_CODE[ans])
    cap.release()
    if not patches:
        return np.empty((0, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8), \
               np.empty((0,), dtype=np.int32)
    return np.stack(patches), np.array(labels, dtype=np.int32)


def main() -> int:
    base = Path("data/verify/phase_w_review")
    review_specs = [
        ("v05_m55_full", "video_05"),
        ("v12_m54_full", "video_12"),
        ("v09_m02_full", "video_09"),
        ("v13_m02_full", "video_13"),
        ("v17_m11_full", "video_17"),
        ("v18_m03_full", "video_18"),
        ("v19_m06_full", "video_19"),
    ]
    for n in range(4, 20):
        review_specs.append((f"violations_50_bg/v{n:02d}", f"video_{n:02d}"))

    all_p: list[np.ndarray] = []
    all_l: list[np.ndarray] = []
    for name, vid in review_specs:
        ps, ls = collect_review(
            base / name / "labels.csv",
            Path(f"data/frames/{vid}.mp4"),
        )
        if ps.size:
            print(f"  review {name}: {ps.shape}")
            all_p.append(ps)
            all_l.append(ls)

    review_patches = np.concatenate(all_p, axis=0)
    review_labels = np.concatenate(all_l, axis=0)
    print(f"review total: {review_patches.shape}")

    # 結合
    sources = [
        ("manual_plus_strict",
         "data/training_phase_u/manual_plus_strict.npz"),
        ("pseudo_v7_all19",
         "data/training_phase_u/pseudo_v7_all19.npz"),
    ]
    parts_p = [review_patches]
    parts_l = [review_labels]
    for name, path in sources:
        d = np.load(path)
        print(f"  {name}: {d['patches'].shape}")
        parts_p.append(d["patches"])
        parts_l.append(d["labels"].astype(np.int32))

    patches = np.concatenate(parts_p, axis=0)
    labels = np.concatenate(parts_l, axis=0)
    print(f"total: {patches.shape}")
    unique, counts = np.unique(labels, return_counts=True)
    print("labels:", dict(zip(unique.tolist(), counts.tolist())))

    out = Path("data/training_phase_u/v13_dataset.npz")
    np.savez(
        out,
        patches=patches, labels=labels,
        review_count=np.array([review_patches.shape[0]], dtype=np.int32),
    )
    print(f"saved: {to_windows_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
