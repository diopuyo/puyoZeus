"""W7-CNN: v05_m55 レビュー結果を既存データセットに追加して CNN v8 訓練。

入力:
    - data/verify/phase_w_review/v05_m55_full/labels.csv (155 件レビュー済)
    - data/frames/video_05.mp4 (パッチ抽出元)
    - data/training_phase_u/manual_labels.npz (4771 既存)

処理:
    1. labels.csv から (time, side, row, col, your_answer) を読む (your_answer != recognized も含む全件)
    2. video_05.mp4 の各時刻フレームから該当パッチを切り出し
    3. (patches, labels) npz として出力 → 既存 manual と結合
    4. CNN v8 訓練 (cnn_phase_u_v7.pt 初期化、x10 augment)

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_train_cnn_v8
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2
import numpy as np

from src.image_reader import (
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
)
from src.board import HIDDEN_ROWS

LABEL_TO_CODE = {
    "EM": 0, "RED": 1, "BLUE": 2, "GRN": 3,
    "YEL": 4, "PUR": 5, "OJM": 9,
}

PATCH_OUT_SIZE = 16


def extract_patch(
    frame: np.ndarray, side: str, vrow: int, vcol: int,
) -> np.ndarray | None:
    if frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(
            frame, (1920, 1080), interpolation=cv2.INTER_AREA,
        )
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    row = vrow + HIDDEN_ROWS
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region.cell_sample_rect(row, vcol)
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return cv2.resize(
        patch, (PATCH_OUT_SIZE, PATCH_OUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )


def build_npz_from_review(
    csv_path: Path,
    video_path: Path,
    out_path: Path,
) -> int:
    """labels.csv (your_answer 採用) と動画から npz 生成。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"video open failed: {video_path}")
        return 0

    patches = []
    labels = []
    skipped = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            truth = row.get("your_answer", "").strip()
            if not truth:
                truth = row.get("recognized", "").strip()
            if truth not in LABEL_TO_CODE:
                skipped += 1
                continue
            code = LABEL_TO_CODE[truth]
            t = float(row["time"])
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, fr = cap.read()
            if not ok or fr is None:
                skipped += 1
                continue
            patch = extract_patch(
                fr, row["side"],
                int(row["row"]), int(row["col"]),
            )
            if patch is None:
                skipped += 1
                continue
            patches.append(patch)
            labels.append(code)
    cap.release()

    if not patches:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        patches=np.array(patches, dtype=np.uint8),
        labels=np.array(labels, dtype=np.int32),
    )
    print(f"saved: {to_windows_path(out_path)} ({len(patches)} samples, "
          f"skipped={skipped})")
    unique, counts = np.unique(labels, return_counts=True)
    for c, n in zip(unique, counts):
        print(f"  code={c}: {n}")
    return len(patches)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/verify/phase_w_review/v05_m55_full/labels.csv",
    )
    parser.add_argument("--video", default="data/frames/video_05.mp4")
    parser.add_argument(
        "--out-npz",
        default="data/training_phase_u/v05_m55_review.npz",
    )
    parser.add_argument(
        "--manual-npz",
        default="data/training_phase_u/manual_labels.npz",
    )
    parser.add_argument(
        "--combined-npz",
        default="data/training_phase_u/manual_plus_v05_review.npz",
    )
    parser.add_argument(
        "--init-model", default="models/cnn_phase_u_v7.pt",
    )
    parser.add_argument(
        "--out-model", default="models/cnn_phase_u_v8.pt",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument(
        "--skip-train", action="store_true",
        help="npz 作成のみで訓練スキップ (デバッグ用)",
    )
    args = parser.parse_args()

    # 1. レビュー結果を npz に
    n = build_npz_from_review(
        Path(args.csv), Path(args.video), Path(args.out_npz),
    )
    if n == 0:
        print("no samples extracted")
        return 1

    # 2. 既存 manual と結合
    manual = np.load(args.manual_npz)
    review = np.load(args.out_npz)
    merged_patches = np.concatenate([manual["patches"], review["patches"]])
    merged_labels = np.concatenate([manual["labels"], review["labels"]])
    np.savez(
        args.combined_npz,
        patches=merged_patches, labels=merged_labels,
    )
    print(
        f"combined: {merged_patches.shape}, "
        f"manual={len(manual['patches'])}, review={n}"
    )

    if args.skip_train:
        return 0

    # 3. CNN v8 訓練 (既存 phase_u_train_cnn を呼ぶ)
    import subprocess
    cmd = [
        "./venv/bin/python", "-m", "scripts.phase_u_train_cnn",
        "--input", args.combined_npz,
        "--init-model", args.init_model,
        "--out-model", args.out_model,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch-size", "64",
        "--max-per-class", "0",
        "--holdout-ratio", "0.10",
    ]
    print(f"\n=== training CNN v8 ===\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
