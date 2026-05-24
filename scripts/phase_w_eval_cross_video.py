"""W8-B: cross-video 評価ハーネス。

レビュー済みラベル (v05_m55_full / v12_m54_full) を ground truth として、
複数 CNN モデル (v7 / v8 / v9 / v10) を比較評価する。
真の cross-video 汎化 (mixed holdout の飽和を超えた実態) を測定する。

入力:
    - data/verify/phase_w_review/{v05_m55_full,v12_m54_full}/labels.csv
    - data/frames/video_{05,12}.mp4
    - models/cnn_phase_u_v{7,8,9}.pt   (8x8 architecture)
    - models/cnn_phase_u_v10.pt        (16x16 ResNet architecture)

出力:
    - data/verify/phase_w_eval_cross_video.tsv
    - 各モデル × 動画ごとに accuracy + 誤分類混同行列
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
from src.patch_classifier import (
    CnnPatchClassifier,
    PatchClassifier,
)
from src.patch_classifier_v2 import CnnPatchClassifierV2

LABEL_TO_CODE = {
    "EM": 0, "RED": 1, "BLUE": 2, "GRN": 3,
    "YEL": 4, "PUR": 5, "OJM": 9,
}
CODE_TO_LABEL = {v: k for k, v in LABEL_TO_CODE.items()}


def extract_patch_from_frame(
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
    return patch


def collect_patches(
    csv_path: Path, video_path: Path,
) -> tuple[list[np.ndarray], list[int]]:
    """labels.csv を読んで動画から (patch, truth_code) を収集。"""
    if not csv_path.exists() or not video_path.exists():
        return [], []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], []
    patches: list[np.ndarray] = []
    truths: list[int] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            truth = (row.get("your_answer") or "").strip()
            if not truth or truth not in LABEL_TO_CODE:
                continue
            t = float(row["time"])
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            patch = extract_patch_from_frame(
                fr, row["side"],
                int(row["row"]), int(row["col"]),
            )
            if patch is None:
                continue
            patches.append(patch)
            truths.append(LABEL_TO_CODE[truth])
    cap.release()
    return patches, truths


def load_model(model_path: Path) -> PatchClassifier:
    """v7/v8/v9 (8x8) と v10 (16x16) を自動判別してロード。

    CnnPatchClassifier.load() は classmethod で新しい instance を返すため、
    戻り値を必ず受け取る。CnnPatchClassifierV2.load() は instance method。
    """
    import torch
    state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    keys = list(state.keys())
    is_v10 = any("conv1.weight" in k or "_ResBlock" in k for k in keys) or any(
        k.endswith(".bn1.weight") for k in keys
    )
    if is_v10:
        cls_v2 = CnnPatchClassifierV2()
        cls_v2.load(model_path)
        return cls_v2
    return CnnPatchClassifier.load(model_path)


def evaluate_model(
    cls: PatchClassifier, patches: list[np.ndarray], truths: list[int],
) -> tuple[int, int, dict[tuple[int, int], int]]:
    correct = 0
    total = 0
    confusion: dict[tuple[int, int], int] = {}
    for p, t in zip(patches, truths):
        pred = cls.classify(p)
        total += 1
        if pred == t:
            correct += 1
        confusion[(t, pred)] = confusion.get((t, pred), 0) + 1
    return correct, total, confusion


def format_confusion(
    confusion: dict[tuple[int, int], int], top_k: int = 5,
) -> str:
    err = sorted(
        ((k, v) for k, v in confusion.items() if k[0] != k[1]),
        key=lambda kv: -kv[1],
    )[:top_k]
    if not err:
        return "(no errors)"
    parts = []
    for (t, p), n in err:
        tl = CODE_TO_LABEL.get(t, str(t))
        pl = CODE_TO_LABEL.get(p, str(p))
        parts.append(f"{tl}->{pl}:{n}")
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=[
            "models/cnn_phase_u_v7.pt",
            "models/cnn_phase_u_v8.pt",
            "models/cnn_phase_u_v9.pt",
            "models/cnn_phase_u_v10.pt",
        ],
    )
    default_sets = [
        "v05_m55_full:data/verify/phase_w_review/v05_m55_full/labels.csv:data/frames/video_05.mp4",
        "v12_m54_full:data/verify/phase_w_review/v12_m54_full/labels.csv:data/frames/video_12.mp4",
    ]
    # W9-B 弱点動画 full sheet review (5 動画 × ~200 cells)
    for name, vid in (
        ("v09_m02_full", "09"), ("v13_m02_full", "13"),
        ("v17_m11_full", "17"), ("v18_m03_full", "18"),
        ("v19_m06_full", "19"),
    ):
        default_sets.append(
            f"{name}:data/verify/phase_w_review/{name}/"
            f"labels.csv:data/frames/video_{vid}.mp4"
        )
    # violations_50_bg: 16 動画 × 50 cell の cross-video レビューを追加
    for n in range(4, 20):
        v = f"v{n:02d}"
        default_sets.append(
            f"viol_{v}:data/verify/phase_w_review/violations_50_bg/{v}/"
            f"labels.csv:data/frames/video_{n:02d}.mp4"
        )
    parser.add_argument(
        "--review-sets", nargs="+", default=default_sets,
    )
    parser.add_argument(
        "--out-tsv",
        default="data/verify/phase_w_eval_cross_video.tsv",
    )
    args = parser.parse_args()

    rows: list[str] = []
    rows.append("model\treview_set\tcorrect\ttotal\taccuracy\ttop_errors")

    # 全レビューセットの patches を先に収集 (動画読み込みは重いので 1 回)
    review_data: dict[str, tuple[list[np.ndarray], list[int]]] = {}
    for spec in args.review_sets:
        name, csv_p, video_p = spec.split(":", 2)
        print(f"collecting: {name} <- {csv_p}")
        patches, truths = collect_patches(Path(csv_p), Path(video_p))
        review_data[name] = (patches, truths)
        print(f"  {name}: {len(patches)} patches")

    # モデルごと評価
    for model_path in args.models:
        mp = Path(model_path)
        if not mp.exists():
            print(f"skip (not found): {model_path}")
            continue
        print(f"\n=== {mp.name} ===")
        cls = load_model(mp)
        for name, (patches, truths) in review_data.items():
            if not patches:
                continue
            c, t, cm = evaluate_model(cls, patches, truths)
            acc = c / max(1, t)
            err = format_confusion(cm)
            rows.append(f"{mp.name}\t{name}\t{c}\t{t}\t{acc:.4f}\t{err}")
            print(f"  {name}: {c}/{t} = {acc:.4f}  errors: {err}")

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\nsaved: {to_windows_path(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
