"""V1.9: CNN v7 を動画ソース別に評価して汎化性能を測定。

データソース:
    - manual: data/training_phase_u/manual_labels.npz (video_01, 4771 件)
    - parallel_strict: data/training_phase_u/parallel_relabeled_strict/*.npz
        各 npz は pl1〜pl4 の動画別 (約 6 万件 × 119 動画)

評価:
    - 各 npz から N 件 (default 1000) ランダムサンプル
    - CNN v7 で predict、accuracy 計測
    - 結果を tsv に出力
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2
import numpy as np
import torch

from src.patch_classifier import (
    CLASS_INDEX_TO_COLOR,
    CnnPatchClassifier,
    PATCH_RESIZE_H,
    PATCH_RESIZE_W,
)


def load_cnn(model_path: str) -> CnnPatchClassifier:
    cnn = CnnPatchClassifier()
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    cnn._model.load_state_dict(state)
    cnn._model.eval()
    return cnn


def patches_to_tensor(patches: np.ndarray) -> torch.Tensor:
    N = patches.shape[0]
    out = np.zeros((N, PATCH_RESIZE_H, PATCH_RESIZE_W, 6), dtype=np.float32)
    for i in range(N):
        resized = cv2.resize(
            patches[i], (PATCH_RESIZE_W, PATCH_RESIZE_H),
            interpolation=cv2.INTER_AREA,
        )
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        combined = np.concatenate([resized, hsv], axis=2)
        out[i] = combined.astype(np.float32) / 255.0
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()


def evaluate_npz(
    cnn: CnnPatchClassifier,
    path: Path,
    n_sample: int,
    rng: random.Random,
    batch: int = 1024,
) -> tuple[int, int]:
    """1 npz の accuracy を計算。(correct, total)."""
    d = np.load(path)
    patches = d["patches"]
    labels = d["labels"].astype(np.int64)
    N = patches.shape[0]
    if N > n_sample:
        idx = rng.sample(range(N), n_sample)
        idx_arr = np.array(sorted(idx), dtype=np.int64)
        patches = patches[idx_arr]
        labels = labels[idx_arr]
    M = patches.shape[0]
    if M == 0:
        return 0, 0

    correct = 0
    for s in range(0, M, batch):
        e = min(s + batch, M)
        tensor = patches_to_tensor(patches[s:e])
        with torch.no_grad():
            logits = cnn._model(tensor)
            idx = torch.argmax(logits, dim=1).cpu().numpy()
        for j, ci in enumerate(idx):
            pred = CLASS_INDEX_TO_COLOR[ci]
            if pred == int(labels[s + j]):
                correct += 1
    return correct, M


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-model", default="models/cnn_phase_u_v7.pt")
    parser.add_argument(
        "--manual",
        default="data/training_phase_u/manual_labels.npz",
    )
    parser.add_argument(
        "--parallel-dir",
        default="data/training_phase_u/parallel_relabeled_strict",
    )
    parser.add_argument("--n-sample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-tsv",
        default="data/verify/phase_u_v7_eval.tsv",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"loading CNN: {args.cnn_model}")
    cnn = load_cnn(args.cnn_model)

    rows: list[str] = []
    rows.append("source\tcorrect\ttotal\taccuracy")

    # 1. manual (video_01)
    if Path(args.manual).exists():
        c, t = evaluate_npz(cnn, Path(args.manual), args.n_sample, rng)
        acc = c / max(1, t)
        rows.append(f"manual_video_01\t{c}\t{t}\t{acc:.4f}")
        print(f"manual_video_01: {c}/{t} = {acc:.4f}")

    # 2. parallel by pl tag
    parallel_dir = Path(args.parallel_dir)
    by_pl: dict[str, list[Path]] = {}
    for p in sorted(parallel_dir.glob("*.npz")):
        tag = p.stem.split("_")[0]  # pl1 / pl2 / pl3 / pl4
        by_pl.setdefault(tag, []).append(p)

    for pl, files in by_pl.items():
        # まとめて評価 (各 npz から sub-sample)
        per_npz = max(1, args.n_sample // len(files))
        c_sum = 0
        t_sum = 0
        for f in files:
            c, t = evaluate_npz(cnn, f, per_npz, rng)
            c_sum += c
            t_sum += t
        acc = c_sum / max(1, t_sum)
        rows.append(f"{pl}\t{c_sum}\t{t_sum}\t{acc:.4f}")
        print(f"{pl} ({len(files)} npz): {c_sum}/{t_sum} = {acc:.4f}")

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"saved: {to_windows_path(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
