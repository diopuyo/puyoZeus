"""Phase Z CNN v18: weak_video_extra (v04/v06/v12/v16/v19 × 試合 3-5) 統合訓練。

phase_z_aggregate_weak_review.py で生成した phase_z_gt_weak.npz を、
v17b パターン (sampled ×30, v18 ×5) と類似のスキームで oversample して
v16 dataset へ統合。weak 動画の動画別偏りを直接埋める設計。

倍率設計 (デフォルト):
    weak ×40   : 弱点動画は cell 数が少ないため強めに oversample
    sampled ×30: 18 動画分の幅広いカバー (v17b と同値)
    v18 ×5     : v18 偏重を抑制 (v17b と同値)

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_train_cnn_v18

注意: phase_z_gt_weak.npz が空 (your_answer 未入力) のときは weak 部分を
スキップする。weak が小さい場合は --weak-multiplier を上げて補強。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import numpy as np  # noqa: E402

from src.patch_classifier import CLASS_INDEX_TO_COLOR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init-model", default="models/cnn_phase_u_v16.pt",
    )
    parser.add_argument(
        "--out-model", default="models/cnn_phase_u_v18.pt",
    )
    parser.add_argument(
        "--out-dataset",
        default="data/training_phase_u/v18_dataset.npz",
    )
    parser.add_argument("--weak-multiplier", type=int, default=40)
    parser.add_argument("--sampled-multiplier", type=int, default=30)
    parser.add_argument("--v18gt-multiplier", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-per-class", type=int, default=20000)
    args = parser.parse_args()

    # === v16_dataset.npz 読み込み ===
    v16_path = Path("data/training_phase_u/v16_dataset.npz")
    d = np.load(v16_path)
    v16_patches = d["patches"]
    v16_labels = d["labels"].astype(np.int32)
    print(f"v16_dataset: {v16_patches.shape} ({v16_labels.size} samples)")

    idx_to_code = np.array(CLASS_INDEX_TO_COLOR, dtype=np.int32)

    # === phase_z_gt_sampled.npz / phase_z_gt_v18.npz / phase_z_gt_weak.npz ===
    sampled = np.load("data/training_phase_u/phase_z_gt_sampled.npz")
    sampled_X = sampled["X"]
    sampled_y = idx_to_code[sampled["y"]]
    print(f"sampled review (18 動画): {sampled_X.shape}")

    v18gt = np.load("data/training_phase_u/phase_z_gt_v18.npz")
    v18gt_X = v18gt["X"]
    v18gt_y = idx_to_code[v18gt["y"]]
    print(f"v18_m03 fixed: {v18gt_X.shape}")

    weak_path = Path("data/training_phase_u/phase_z_gt_weak.npz")
    if weak_path.exists():
        weak = np.load(weak_path)
        weak_X = weak["X"]
        weak_y = idx_to_code[weak["y"]]
        print(f"weak review (5 弱点動画 × 試合 3-5): {weak_X.shape}")
    else:
        weak_X = np.empty((0, 8, 8, 3), dtype=np.uint8)
        weak_y = np.empty((0,), dtype=np.int32)
        print("[warn] phase_z_gt_weak.npz 未生成 → weak スキップ")

    # === oversample ===
    sm = args.sampled_multiplier
    sampled_X_x = np.tile(sampled_X, (sm, 1, 1, 1))
    sampled_y_x = np.tile(sampled_y, sm)
    print(f"sampled oversampled x{sm}: {sampled_X_x.shape}")

    vm = args.v18gt_multiplier
    v18gt_X_x = np.tile(v18gt_X, (vm, 1, 1, 1))
    v18gt_y_x = np.tile(v18gt_y, vm)
    print(f"v18_m03 oversampled x{vm}: {v18gt_X_x.shape}")

    if weak_X.size > 0:
        wm = args.weak_multiplier
        weak_X_x = np.tile(weak_X, (wm, 1, 1, 1))
        weak_y_x = np.tile(weak_y, wm)
        print(f"weak oversampled x{wm}: {weak_X_x.shape}")
    else:
        weak_X_x = weak_X
        weak_y_x = weak_y

    # === shape 統一 (v16_dataset 側に合わせる) ===
    if v16_patches.shape[1:3] != (8, 8):
        import cv2
        target_h, target_w = v16_patches.shape[1:3]
        sampled_X_x = np.array([
            cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_AREA)
            for p in sampled_X_x
        ], dtype=np.uint8)
        v18gt_X_x = np.array([
            cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_AREA)
            for p in v18gt_X_x
        ], dtype=np.uint8)
        if weak_X_x.size > 0:
            weak_X_x = np.array([
                cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_AREA)
                for p in weak_X_x
            ], dtype=np.uint8)
        print(f"resized to {target_h}x{target_w}")

    # === 統合 ===
    parts = [v16_patches, sampled_X_x, v18gt_X_x]
    label_parts = [v16_labels, sampled_y_x, v18gt_y_x]
    if weak_X_x.size > 0:
        parts.append(weak_X_x)
        label_parts.append(weak_y_x)
    patches = np.concatenate(parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    print(f"total: {patches.shape}")
    unique, counts = np.unique(labels, return_counts=True)
    print("labels:", dict(zip(unique.tolist(), counts.tolist())))
    n_phase_z = (
        sampled_X_x.shape[0] + v18gt_X_x.shape[0]
        + (weak_X_x.shape[0] if weak_X_x.size > 0 else 0)
    )
    print(
        f"phase_z 比率: {n_phase_z}/{patches.shape[0]} = "
        f"{100 * n_phase_z / patches.shape[0]:.2f}%"
    )

    out_ds = Path(args.out_dataset)
    out_ds.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_ds, patches=patches, labels=labels)
    print(f"saved dataset: {to_windows_path(out_ds)}")

    cmd = [
        "./venv/bin/python", "-m", "scripts.phase_u_train_cnn",
        "--input", str(out_ds),
        "--init-model", args.init_model,
        "--out-model", args.out_model,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch-size", "64",
        "--max-per-class", str(args.max_per_class),
        "--holdout-ratio", "0.10",
    ]
    print(f"\n=== training v18 ===\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
