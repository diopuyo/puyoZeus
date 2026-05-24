"""Phase Z CNN v17b: v17 失敗 (v18 偏重で過学習) を受けたバランス調整再訓練。

v17 で v18_m03 GT が 51% を占め、v12/v13/v07 等で精度悪化。
対策: sampled review (18 動画分) を主、v18 を抑制した oversample。

倍率設計:
    sampled (346) × 30 倍 = 10,380 → 18 動画から幅広くカバー
    v18 (360) × 5 倍 = 1,800   → v18 影響を抑制 (元 ×50 の 1/10)
    合計 12,180 (元 v17 35,300 から削減)

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_train_cnn_v17b
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
        "--out-model", default="models/cnn_phase_u_v17b.pt",
    )
    parser.add_argument(
        "--out-dataset",
        default="data/training_phase_u/v17b_dataset.npz",
    )
    parser.add_argument("--sampled-multiplier", type=int, default=30)
    parser.add_argument("--v18-multiplier", type=int, default=5)
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

    # === phase_z_gt split 読み込み ===
    sampled = np.load("data/training_phase_u/phase_z_gt_sampled.npz")
    sampled_X = sampled["X"]
    idx_to_code = np.array(CLASS_INDEX_TO_COLOR, dtype=np.int32)
    sampled_y = idx_to_code[sampled["y"]]
    print(f"sampled review (18 動画): {sampled_X.shape}")

    v18 = np.load("data/training_phase_u/phase_z_gt_v18.npz")
    v18_X = v18["X"]
    v18_y = idx_to_code[v18["y"]]
    print(f"v18_m03 fixed: {v18_X.shape}")

    # === oversample (バランス調整) ===
    sm = args.sampled_multiplier
    sampled_X_x = np.tile(sampled_X, (sm, 1, 1, 1))
    sampled_y_x = np.tile(sampled_y, sm)
    print(f"sampled oversampled x{sm}: {sampled_X_x.shape}")

    vm = args.v18_multiplier
    v18_X_x = np.tile(v18_X, (vm, 1, 1, 1))
    v18_y_x = np.tile(v18_y, vm)
    print(f"v18 oversampled x{vm}: {v18_X_x.shape}")

    # === 16x16 にリサイズ (v16_dataset と shape 統一) ===
    if v16_patches.shape[1:3] != (8, 8):
        import cv2
        target_h, target_w = v16_patches.shape[1:3]
        sampled_X_x = np.array([
            cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_AREA)
            for p in sampled_X_x
        ], dtype=np.uint8)
        v18_X_x = np.array([
            cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_AREA)
            for p in v18_X_x
        ], dtype=np.uint8)
        print(f"resized to {target_h}x{target_w}")

    # === 統合 ===
    patches = np.concatenate([v16_patches, sampled_X_x, v18_X_x], axis=0)
    labels = np.concatenate([v16_labels, sampled_y_x, v18_y_x], axis=0)
    print(f"total: {patches.shape}")
    unique, counts = np.unique(labels, return_counts=True)
    print("labels:", dict(zip(unique.tolist(), counts.tolist())))
    n_phase_z = sampled_X_x.shape[0] + v18_X_x.shape[0]
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
    print(f"\n=== training v17b ===\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
