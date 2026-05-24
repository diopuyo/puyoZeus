"""Phase Z: CNN v17 fine-tune。v16 init + Phase Z GT 統合。

訓練データ:
    1. v16_dataset.npz (manual + pseudo + review×100 oversample)
    2. phase_z_gt.npz (706 cells × 50倍 oversample = 35,300)

v16 で 99.923% (v18_m03) 達成済 → Phase Z GT (弱点動画含む 18 動画分)
を強学習させて全動画 99.9% 達成を目指す。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_train_cnn_v17
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

# label code → patch_classifier の class index は一致しない (codeは0/1/2/3/4/5/9)
# CnnPatchClassifier 内部の class index は CLASS_INDEX_TO_COLOR で変換
# v16_dataset の labels は color code (0/1/2/3/4/5/9)
# phase_z_gt の y は class index (0/1/2/3/4/5/6)
# → 統合時に v16 と同じ color code に統一する
from src.patch_classifier import CLASS_INDEX_TO_COLOR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init-model", default="models/cnn_phase_u_v16.pt",
    )
    parser.add_argument(
        "--out-model", default="models/cnn_phase_u_v17.pt",
    )
    parser.add_argument(
        "--out-dataset",
        default="data/training_phase_u/v17_dataset.npz",
    )
    parser.add_argument(
        "--phase-z-multiplier", type=int, default=50,
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-per-class", type=int, default=20000)
    args = parser.parse_args()

    # === v16_dataset.npz 読み込み ===
    v16_path = Path("data/training_phase_u/v16_dataset.npz")
    if v16_path.exists():
        d = np.load(v16_path)
        v16_patches = d["patches"]
        v16_labels = d["labels"].astype(np.int32)
        print(f"v16_dataset: {v16_patches.shape} ({v16_labels.size} samples)")
    else:
        print(f"WARN: {v16_path} not found, creating from sources")
        # v16_dataset がなければ phase_w_train_cnn_v16 と同じソースから組み立て
        sources = [
            ("manual_plus_strict",
             "data/training_phase_u/manual_plus_strict.npz"),
            ("pseudo_v7_all19",
             "data/training_phase_u/pseudo_v7_all19.npz"),
        ]
        parts_p = []
        parts_l = []
        for name, path in sources:
            d = np.load(path)
            parts_p.append(d["patches"])
            parts_l.append(d["labels"].astype(np.int32))
            print(f"  {name}: {d['patches'].shape}")
        v16_patches = np.concatenate(parts_p, axis=0)
        v16_labels = np.concatenate(parts_l, axis=0)

    # === phase_z_gt.npz 読み込み ===
    pz_path = Path("data/training_phase_u/phase_z_gt.npz")
    if not pz_path.exists():
        print(f"ERROR: {pz_path} not found")
        return 1
    pz = np.load(pz_path)
    pz_patches = pz["X"]  # (706, 8, 8, 3)
    pz_class_idx = pz["y"]  # class index (0..6)
    # class index → color code に変換
    idx_to_code = np.array(CLASS_INDEX_TO_COLOR, dtype=np.int32)
    pz_labels = idx_to_code[pz_class_idx]
    print(f"phase_z_gt: {pz_patches.shape} ({pz_labels.size} samples)")

    # === oversample phase_z_gt ===
    mult = args.phase_z_multiplier
    pz_patches_x = np.tile(pz_patches, (mult, 1, 1, 1))
    pz_labels_x = np.tile(pz_labels, mult)
    print(f"phase_z oversampled x{mult}: {pz_patches_x.shape}")

    # === 統合 ===
    # v16_patches は 16x16、phase_z_gt は 8x8 → patch_to_feature が 8x8 化するので OK
    # ただし shape が異なるので concat できない
    # phase_z_gt を 16x16 に resize して合わせる
    if pz_patches_x.shape[1:3] != v16_patches.shape[1:3]:
        import cv2
        target_h, target_w = v16_patches.shape[1:3]
        pz_resized = np.array([
            cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_AREA)
            for p in pz_patches_x
        ], dtype=np.uint8)
        pz_patches_x = pz_resized
        print(f"phase_z resized to {target_h}x{target_w}: {pz_patches_x.shape}")

    patches = np.concatenate([v16_patches, pz_patches_x], axis=0)
    labels = np.concatenate([v16_labels, pz_labels_x], axis=0)
    print(f"total: {patches.shape}")
    unique, counts = np.unique(labels, return_counts=True)
    print("labels:", dict(zip(unique.tolist(), counts.tolist())))

    out_ds = Path(args.out_dataset)
    out_ds.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_ds, patches=patches, labels=labels)
    print(f"saved dataset: {to_windows_path(out_ds)}")

    # === phase_u_train_cnn 呼び出し ===
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
    print(f"\n=== training v17 ===\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
