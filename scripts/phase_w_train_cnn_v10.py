"""W8-C: CNN v10 訓練 (16x16 入力 + ResNet + 強化拡張)。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import numpy as np

from src.patch_classifier import PatchSample
from src.patch_classifier_v2 import CnnPatchClassifierV2, CnnTrainerV2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/training_phase_u/manual_plus_v05_pseudo.npz",
    )
    parser.add_argument(
        "--out-model", default="models/cnn_phase_u_v10.pt",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=2000)
    parser.add_argument("--holdout-ratio", type=float, default=0.10)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    print(f"loading: {args.input}")
    data = np.load(args.input)
    patches = data["patches"]
    labels = data["labels"]
    print(f"  shape: {patches.shape}")

    # クラス均衡化
    rng = np.random.default_rng(42)
    unique = np.unique(labels)
    if args.max_per_class > 0:
        keep = []
        for c in unique:
            idxs = np.where(labels == c)[0]
            if len(idxs) > args.max_per_class:
                idxs = rng.choice(
                    idxs, size=args.max_per_class, replace=False,
                )
            keep.extend(idxs.tolist())
        keep = np.array(sorted(keep))
        patches = patches[keep]
        labels = labels[keep]
        print(f"after balancing: {len(patches)}")

    # train/holdout split
    n = len(patches)
    perm = rng.permutation(n)
    n_holdout = int(n * args.holdout_ratio)
    test_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    train_patches = patches[train_idx]
    train_labels = labels[train_idx]
    test_patches = patches[test_idx]
    test_labels = labels[test_idx]
    print(f"train: {len(train_patches)} / holdout: {len(test_patches)}")

    samples = [
        PatchSample(patch=train_patches[i], color=int(train_labels[i]))
        for i in range(len(train_patches))
    ]

    cls = CnnPatchClassifierV2()
    trainer = CnnTrainerV2(cls, augment=not args.no_augment)
    print(
        f"training v10: epochs={args.epochs} lr={args.lr} "
        f"augment={not args.no_augment}"
    )
    losses = trainer.fit(
        samples, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, class_weighted=True,
    )
    print(f"final loss: {losses[-1]:.4f}")

    # holdout 評価
    correct = 0
    cls._model.eval()
    for i in range(len(test_patches)):
        pred = cls.classify(test_patches[i])
        if pred == int(test_labels[i]):
            correct += 1
    holdout_acc = correct / max(1, len(test_patches))
    print(f"holdout accuracy: {correct}/{len(test_patches)} ({holdout_acc:.3f})")

    out_path = Path(args.out_model)
    cls.save(out_path)
    print(f"saved: {to_windows_path(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
