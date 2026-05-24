"""W8-E: CNN v11 訓練。

v10 (16x16 + ResNet + aug) のアーキテクチャを v7 の大規模 multi-video データで再訓練。

v10 の問題: cross-video の violations_50_bg で v7 (avg 96.4%) より劣る (avg 90.2%)。
v10 は v05 を中心にしたデータ (13K) で訓練されており多様性に欠ける。

v11 = manual_plus_strict (451K, multi-video pl1-pl4) + manual_plus_v05_pseudo (13K)
      を結合し、各クラス 5000 件まで均衡化、v10 アーキで訓練。
"""
from __future__ import annotations

import argparse
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


def load_and_combine(
    inputs: list[str], max_per_class: int, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_patches: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for inp in inputs:
        d = np.load(inp)
        all_patches.append(d["patches"])
        all_labels.append(d["labels"])
        print(f"  loaded {inp}: {d['patches'].shape}")
    patches = np.concatenate(all_patches, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"combined: {patches.shape}")

    # クラス均衡化
    if max_per_class > 0:
        unique = np.unique(labels)
        keep_idx = []
        for c in unique:
            idxs = np.where(labels == c)[0]
            if len(idxs) > max_per_class:
                idxs = rng.choice(idxs, size=max_per_class, replace=False)
            keep_idx.extend(idxs.tolist())
        keep_idx = np.array(sorted(keep_idx))
        patches = patches[keep_idx]
        labels = labels[keep_idx]
        print(f"after balancing (<= {max_per_class}/class): {patches.shape}")

    return patches, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs", nargs="+", default=[
            "data/training_phase_u/manual_plus_strict.npz",
            "data/training_phase_u/manual_plus_v05_pseudo.npz",
        ],
    )
    parser.add_argument(
        "--out-model", default="models/cnn_phase_u_v11.pt",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-per-class", type=int, default=5000)
    parser.add_argument("--holdout-ratio", type=float, default=0.10)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    patches, labels = load_and_combine(
        args.inputs, args.max_per_class,
    )

    rng = np.random.default_rng(42)
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
        f"training v11: epochs={args.epochs} lr={args.lr} "
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
