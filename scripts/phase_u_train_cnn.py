"""手動ラベルで CnnPatchClassifier を fine-tune する (Phase U-3)。

入力: scripts/phase_u_build_dataset.py が出力する npz
出力: PyTorch state_dict (.pt)
初期化: cnn_global_best.pt があればロード、なければランダム初期化。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.console_init import init_console  # noqa: E402
init_console()

import numpy as np

from src.patch_classifier import CnnPatchClassifier, PatchSample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training_phase_u/manual_labels.npz")
    parser.add_argument("--init-model", default="models/cnn_global_best.pt")
    parser.add_argument("--out-model", default="models/cnn_phase_u_v1.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=500,
                        help="クラス均衡化、各色最大サンプル数")
    parser.add_argument("--holdout-ratio", type=float, default=0.10,
                        help="hold-out 検証セット比率")
    args = parser.parse_args()

    data = np.load(args.input)
    patches = data["patches"]
    labels = data["labels"]
    print(f"loaded: {patches.shape}")
    unique, counts = np.unique(labels, return_counts=True)
    print("label distribution:")
    for c, n in zip(unique, counts):
        print(f"  code={c}: {n}")

    # クラス均衡化
    rng = np.random.default_rng(42)
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

    classifier = CnnPatchClassifier()
    if args.init_model and Path(args.init_model).exists():
        try:
            import torch
            state = torch.load(
                args.init_model, map_location="cpu", weights_only=True,
            )
            classifier._model.load_state_dict(state)
            print(f"loaded init: {args.init_model}")
        except Exception as e:
            print(f"failed to load init ({e}), random initialization")

    print(f"training: epochs={args.epochs} lr={args.lr} batch={args.batch_size}")
    losses = classifier.fit(
        samples, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, class_weighted=True,
    )
    print(f"final loss: {losses[-1]:.4f}")

    # holdout 評価
    correct = 0
    classifier._model.eval()
    for i in range(len(test_patches)):
        pred = classifier.classify(test_patches[i])
        if pred == int(test_labels[i]):
            correct += 1
    holdout_acc = correct / max(1, len(test_patches))
    print(f"holdout accuracy: {correct}/{len(test_patches)} ({holdout_acc:.3f})")

    # 保存
    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(classifier._model.state_dict(), str(out_path))
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
