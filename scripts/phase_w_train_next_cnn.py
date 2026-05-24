"""W8-E: NextPairClassifier (32x32 専用 CNN) 訓練。

入力: data/training_phase_u/next_pair_labels.npz
      (W8-D で 19 動画から StableNextDetector で収集、28576 件)
出力: models/next_pair_v1.pt
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

import cv2
import numpy as np

from src.next_pair_classifier import (
    COLOR_TO_CLASS_INDEX,
    NUM_CLASSES,
    NextPairClassifier,
    PATCH_SIZE,
    _augment_next_patch,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/training_phase_u/next_pair_labels.npz",
    )
    parser.add_argument(
        "--out-model", default="models/next_pair_v1.pt",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--holdout-ratio", type=float, default=0.10)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    print(f"loading: {args.input}")
    d = np.load(args.input)
    patches = d["patches"]
    labels = d["labels"]
    videos = d["videos"] if "videos" in d.files else None
    print(f"  shape: {patches.shape}")

    # 動画別 holdout (cross-video 評価): 1 動画分を holdout に
    rng = np.random.default_rng(42)
    if videos is not None:
        unique_videos = np.unique(videos)
        holdout_video = str(unique_videos[-1])  # 最後の動画
        print(f"holdout video: {holdout_video}")
        train_mask = videos != holdout_video
        test_mask = videos == holdout_video
        train_patches = patches[train_mask]
        train_labels = labels[train_mask]
        test_patches = patches[test_mask]
        test_labels = labels[test_mask]
    else:
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

    cls = NextPairClassifier()
    torch = cls._torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls._model.to(device)
    cls._model.train()

    labels_idx = np.array([
        COLOR_TO_CLASS_INDEX[int(c)] for c in train_labels
    ], dtype=np.int64)

    counts = np.bincount(labels_idx, minlength=NUM_CLASSES)
    inv = 1.0 / np.clip(counts, 1, None)
    inv = inv / inv.sum() * NUM_CLASSES
    weight_t = torch.tensor(inv, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_t)
    optimizer = torch.optim.Adam(
        cls._model.parameters(), lr=args.lr, weight_decay=1e-4,
    )

    losses: list[float] = []
    n = len(train_patches)
    augment = not args.no_augment
    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        total_loss = 0.0
        n_batch = 0
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            batch = []
            for i in idx:
                p = train_patches[i]
                if augment:
                    p = _augment_next_patch(p, rng)
                resized = cv2.resize(
                    p, (PATCH_SIZE, PATCH_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
                hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
                combined = np.concatenate([resized, hsv], axis=2)
                batch.append(combined)
            X = np.stack(batch).astype(np.float32) / 255.0
            X_t = torch.from_numpy(X).permute(0, 3, 1, 2).to(device)
            y_t = torch.from_numpy(labels_idx[idx]).to(device)
            optimizer.zero_grad()
            logits = cls._model(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batch += 1
        avg = total_loss / max(1, n_batch)
        losses.append(avg)
        print(f"  epoch {epoch + 1}/{args.epochs}: loss={avg:.4f}")

    cls._model.eval()
    print(f"final loss: {losses[-1]:.4f}")

    # holdout 評価
    correct = 0
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
