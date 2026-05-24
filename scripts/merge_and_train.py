"""PL1 + PL2 を統合して最終 CNN 学習"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.calibration import CalibratedConfig
from src.patch_extraction import PatchDataset, balance_dataset
from src.patch_classifier import CnnPatchClassifier, PatchSample
from src.board import (
    COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
    COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA,
)

NAMES = {
    COLOR_EMPTY: "空", COLOR_RED: "赤", COLOR_BLUE: "青", COLOR_GREEN: "緑",
    COLOR_YELLOW: "黄", COLOR_PURPLE: "紫", COLOR_OJAMA: "お邪魔",
}


def latest(pattern: str) -> Path | None:
    files = sorted(Path("data/training").glob(pattern))
    return files[-1] if files else None


def main() -> None:
    pl1 = latest("bulk_patches_balanced_through_v*.npz")
    pl2 = latest("bulk2_patches_balanced_through_v*.npz")
    print(f"PL1: {pl1}")
    print(f"PL2: {pl2}")
    if not pl1 and not pl2:
        print("データなし")
        return

    patches, labels = [], []
    total = 0
    for p in [pl1, pl2]:
        if p is None or not p.exists():
            continue
        ds = PatchDataset.load(p)
        patches.append(ds.patches)
        labels.append(ds.labels)
        total += len(ds.labels)
        print(f"  {p.name}: {len(ds.labels)}")
    print(f"統合前: {total}")

    merged_p = np.concatenate(patches)
    merged_l = np.concatenate(labels)
    ds = PatchDataset(patches=merged_p, labels=merged_l)
    ds.stats.patches_total = len(merged_l)
    unique, counts = np.unique(merged_l, return_counts=True)
    ds.stats.per_class_count = {int(k): int(v) for k, v in zip(unique, counts)}

    balanced = balance_dataset(ds, empty_ratio_cap=0.35)
    balanced.save(Path("data/training/merged_patches_balanced.npz"))
    print(f"\n最終統合 (balanced): {balanced.stats.patches_total}")
    for k in sorted(NAMES.keys()):
        print(f"  {NAMES[k]}: {balanced.stats.per_class_count.get(k, 0)}")

    # CNN学習
    N = len(balanced.labels)
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    to_s = lambda ii: [
        PatchSample(patch=balanced.patches[i], color=int(balanced.labels[i]))
        for i in ii
    ]
    train = to_s(perm[:int(N*0.8)])
    val = to_s(perm[int(N*0.8):int(N*0.9)])
    test = to_s(perm[int(N*0.9):])

    cnn = CnnPatchClassifier()
    start = time.time()
    losses = cnn.fit(train, epochs=30, lr=0.005, batch_size=128)
    print(f"\n学習: {time.time()-start:.1f}s {losses[0]:.3f}→{losses[-1]:.3f}")
    print(f"val={cnn.accuracy(val):.4f} test={cnn.accuracy(test):.4f}")

    y_t = np.array([s.color for s in test])
    y_p = np.array([cnn.classify(s.patch) for s in test])
    for code in sorted(NAMES.keys()):
        m = y_t == code
        if m.sum() == 0:
            continue
        print(f"  {NAMES[code]} (n={m.sum()}): {(y_p[m] == code).mean():.4f}")

    cnn.save(Path("models/cnn_merged_v18.pt"))
    print("保存: models/cnn_merged_v18.pt")


if __name__ == "__main__":
    main()
