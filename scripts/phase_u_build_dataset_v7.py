"""Phase U V1.4': multi-source データセット (manual + relabeled parallel)。

Manual (高品質、video_01) と CNN v6 で再ラベル + 高信頼度フィルタした
parallel (中品質、多動画) を混合して CNN v7 訓練用 npz を生成する。

ミックス戦略:
    - manual_labels_aug20.npz (約 9.5万件、16x16) をベースに採用 (高重み)
    - parallel_relabeled/*.npz から各 npz n 件サンプル → 16x16 にリサイズ
    - 結合 → data/training_phase_u/manual_plus_relabeled.npz

WSL2 OOM 回避のため、巨大 concat 前にサブサンプル (memory feedback 通り)。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_u_build_dataset_v7 \
        --per-npz 2500
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

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2
import numpy as np

PATCH_SIZE = 16


def resize_to_target(patches: np.ndarray) -> np.ndarray:
    """N x H x W x 3 を N x PATCH_SIZE x PATCH_SIZE x 3 にリサイズ。"""
    N, H, W, C = patches.shape
    if (H, W) == (PATCH_SIZE, PATCH_SIZE):
        return patches
    out = np.zeros((N, PATCH_SIZE, PATCH_SIZE, C), dtype=patches.dtype)
    for i in range(N):
        out[i] = cv2.resize(
            patches[i], (PATCH_SIZE, PATCH_SIZE),
            interpolation=cv2.INTER_AREA,
        )
    return out


def sample_npz(
    path: Path, n: int, rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """1 npz から n 件 (or 全件) ランダムサンプル。"""
    d = np.load(path)
    patches = d["patches"]
    labels = d["labels"]
    N = patches.shape[0]
    if N <= n:
        return patches, labels
    idx = rng.sample(range(N), n)
    idx_arr = np.array(sorted(idx), dtype=np.int64)
    return patches[idx_arr], labels[idx_arr]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual",
        default="data/training_phase_u/manual_labels_aug20.npz",
    )
    parser.add_argument(
        "--parallel-dir",
        default="data/training_phase_u/parallel_relabeled",
    )
    parser.add_argument(
        "--out",
        default="data/training_phase_u/manual_plus_relabeled.npz",
    )
    parser.add_argument(
        "--per-npz", type=int, default=2500,
        help="parallel/ の各 npz から取り出す件数 (119 * 2500 = 約30万)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1. manual ロード
    print(f"manual: {args.manual}")
    md = np.load(args.manual)
    manual_patches = md["patches"]
    manual_labels = md["labels"].astype(np.int64)
    print(f"  shape: {manual_patches.shape}")
    manual_patches = resize_to_target(manual_patches)
    print(f"  resized: {manual_patches.shape}")

    # 2. parallel_relabeled をストリーミング結合
    parallel_dir = Path(args.parallel_dir)
    npz_files = sorted(p for p in parallel_dir.glob("*.npz") if p.is_file())
    print(f"parallel: {len(npz_files)} npz files, per-npz={args.per_npz}")
    p_patches_list: list[np.ndarray] = []
    p_labels_list: list[np.ndarray] = []
    for i, npz_path in enumerate(npz_files, 1):
        ps, ls = sample_npz(npz_path, args.per_npz, rng)
        ps = resize_to_target(ps)
        p_patches_list.append(ps)
        p_labels_list.append(ls.astype(np.int64))
        if i % 20 == 0:
            print(f"  [{i}/{len(npz_files)}] sampled")

    parallel_patches = np.concatenate(p_patches_list)
    parallel_labels = np.concatenate(p_labels_list)
    del p_patches_list, p_labels_list
    print(f"  parallel total: {parallel_patches.shape}")

    # 3. 結合
    all_patches = np.concatenate([manual_patches, parallel_patches])
    all_labels = np.concatenate([manual_labels, parallel_labels])
    del manual_patches, manual_labels, parallel_patches, parallel_labels

    print(f"\nfinal: {all_patches.shape}, {all_labels.shape}")
    unique, counts = np.unique(all_labels, return_counts=True)
    for c, n in zip(unique, counts):
        print(f"  label={c}: {n}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        patches=all_patches.astype(np.uint8),
        labels=all_labels.astype(np.int32),
    )
    print(f"\nsaved: {to_windows_path(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
