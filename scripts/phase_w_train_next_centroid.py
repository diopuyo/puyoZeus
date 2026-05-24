"""W9-G: NEXT/DNEXT 専用 centroid (5色平均) 分類器。

ユーザー要望: 「ネクスト、ダブルネクストもセル内のピクセルの色の平均化した値を
もとにマッチングさせる」

W8-D で収集した 28576 件 (19動画) のラベル付き next pair patches から
クラスごとの平均色 centroid を構築。

学習:  data/training_phase_u/next_pair_labels.npz (32x32 patches, 5 colors)
出力:  models/next_pair_centroid_v1.npz
評価:  動画別 holdout (cross-video) accuracy
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

from src.centroid_classifier import CentroidClassifier


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/training_phase_u/next_pair_labels.npz",
    )
    parser.add_argument(
        "--out-model",
        default="models/next_pair_centroid_v1.npz",
    )
    parser.add_argument(
        "--out-tsv",
        default="data/verify/phase_w_eval_next_centroid.tsv",
    )
    args = parser.parse_args()

    print(f"loading: {args.input}")
    d = np.load(args.input)
    patches = d["patches"]
    labels = d["labels"]
    videos = d["videos"]
    sides = d["sides"]
    print(f"  shape: {patches.shape}")

    unique_videos = np.unique(videos)
    print(f"  videos: {len(unique_videos)}")

    rows: list[str] = ["holdout_video\tcorrect\ttotal\taccuracy\tn_train"]
    grand = {"correct": 0, "total": 0}
    # leave-one-video-out で実評価
    for vid in unique_videos:
        train_mask = videos != vid
        test_mask = videos == vid
        train_p = patches[train_mask]
        train_l = labels[train_mask]
        test_p = patches[test_mask]
        test_l = labels[test_mask]

        cls = CentroidClassifier()
        cls.fit(list(train_p), list(train_l), normalize=True)

        c = 0
        for i in range(len(test_p)):
            pred = cls.classify(test_p[i])
            if pred == int(test_l[i]):
                c += 1
        t = len(test_p)
        acc = c / max(1, t)
        rows.append(f"{vid}\t{c}\t{t}\t{acc:.4f}\t{len(train_p)}")
        print(f"  {vid}: {c}/{t} = {acc:.4f}  (train={len(train_p)})")
        grand["correct"] += c
        grand["total"] += t

    grand_acc = grand["correct"] / max(1, grand["total"])
    rows.append(f"TOTAL\t{grand['correct']}\t{grand['total']}\t{grand_acc:.4f}\t-")
    print(f"\nTOTAL (LOOV): {grand['correct']}/{grand['total']} = {grand_acc:.4f}")

    # 全データで final centroid を学習・保存
    cls = CentroidClassifier()
    cls.fit(list(patches), list(labels), normalize=True)
    print(f"final centroid counts: {cls.counts}")
    out_model = Path(args.out_model)
    cls.save(out_model)
    print(f"saved model: {to_windows_path(out_model)}")

    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"saved tsv: {to_windows_path(out_tsv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
