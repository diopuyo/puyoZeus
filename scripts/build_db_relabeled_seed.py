"""DB ranges で各動画の pseudo を再分類して高純度 seed を作成 (v8 用 Phase I.d)。

各動画の pseudo label store の patch を ColorClassifier (DB 注入版) で再分類し、
HSV-only 判定が確定した cell のみ seed として保存。

これは pseudo の transient label (= settle 多数決) ではなく、 動画別最適化済
HSV ranges による直接判定なので mode collapse とは独立。

使い方:
    python scripts/build_db_relabeled_seed.py \
        --videos v29 v40 v89 \
        --out-root data/pseudo_labels_db_relabeled
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.image_reader import ColorClassifier
from src.self_supervised.label_store import LabelStore
from src.self_supervised.pseudo_label import (
    COMPONENT_CELL,
    PseudoLabelSample,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos", nargs="+", required=True)
    p.add_argument("--db-root", type=Path,
                    default=Path("data/per_video_hsv_ranges"))
    p.add_argument("--in-root", type=Path,
                    default=Path("data/pseudo_labels"))
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--limit-per-video", type=int, default=100000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    overall = Counter()
    for vid in args.videos:
        db_path = args.db_root / f"{vid}.json"
        if not db_path.is_file():
            print(f"[seed] {vid}: SKIP (no DB at {db_path})")
            continue
        with db_path.open() as f:
            d = json.load(f)
        ranges = {
            int(k): tuple(int(x) for x in v)
            for k, v in d["per_video_ranges"].items()
        }
        # 動画別 ranges を ColorClassifier に注入
        cc = ColorClassifier()
        cc.set_color_ranges_from_simple(ranges)
        # pseudo を再分類して高純度 seed 構築
        in_store = LabelStore(video_id=vid, root=args.in_root)
        out_store = LabelStore(video_id=vid, root=args.out_root)
        n_in = n_out = 0
        by_label: Counter[int] = Counter()
        keep: list[PseudoLabelSample] = []
        t0 = time.time()
        for s in in_store.load(COMPONENT_CELL):
            if not isinstance(s.input_data, dict):
                continue
            patch = s.input_data.get("patch")
            if not isinstance(patch, np.ndarray) or patch.size == 0:
                continue
            try:
                pseudo_label = int(s.label)
            except (TypeError, ValueError):
                continue
            # DB 注入済 ColorClassifier で再分類
            db_pred = int(cc.classify(patch))
            # pseudo label と DB 判定が一致した sample のみ seed として採用
            if db_pred == pseudo_label:
                keep.append(s)
                by_label[db_pred] += 1
                if len(keep) >= 256:
                    out_store.append(keep)
                    n_out += len(keep)
                    keep.clear()
            n_in += 1
            if args.limit_per_video and n_in >= args.limit_per_video:
                break
        if keep:
            out_store.append(keep)
            n_out += len(keep)
        for k, v in by_label.items():
            overall[k] += v
        print(
            f"[seed] {vid}: in={n_in} out={n_out} "
            f"ratio={n_out/max(1,n_in):.3f} by_label={dict(by_label)} "
            f"t={time.time()-t0:.1f}s",
        )
    print(f"[seed] DONE total by_label={dict(sorted(overall.items()))}")


if __name__ == "__main__":
    main()
