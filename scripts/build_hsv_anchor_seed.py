"""HSV-based 高信頼 anchor seed dataset 構築 (Phase I.d)。

CNN mode collapse を回避するため、 HSV ranges の中心領域に確実に入る cell のみを
seed として保存する。SST (Self-training with Self-Adaptive Thresholding,
ICLR 2025) の class-specific threshold アイデアを応用。

設計:
    1. ColorClassifier (HSV-only) で各 cell を分類
    2. cell の HSV 中央値が「色 ranges の中心からの相対距離」 ≤ THRESHOLD なら採用
    3. 各色で独立した閾値 (range 幅で正規化)
    4. fine-tune の anchor (mode collapse 抑制) として使う

使い方:
    python scripts/build_hsv_anchor_seed.py \
        --videos v29 v30 v89 \
        --out-root data/pseudo_labels_hsv_anchor \
        --threshold 0.3 \
        [--limit-per-video 50000]
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.image_reader import ColorClassifier, HsvRange, DEFAULT_COLOR_RANGES
from src.self_supervised.label_store import LabelStore
from src.self_supervised.pseudo_label import (
    COMPONENT_CELL,
    PseudoLabelSample,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos", nargs="+", type=str, required=True)
    p.add_argument("--in-root", type=Path,
                    default=Path("data/pseudo_labels"))
    p.add_argument("--out-root", type=Path,
                    default=Path("data/pseudo_labels_hsv_anchor"))
    p.add_argument("--threshold", type=float, default=0.3,
                    help="HSV 中心からの相対距離閾値 (0-1, 0.3=中心 30%領域)")
    p.add_argument("--limit-per-video", type=int, default=0)
    p.add_argument("--db-root", type=Path, default=None,
                    help="per_video_hsv_ranges/{vid}.json の root。"
                         "指定すると動画別 ranges を使って中心距離を計算。")
    return p.parse_args()


def _ranges_center_dist_normalized(
    h: int, s: int, v: int, ranges: list[HsvRange],
) -> float:
    """HSV 値を最も近い range の中心からの正規化距離で評価.

    Returns:
        最も近い range の正規化距離 (0=中心、1=境界)。 全範囲外なら 1.0+。
    """
    if not ranges:
        return float("inf")
    best = float("inf")
    for r in ranges:
        if not (r.h_min <= h <= r.h_max
                and r.s_min <= s <= r.s_max
                and r.v_min <= v <= r.v_max):
            continue
        # 各軸の正規化距離 (0-1)
        h_center = (r.h_min + r.h_max) / 2
        s_center = (r.s_min + r.s_max) / 2
        v_center = (r.v_min + r.v_max) / 2
        h_half = max(1, (r.h_max - r.h_min) / 2)
        s_half = max(1, (r.s_max - r.s_min) / 2)
        v_half = max(1, (r.v_max - r.v_min) / 2)
        dh = abs(h - h_center) / h_half
        ds = abs(s - s_center) / s_half
        dv = abs(v - v_center) / v_half
        d = max(dh, ds, dv)  # max 軸距離 (= 一番外れている軸)
        if d < best:
            best = d
    return best


def _load_db_ranges(
    db_root: Path | None, video_id: str,
) -> dict[int, list[HsvRange]] | None:
    """per_video_hsv_ranges DB から動画別 ranges を load."""
    if db_root is None:
        return None
    db_path = db_root / f"{video_id}.json"
    if not db_path.is_file():
        return None
    import json as _json
    with db_path.open() as f:
        d = _json.load(f)
    ranges_dict: dict[int, list[HsvRange]] = {}
    for k, v in d.get("per_video_ranges", {}).items():
        try:
            color = int(k)
        except (TypeError, ValueError):
            continue
        ranges_dict[color] = [HsvRange(
            h_min=int(v[0]), h_max=int(v[1]),
            s_min=int(v[2]), s_max=int(v[3]),
            v_min=int(v[4]), v_max=int(v[5]),
        )]
    return ranges_dict


def process_video(
    video_id: str,
    in_root: Path,
    out_root: Path,
    threshold: float,
    limit: int,
    db_root: Path | None = None,
) -> dict:
    in_store = LabelStore(video_id=video_id, root=in_root)
    out_store = LabelStore(video_id=video_id, root=out_root)
    db_ranges = _load_db_ranges(db_root, video_id)
    n_in = 0
    n_out = 0
    by_label: Counter[int] = Counter()
    keep_buf: list[PseudoLabelSample] = []

    def _flush() -> None:
        nonlocal n_out
        if keep_buf:
            out_store.append(keep_buf)
            n_out += len(keep_buf)
            keep_buf.clear()

    for s in in_store.load(COMPONENT_CELL):
        if not isinstance(s.input_data, dict):
            continue
        patch = s.input_data.get("patch")
        if not isinstance(patch, np.ndarray) or patch.size == 0:
            continue
        try:
            color = int(s.label)
        except (TypeError, ValueError):
            continue
        # 該当色の ranges を取得 (DB 注入時は動画別 ranges を優先)
        if db_ranges is not None and color in db_ranges:
            ranges = db_ranges[color]
        else:
            ranges = DEFAULT_COLOR_RANGES.get(color, [])
        if not ranges:
            continue
        # HSV 中央値
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h_med = int(np.median(hsv[:, :, 0]))
        s_med = int(np.median(hsv[:, :, 1]))
        v_med = int(np.median(hsv[:, :, 2]))
        # ranges の中心からの正規化距離 (max 軸)
        dist = _ranges_center_dist_normalized(h_med, s_med, v_med, ranges)
        if dist <= threshold:
            keep_buf.append(s)
            by_label[color] += 1
            if len(keep_buf) >= 256:
                _flush()
        n_in += 1
        if limit and n_in >= limit:
            break
    _flush()
    return {
        "video_id": video_id,
        "n_in": n_in, "n_out": n_out,
        "ratio": n_out / max(1, n_in),
        "by_label": dict(by_label),
    }


def main() -> None:
    args = parse_args()
    print(f"[hsv_anchor] videos={args.videos} threshold={args.threshold}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    overall_in = overall_out = 0
    overall_dist: Counter[int] = Counter()
    t0 = time.time()
    for vid in args.videos:
        t1 = time.time()
        st = process_video(
            vid, args.in_root, args.out_root, args.threshold,
            args.limit_per_video, db_root=args.db_root,
        )
        overall_in += st["n_in"]
        overall_out += st["n_out"]
        for k, v in st["by_label"].items():
            overall_dist[k] += v
        print(
            f"[hsv_anchor] {vid}: in={st['n_in']} out={st['n_out']} "
            f"ratio={st['ratio']:.3f} by_label={st['by_label']} "
            f"t={time.time()-t1:.1f}s",
        )
    print(
        f"[hsv_anchor] DONE total in={overall_in} out={overall_out} "
        f"ratio={overall_out/max(1,overall_in):.3f} "
        f"by_label_total={dict(sorted(overall_dist.items()))} "
        f"elapsed={time.time()-t0:.1f}s",
    )


if __name__ == "__main__":
    main()
