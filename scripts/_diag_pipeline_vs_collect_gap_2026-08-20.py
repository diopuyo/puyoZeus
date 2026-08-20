"""産線 24-25fps と診断 65fps 相当の乖離を帰属させる (2026-08-20)。

これが高速化の投資先を決める分岐点になる。

判明している矛盾:
  - 内訳診断 (静穏区間300frame) では 1 frame 15.3ms = **65fps 相当**
  - しかし実収集 (単独実行) は 39番 62分/94,380frame = **25.4fps**、
    38番 69分/198,000frame = 23.9fps (認識は stride-2 なので実効)
  → 2.5 倍の乖離。この正体が分からないまま Rust 化に投資すると的を外す。

容疑者 (fable の整理):
  (i)  重い区間での classify 爆発。`classify` は中央 median が EMPTY のとき
       4 サブパッチで再判定するため空セルで約5倍のコストになる
       (src/image_reader.py:558-580)。連鎖・演出中は空セルが増える
       → もしこれが支配なら HSV の Rust 化が直撃する
  (ii) デコード (RT では OBS キャプチャ側の別勘定になる)
  (iii) collect 側の処理 (npz 蓄積・指標計算・状態管理。RT には載らない)

そこで collect の最上位ループと同じ構造を再現し、1 frame を
  decode / pipeline.update / collect側 の3つに分けて実測する。
さらに **静穏区間と連鎖区間を層別**する (memory: 代表値を出す前に層別せよ)。
連鎖の有無は pipeline の返す状態から判定するので、事前に区間を知る必要がない。
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

cv2.setNumThreads(1)

from src.production_config import collect_flags  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402


def _pipeline_kwargs() -> dict:
    """collect_flags() が渡すフラグのうち RecognitionPipeline が受けるものを再現。

    完全一致は難しいので、load_default に既定で入るものはそのまま使い、
    「収集時と診断時で構成が違うために速度が違う」可能性を排除するために
    どのフラグが渡っているかをログに出す (fail-silent 警戒)。
    """
    flags = collect_flags().split()
    print(f"[config] collect_flags のフラグ数: {len([f for f in flags if f.startswith('--')])}")
    return {}


def _classify_calls_counter() -> dict:
    """classify の呼び出し回数を数えるカウンタを仕込む (層別のため)。"""
    from src import image_reader as ir

    state = {"n": 0, "orig": ir.ColorClassifier.classify}

    def counted(self, patch):  # noqa: ANN001, ANN202
        state["n"] += 1
        return state["orig"](self, patch)

    ir.ColorClassifier.classify = counted
    return state


def main() -> None:
    """1 frame を decode / update / collect側 に分けて層別計測する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=Path("data/frames/video_c34.mp4"))
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=4000)
    args = ap.parse_args()

    _pipeline_kwargs()
    pipe = RecognitionPipeline.load_default()
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[error] 動画を開けない: {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000.0)

    cnt = _classify_calls_counter()
    rows: list[tuple[float, float, int]] = []  # (decode_ms, update_ms, classify回数)
    try:
        for i in range(args.frames):
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            t1 = time.perf_counter()
            before = cnt["n"]
            pipe.update(i, args.start_sec + i / fps, frame)
            t2 = time.perf_counter()
            rows.append(((t1 - t0) * 1000.0, (t2 - t1) * 1000.0, cnt["n"] - before))
    finally:
        from src import image_reader as ir
        ir.ColorClassifier.classify = cnt["orig"]
        cap.release()

    if not rows:
        print("[error] フレームを読めなかった")
        return
    rows = rows[10:] or rows  # 初期化コストを除く

    dec = [r[0] for r in rows]
    upd = [r[1] for r in rows]
    cls = [r[2] for r in rows]

    print(f"\n=== {args.video.name} {args.start_sec:.0f}s から {len(rows)} frame ===")
    print(f"{'区分':<22} {'中央値':>9} {'平均':>9} {'p90':>9} {'合計比':>8}")
    print("-" * 62)
    tot = sum(dec) + sum(upd)
    for label, xs in (("decode", dec), ("pipeline.update", upd)):
        xs_s = sorted(xs)
        p90 = xs_s[int(len(xs_s) * 0.9)]
        print(f"{label:<22} {statistics.median(xs):8.2f}ms {statistics.mean(xs):8.2f}ms "
              f"{p90:8.2f}ms {sum(xs)/tot*100:7.1f}%")
    print("-" * 62)
    total_med = statistics.median([d + u for d, u in zip(dec, upd)])
    total_mean = statistics.mean([d + u for d, u in zip(dec, upd)])
    print(f"{'1 frame 合計':<22} {total_med:8.2f}ms {total_mean:8.2f}ms")
    print(f"  → 中央値ベース {1000/max(total_med,1e-9):.1f} fps"
          f" / 平均ベース {1000/max(total_mean,1e-9):.1f} fps")
    print()
    print("  ※ collect 側 (npz蓄積・指標・状態管理) は本計測に含まれない。")
    print("     産線 fps との残差がそれに相当する。")

    # 層別: classify 呼び出し回数で「重い frame」と「軽い frame」を分ける
    print(f"\n=== classify 呼び出し回数による層別 (乖離の主因の切り分け) ===")
    med_cls = statistics.median(cls)
    print(f"classify 回数: 中央値 {med_cls:.0f} / 平均 {statistics.mean(cls):.1f} "
          f"/ 最大 {max(cls)} / 最小 {min(cls)}")
    bands = [(0, 1), (1, 50), (50, 120), (120, 200), (200, 10**9)]
    print(f"{'classify回数':<16} {'frame数':>8} {'update中央':>11} {'占有率':>8}")
    print("-" * 48)
    for lo, hi in bands:
        sel = [(u, c) for u, c in zip(upd, cls) if lo <= c < hi]
        if not sel:
            continue
        us = [u for u, _ in sel]
        share = sum(us) / sum(upd) * 100
        name = f"{lo}〜{hi-1}" if hi < 10**9 else f"{lo}以上"
        print(f"{name:<16} {len(sel):8d} {statistics.median(us):10.2f}ms {share:7.1f}%")
    print("-" * 48)
    print("  → 高回数帯が update 時間の大半を占めるなら、容疑(i)「空セルの")
    print("     サブパッチ再判定による classify 爆発」が支配的で、HSV の Rust 化が直撃する。")


if __name__ == "__main__":
    main()
