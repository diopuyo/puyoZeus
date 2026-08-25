"""117分全編の試合境界時刻を score OCR だけの軽い経路で洗い出す (2026-08-21)。

フル RecognitionPipeline (盤面78セル x2 + 状態機械 + 連鎖追跡) を全編に通すと
実時間の約0.95倍 (≈2時間) かかる見積もり (coordinator算出)。
本スクリプトは **score ROI の OCR だけ** (`src.score_ocr.ScoreOcr`、
盤面色分類より遥かに軽い、memory project_speed_4to26fps 実測 0.7ms/回) を
全フレームに通し、`scripts.visualize_advantage_overlay._detect_score_reset`
(本体の境界判定と同一関数、再実装しない) で境界を検知する。

本体 (visualize_advantage_overlay.py) は変更しない。関数を import して
再利用するだけ。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

cv2.setNumThreads(1)

from scripts.visualize_advantage_overlay import (  # noqa: E402
    GAME_BOUNDARY_DEBOUNCE_SEC,
    _detect_score_reset,
)
from src.score_ocr import ScoreOcr  # noqa: E402

VIDEO = PROJECT_ROOT / "data" / "frames" / "video_zenchi_c0BQoMJwwQU.mp4"
OUT_PATH = PROJECT_ROOT / "data" / "verify" / "zenchi_boundaries_full_2026-08-21.tsv"
NATIVE_W, NATIVE_H = 1920, 1080


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--max-sec", type=float, default=0.0,
                     help="0=全編、テスト用に短縮する場合に指定")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise SystemExit(f"[error] 動画を開けない: {VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000.0)
    frame_limit = (
        int(args.max_sec * fps) if args.max_sec > 0 else n_total
    )
    ocr = ScoreOcr.load_default()

    prev1: int | None = None
    prev2: int | None = None
    last_boundary_t: float | None = None
    boundaries: list[float] = []
    t_wall0 = time.perf_counter()
    fi = 0
    n_ocr_fail = 0
    while fi < frame_limit:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = args.start_sec + fi / fps
        if frame.shape[:2] != (NATIVE_H, NATIVE_W):
            frame = cv2.resize(frame, (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_AREA)
        result = ocr.read(frame)
        s1, s2 = result.score_1p, result.score_2p
        if s1 is None or s2 is None:
            n_ocr_fail += 1
        if _detect_score_reset(s1, s2, prev1, prev2):
            if last_boundary_t is None or t - last_boundary_t >= GAME_BOUNDARY_DEBOUNCE_SEC:
                boundaries.append(t)
                last_boundary_t = t
        if s1 is not None:
            prev1 = s1
        if s2 is not None:
            prev2 = s2
        fi += 1
        if fi % 36000 == 0:  # 10分 (60fps) ごとに進捗
            elapsed = time.perf_counter() - t_wall0
            rate = fi / elapsed if elapsed > 0 else 0.0
            eta = (frame_limit - fi) / rate if rate > 0 else float("nan")
            print(f"  progress: frame={fi}/{frame_limit} t={t:.0f}s "
                  f"境界検知数={len(boundaries)} 壁時間={elapsed:.0f}s "
                  f"ETA残り={eta:.0f}s")
    cap.release()
    wall_total = time.perf_counter() - t_wall0
    video_sec = fi / fps
    print(f"\n[done] frames={fi} 処理動画実時間={video_sec:.1f}s 壁時間={wall_total:.1f}s "
          f"実時間倍率={video_sec/wall_total:.3f}x OCR失敗={n_ocr_fail}件"
          f"({n_ocr_fail/max(1,fi)*100:.2f}%)")
    print(f"境界検知数={len(boundaries)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write("boundary_t_sec\n")
        for b in boundaries:
            f.write(f"{b:.2f}\n")
    print(f"[saved] {args.out}")

    # 隣接間隔からセット境界候補 (最大級ギャップ) を出す
    if len(boundaries) >= 2:
        gaps = [(boundaries[i + 1] - boundaries[i], boundaries[i], boundaries[i + 1])
                for i in range(len(boundaries) - 1)]
        gaps.sort(reverse=True)
        print("\n[セット境界候補 (試合間隔が大きい上位5件)]")
        for gap, t0, t1 in gaps[:5]:
            print(f"  gap={gap:.1f}s  試合終了~次試合開始: {t0:.1f}s -> {t1:.1f}s")


if __name__ == "__main__":
    main()
