"""指摘16調査: 元動画から絶対時刻帯のフレームを抽出して保存する (計装専用、本番コード不変更)。

対象: data/frames/review_demo_2026-08-12.mp4 (60fps, 1280x720)
window: 絶対時刻 t_start〜t_end を step秒刻みでPNG保存。
出力: data/verify/diag_issue16_2026-08-15/raw_frames/frame_t{sec:.2f}.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT_DIR = Path("data/verify/diag_issue16_2026-08-15/raw_frames")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=VIDEO)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--start-sec", type=float, default=282.0)
    ap.add_argument("--end-sec", type=float, default=292.0)
    ap.add_argument("--step-sec", type=float, default=0.5)
    ap.add_argument("--prefix", type=str, default="frame")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"video fps={fps}")

    t = args.start_sec
    saved = []
    while t <= args.end_sec + 1e-9:
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"read failed at t={t:.2f} frame_idx={frame_idx}")
            t += args.step_sec
            continue
        out_path = out_dir / f"{args.prefix}_t{t:06.2f}_idx{frame_idx}.png"
        cv2.imwrite(str(out_path), frame)
        saved.append(str(out_path))
        t += args.step_sec
    cap.release()
    print(f"saved {len(saved)} frames")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
