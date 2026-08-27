"""Gate 3R-6: 指定時刻の実ゲーム画面フレームを JPG 保存する (証拠用)。

使い方:
  python scripts/_diag_gate3r6_extract_frames_2026-08-25.py \
      --video data/frames/video_zenchi_c0BQoMJwwQU.mp4 \
      --out-dir data/verify/gate3r6_diag_2026-08-25/evidence_frames \
      --times 18.1 30.0 47.7 164.0 170.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--times", type=float, nargs="+", required=True)
    ap.add_argument("--prefix", default="t")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"動画が開けない: {a.video}")
    for t in a.times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            print(f"[失敗] t={t}")
            continue
        p = a.out_dir / f"{a.prefix}{t:08.2f}s.jpg"
        cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[保存] {p}")
    cap.release()


if __name__ == "__main__":
    main()
