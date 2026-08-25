"""ラベルsheetのアンカーframeそのものを抽出する簡易ツール (計装、本体コード非変更)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_extract_anchor_frame_2026-08-17 \
        --video c22 --frame 108676 --side 1P --r 12 --c 2 --tag c22_anchor
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402

VIDEO_DIR = Path.home() / "frames"
OUT_DIR = _ROOT / "data" / "verify" / "diag_c13c22_recheck_2026-08-17" / "frames_extra"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--side", required=True, choices=["1P", "2P"])
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--c", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--deltas", default="-3,-2,-1,0,1,2,3")
    args = ap.parse_args()

    video_path = VIDEO_DIR / f"video_{args.video}.mp4"
    region = DEFAULT_P1_REGION if args.side == "1P" else DEFAULT_P2_REGION
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    for delta in [int(x) for x in args.deltas.split(",")]:
        f_idx = args.frame + delta
        if f_idx < 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"read failed at {f_idx}")
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        x1, y1, x2, y2 = region.cell_sample_rect(args.r, args.c)
        vis = frame.copy()
        cv2.rectangle(vis, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 255), 2)
        cv2.rectangle(
            vis, (region.x, region.y),
            (region.x + region.width, region.y + region.height), (0, 255, 255), 1,
        )
        fname = f"{args.tag}_f{f_idx}_d{delta:+d}.png"
        cv2.imwrite(str(OUT_DIR / fname), vis)
        print(f"saved {fname}")
    cap.release()


if __name__ == "__main__":
    main()
