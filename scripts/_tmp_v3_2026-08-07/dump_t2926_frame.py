"""t=2926s 付近フレームを切り出して1P盤面全体crop+個別セルpatchを保存 (診断用)。"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.image_reader import DEFAULT_P1_REGION

VIDEO_PATH = _ROOT / "data/frames/video_olRyxDGacbg.mp4"
T_SEC = 2926.0
TARGET_SIZE = (1920, 1080)
OUT_DIR = _ROOT / "data/verify/youtube_demo_2026-08-07"


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    cap.set(cv2.CAP_PROP_POS_MSEC, T_SEC * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("read failed")
    if frame.shape[1::-1] != TARGET_SIZE:
        frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    region = DEFAULT_P1_REGION
    x1, y1, x2, y2 = region.x, region.y, region.x + region.width, region.y + region.height
    crop = frame[y1:y2, x1:x2]
    crop_big = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT_DIR / "_t2926_1p_board_crop.png"), crop_big)
    print("saved crop", crop_big.shape)


if __name__ == "__main__":
    main()
