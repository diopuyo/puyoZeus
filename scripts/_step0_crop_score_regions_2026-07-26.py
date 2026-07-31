"""Step0 検証用: c 系動画のスコア領域を切り出して目視比較用 PNG を出力する。"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2

from src.score_zero import SCORE_1P_REGION, SCORE_2P_REGION

OUT_DIR = "data/verify/step0_winstar_cseries_2026-07-26"
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = {
    "c1": 300,
    "c4": 500,
    "c34": 300,
    "c82": 400,
}


def resize_if_needed(frame):
    if frame.shape[:2] != (1080, 1920):
        return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def main() -> int:
    for vid, t in TARGETS.items():
        path = f"data/frames/video_{vid}.mp4"
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"video_{vid}: OPEN FAILED")
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"video_{vid}: READ FAILED at t={t}")
            continue
        frame = resize_if_needed(frame)
        y1, y2, x1, x2 = SCORE_1P_REGION
        crop_1p = frame[y1:y2, x1:x2]
        y1, y2, x1, x2 = SCORE_2P_REGION
        crop_2p = frame[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(OUT_DIR, f"video_{vid}_score1p_t{t}.png"), crop_1p)
        cv2.imwrite(os.path.join(OUT_DIR, f"video_{vid}_score2p_t{t}.png"), crop_2p)
        print(f"video_{vid}: saved crops at t={t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
