"""案B差分1件 (video_c60 frame=45038, 1P row=4,col=1) の実画面を保存する。"""
from __future__ import annotations

import cv2

from src.image_reader import DEFAULT_P1_REGION

TARGET_W, TARGET_H = 1920, 1080
VIDEO = "data/frames/video_c60.mp4"
TARGET_FRAME = 45038
ROW, COL = 4, 1
OUT_FULL = "data/verify/ui_mask_fire_2026-07-30/diff_frame_45038_full.png"
OUT_CELL = "data/verify/ui_mask_fire_2026-07-30/diff_frame_45038_cell_context.png"


def main() -> None:
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("フレーム読み込み失敗")
        return
    if frame.shape[:2] != (TARGET_H, TARGET_W):
        frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    cv2.imwrite(OUT_FULL, frame)

    x1, y1, x2, y2 = DEFAULT_P1_REGION.cell_sample_rect(ROW, COL)
    margin = 100
    cx1 = max(0, int(x1) - margin)
    cy1 = max(0, int(y1) - margin)
    cx2 = min(TARGET_W, int(x2) + margin)
    cy2 = min(TARGET_H, int(y2) + margin)
    crop = frame[cy1:cy2, cx1:cx2].copy()
    box_pt1 = (int(x1) - cx1, int(y1) - cy1)
    box_pt2 = (int(x2) - cx1, int(y2) - cy1)
    cv2.rectangle(crop, box_pt1, box_pt2, (0, 0, 255), 2)
    cv2.imwrite(OUT_CELL, crop)
    print(f"保存: {OUT_FULL}")
    print(f"保存: {OUT_CELL}")


if __name__ == "__main__":
    main()
