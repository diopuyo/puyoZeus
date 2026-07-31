"""
Step0 検証用: 各 c 系動画から WIN★ パネル領域のフルカラー crop を数枚出力する。
user 目視レビュー用 (勝敗判定の妥当性確認)。
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2

from src.win_panel import PANEL_Y_RANGE, PANEL_X_RANGE

OUT_DIR = "data/verify/step0_winstar_cseries_2026-07-26"
os.makedirs(OUT_DIR, exist_ok=True)

# (video_id, 確認したい時刻[秒], ラベル) : 各 JSON の game start + offset 付近
TARGETS = [
    ("c82", 130.0, "near_start_panel"),
    ("c82", 2895.0, "near_end_panel"),
    ("c1", 228.0, "near_start_panel"),
    ("c1", 3720.0, "near_end_panel"),
    ("c4", 3858.0, "near_end_panel"),
    ("c34", 2360.0, "near_end_panel"),
]


def resize_if_needed(frame):
    if frame.shape[:2] != (1080, 1920):
        return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def main() -> int:
    for vid, t, label in TARGETS:
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
        y1, y2 = PANEL_Y_RANGE
        x1, x2 = PANEL_X_RANGE
        # 少し余白を広げて視認性を上げる
        y1b, y2b = max(0, y1 - 20), min(1080, y2 + 20)
        x1b, x2b = max(0, x1 - 20), min(1920, x2 + 20)
        crop = frame[y1b:y2b, x1b:x2b]
        out_png = os.path.join(OUT_DIR, f"video_{vid}_t{int(t)}_{label}.png")
        cv2.imwrite(out_png, crop)
        print(f"saved: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
