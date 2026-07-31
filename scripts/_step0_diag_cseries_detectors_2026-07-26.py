"""
Step0 検証用診断スクリプト。

c 系 4 動画 (c1, c4, c34, c82) について、WinPanelDetector / ScoreZeroDetector の
生スコア (NCC) を複数時刻でサンプリングし、v 系との検出率乖離の原因を切り分ける。
また目視確認用に各動画数枚のフレーム PNG を出力する。
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2

from src.win_panel import WinPanelDetector
from src.score_zero import ScoreZeroDetector

OUT_DIR = "data/verify/step0_winstar_cseries_2026-07-26"
os.makedirs(OUT_DIR, exist_ok=True)

panel_det = WinPanelDetector.load_default()
zero_det = ScoreZeroDetector.load_default()

# 各動画: (video_id, サンプル秒のリスト)
SAMPLES = {
    "c1": [50, 226, 400, 742, 1500, 2016, 3126, 3324, 3600],
    "c4": [50, 130, 500, 1000, 1500, 2000, 2500, 3000, 3500, 3800],
    "c34": [50, 136, 300, 472, 574, 654, 1206, 1780, 2010, 2300],
    "c82": [50, 128, 400, 756, 1000, 1390, 1476, 2000, 2902 - 5],
}


def resize_if_needed(frame):
    if frame.shape[:2] != (1080, 1920):
        return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def main() -> int:
    for vid, times in SAMPLES.items():
        path = f"data/frames/video_{vid}.mp4"
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"video_{vid}: OPEN FAILED")
            continue
        print(f"\n=== video_{vid} ===")
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"  t={t:>6.0f}s  READ FAILED")
                continue
            frame_r = resize_if_needed(frame)
            panel = panel_det.detect(frame_r)
            zero = zero_det.detect(frame_r)
            print(
                f"  t={t:>6.0f}s  panel_score={panel.score:.3f} present={panel.present}  "
                f"zero1p={zero.score_1p:.3f} zero2p={zero.score_2p:.3f} "
                f"both_zero={zero.both_zero}"
            )
            out_png = os.path.join(OUT_DIR, f"video_{vid}_t{int(t):05d}.png")
            cv2.imwrite(out_png, frame_r)
        cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
