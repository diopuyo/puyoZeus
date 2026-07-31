"""
Step0 検証用: c4 全体を SCAN_INTERVAL=0.5s で再走査し、粗い 2.0s サンプリングが
under-segmentation の原因かどうかを検証する (仮説検証、本実装ではない使い捨てスクリプト)。
"""
from __future__ import annotations

import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2

from src.win_panel import WinPanelDetector
from src.score_zero import ScoreZeroDetector
from scripts.extract_match_winners import detect_match_starts

# 検証用に細かい間隔で上書き (本番定数は変更しない)
import scripts.extract_match_winners as emw
emw.SCAN_INTERVAL_SEC = 0.5


def main() -> int:
    panel_det = WinPanelDetector.load_default()
    zero_det = ScoreZeroDetector.load_default()
    cap = cv2.VideoCapture("data/frames/video_c4.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = total / fps
    t0 = time.time()
    starts = detect_match_starts(cap, dur, panel_det, zero_det)
    elapsed = time.time() - t0
    cap.release()
    print(f"duration={dur:.1f}s  検出 match_starts 件数={len(starts)}  所要={elapsed:.1f}s")
    print(starts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
