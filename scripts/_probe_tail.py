"""動画末尾の状態を 1 秒刻みでダンプする。"""
from __future__ import annotations
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
from src.score_zero import ScoreZeroDetector
from src.win_panel import WinPanelDetector

zero_det = ScoreZeroDetector.load_default()
panel_det = WinPanelDetector.load_default()

cap = cv2.VideoCapture("data/frames/video_02.mp4")
for t in range(2900, 3060):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        continue
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080))
    panel = panel_det.detect(frame)
    if panel.present:
        z = zero_det.detect(frame)
        state = "zero" if z.both_zero else "playing"
        score1 = z.score_1p
        score2 = z.score_2p
    else:
        state = "none"
        score1 = score2 = 0.0
    print(f"t={t:4d}  state={state:7s}  panel={panel.score:.3f}  zero_1p={score1:.3f}  zero_2p={score2:.3f}")
cap.release()
