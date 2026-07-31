"""Step0 検証用: c4 の先頭 600 秒を 1 秒刻みで走査し、zero 状態の持続時間を調べる。"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2

from src.score_zero import ScoreZeroDetector
from src.win_panel import WinPanelDetector

panel_det = WinPanelDetector.load_default()
zero_det = ScoreZeroDetector.load_default()


def resize_if_needed(frame):
    if frame.shape[:2] != (1080, 1920):
        return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def main() -> int:
    cap = cv2.VideoCapture("data/frames/video_c4.mp4")
    for t in range(0, 700, 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = resize_if_needed(frame)
        panel = panel_det.detect(frame)
        zero = zero_det.detect(frame)
        state = "playing" if panel.present and not zero.both_zero else (
            "zero" if panel.present and zero.both_zero else "none"
        )
        print(f"t={t:>4d}s state={state:8s} panel={panel.score:.3f} z1={zero.score_1p:.3f} z2={zero.score_2p:.3f}")
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
