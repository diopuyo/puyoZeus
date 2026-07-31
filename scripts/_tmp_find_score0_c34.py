"""c34 game1 (472-518付近) の score0 境界を軽量スキャンで特定 (使い捨て・読み取り専用)。"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2

sys.path.insert(0, "/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer")
from src.score_ocr import ScoreOcr  # noqa: E402

VIDEO_PATH = "data/frames/video_c34.mp4"
WINDOW_START = (465.0, 480.0)
WINDOW_END = (510.0, 525.0)
STEP_SEC = 0.2


def scan_window(cap, fps, t0, t1):
    print(f"--- window {t0:.1f}s - {t1:.1f}s ---")
    ocr = ScoreOcr.load_default()
    frame_step = max(1, int(round(STEP_SEC * fps)))
    start_frame = int(t0 * fps)
    end_frame = int(t1 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fi = start_frame
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if (fi - start_frame) % frame_step == 0:
            t = fi / fps
            s1, c1 = ocr.read_side(frame, "1P")
            s2, c2 = ocr.read_side(frame, "2P")
            print(f"t={t:7.2f}s  1P={s1} (conf={c1:.2f})  2P={s2} (conf={c2:.2f})")
        fi += 1


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"fps={fps}")
    scan_window(cap, fps, *WINDOW_START)
    scan_window(cap, fps, *WINDOW_END)
    cap.release()


if __name__ == "__main__":
    main()
