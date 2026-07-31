"""c62 game9 の score0 境界 (試合開始/終了) を軽量スキャンで特定する (一時スクリプト・使い捨て)。

RecognitionPipeline 全体 (CNN board 認識込み) は重いため、ScoreOcr 単体を
cv2.VideoCapture で直接回して 1P/2P スコアが 0 付近になる瞬間だけを高速に探す。
src/ は無改修 (読み取りのみ)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.score_ocr import ScoreOcr  # noqa: E402

VIDEO_PATH = "data/frames/video_c62.mp4"
# game8 last STABLE sample t=872.0 (score 8805) 〜 game9 first STABLE sample t=877.4 (score 38/36)
WINDOW_START = (868.0, 882.0)
# game9 last STABLE sample t=949.2 〜 game10 first STABLE sample t=953.8 (score 26)
WINDOW_END = (946.0, 958.0)
STEP_SEC = 0.1  # 0.1秒刻み(6fps相当)でスキャン


def scan_window(cap: cv2.VideoCapture, fps: float, t0: float, t1: float) -> None:
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


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"fps={fps}")
    scan_window(cap, fps, *WINDOW_START)
    scan_window(cap, fps, *WINDOW_END)
    cap.release()


if __name__ == "__main__":
    main()
