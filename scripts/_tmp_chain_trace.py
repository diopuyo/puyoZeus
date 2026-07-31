"""t=142連鎖の1フレーム精密トレース (scratch).

各フレームで 1P の:
  - 掛け算式検知(式ボックス有無)
  - 生OCR score 値
  - 平滑化 score (SideResult.score)
  - 状態
を出力し、「式が消えた後もscoreが上がるか」「式は連続か点滅か」を確定する。
"""
from __future__ import annotations

import sys

import cv2

sys.path.insert(0, ".")

from src.recognition_pipeline import RecognitionPipeline

VIDEO = "data/frames/video_124_4min.mp4"
TARGET_W, TARGET_H = 1920, 1080
T0, T1 = 130.0, 137.0


def main() -> None:
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
    )
    if hasattr(pipe, "set_video_id"):
        pipe.set_video_id("video_124")
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    last1 = None
    print(f"fps={fps:.2f}")
    print("  t     状態          式box  生OCR   平滑score")
    for fi in range(int((T1 + 1) * fps)):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        t = fi / fps
        r = pipe.update(fi, t, frame)
        if r.p1.score is not None:
            last1 = r.p1.score
        if T0 <= t <= T1:
            formula = RecognitionPipeline._check_formula_detected(
                frame, pipe._score_ocr, "1P", last1,
            )
            raw_val, _conf = pipe._score_ocr.read_side(frame, "1P")
            print(
                f"{t:6.2f} {str(r.p1.state):13s} "
                f"{'式あり' if formula else '  -  '} "
                f"{str(raw_val):>7s}  {str(r.p1.score):>8s}"
            )
    cap.release()


if __name__ == "__main__":
    main()
