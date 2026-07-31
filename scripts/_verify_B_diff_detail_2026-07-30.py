"""案B: cnn_board (1P, row=4,col=1) の1件差分を詳細確認する使い捨てスクリプト。"""
from __future__ import annotations

import cv2

from src.recognition_pipeline import RecognitionPipeline
from src.ui_mask import UI_MASK_TARGET_CELLS

TARGET_W, TARGET_H = 1920, 1080
VIDEO = "data/frames/video_c60.mp4"
START_SEC = 1490.0
N_FRAMES = 400
TARGET_ROW, TARGET_COL = 4, 1


def build_pipeline(enable_b: bool) -> RecognitionPipeline:
    ui_mask_cells = UI_MASK_TARGET_CELLS if enable_b else None
    return RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True,
        enable_chain_tracker=True, temporal_smoothing=1,
        force_in_match=True, ui_mask_cells=ui_mask_cells,
    )


def run(enable_b: bool):
    pipeline = build_pipeline(enable_b)
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(START_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    values = []
    for i in range(N_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H),
                               interpolation=cv2.INTER_AREA)
        fi = start_frame + i
        result = pipeline.update(fi, fi / fps, frame)
        v = int(result.p1.cnn_board.get(TARGET_ROW, TARGET_COL))
        values.append((fi, fi / fps, v))
    cap.release()
    return values


def main() -> None:
    a = run(enable_b=False)
    b = run(enable_b=True)
    print(f"{'frame':>8} {'t_sec':>8} {'A':>4} {'A+B':>4}  diff?")
    for (fi, t, va), (_, _, vb) in zip(a, b):
        mark = " <-- DIFF" if va != vb else ""
        if mark or True:
            pass
        if va != vb:
            print(f"{fi:>8} {t:>8.2f} {va:>4} {vb:>4}{mark}")
    print("\n(表示なしなら差分ゼロ)")


if __name__ == "__main__":
    main()
