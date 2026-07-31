"""案B検証: is_ui() 呼出回数 (751→約12/フレーム相当) を実フレームで実測する
使い捨てスクリプト。src/ は変更せずモンキーパッチのみで計測する。
"""
from __future__ import annotations

import argparse

import cv2

from src.recognition_pipeline import RecognitionPipeline
from src.ui_mask import UI_MASK_TARGET_CELLS, UiMaskMatcher

TARGET_W, TARGET_H = 1920, 1080


def build_pipeline(enable_b: bool) -> RecognitionPipeline:
    ui_mask_cells = UI_MASK_TARGET_CELLS if enable_b else None
    return RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True,
        enable_chain_tracker=True, temporal_smoothing=1,
        force_in_match=True, ui_mask_cells=ui_mask_cells,
    )


def run(video: str, start_sec: float, n_frames: int, enable_b: bool) -> None:
    original_is_ui = UiMaskMatcher.is_ui
    call_count = {"n": 0}

    def patched(self, bgr_patch):  # noqa: ANN001
        call_count["n"] += 1
        return original_is_ui(self, bgr_patch)

    UiMaskMatcher.is_ui = patched
    try:
        pipeline = build_pipeline(enable_b)
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        done = 0
        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (TARGET_H, TARGET_W):
                frame = cv2.resize(frame, (TARGET_W, TARGET_H),
                                   interpolation=cv2.INTER_AREA)
            fi = start_frame + i
            pipeline.update(fi, fi / fps, frame)
            done += 1
        cap.release()
    finally:
        UiMaskMatcher.is_ui = original_is_ui

    label = "A+B" if enable_b else "Aのみ"
    print(f"[{label}] フレーム数={done} is_ui()総呼出={call_count['n']} "
          f"1フレームあたり={call_count['n'] / max(done, 1):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-sec", type=float, default=1451.0)
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()
    run(args.video, args.start_sec, args.frames, enable_b=False)
    run(args.video, args.start_sec, args.frames, enable_b=True)


if __name__ == "__main__":
    main()
