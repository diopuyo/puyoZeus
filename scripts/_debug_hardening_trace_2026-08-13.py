"""案4-lite hardening が実際に発火するか frame 単位で追跡するデバッグ専用スクリプト.

一時的な調査用 (read-only、src/ は変更しない)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_visual_detector import _count_top_ojama  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

PROD_TARGET_W, PROD_TARGET_H = 1920, 1080


def _resize(frame):
    h, w = frame.shape[:2]
    if (h, w) == (PROD_TARGET_H, PROD_TARGET_W):
        return frame
    interp = cv2.INTER_LANCZOS4 if h < PROD_TARGET_H else cv2.INTER_AREA
    return cv2.resize(frame, (PROD_TARGET_W, PROD_TARGET_H), interpolation=interp)


def main() -> int:
    video = Path("data/frames/review_demo_2026-08-12.mp4")
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    decode_from_sec = 170.0
    print_from_sec = 198.0
    print_to_sec = 199.0
    decode_from_frame = int(decode_from_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, decode_from_frame)

    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
        enable_effect_gate=True, enable_burst_guard_v2=True,
        burst_gate_open_threshold=0.954, enable_hidden_row_burst_guard=True,
        enable_transition_merge_guard=True, enable_match_transition_debounce=True,
        enable_ojama_fall_placement_override=True,
        enable_ojama_fall_entry_hardening=True,
        enable_chain_gate_raw_fallback=True,
    )
    ovd = None
    for det in pipe._sm_1p._detectors:
        if type(det).__name__ == "OjamaVisualDetector":
            ovd = det
    assert ovd is not None

    stride = 2
    frame_idx = decode_from_frame
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - decode_from_frame) % stride != 0:
            frame_idx += 1
            continue
        t_sec = frame_idx / fps
        frame_in = _resize(frame)
        prev_state = pipe._sm_1p.context.state
        r = pipe.update(frame_idx, t_sec, frame_in.copy())
        if print_from_sec <= t_sec <= print_to_sec:
            cur_count = _count_top_ojama(r.p1.cnn_board, None)
            print(
                f"t={t_sec:.3f} prev_state={prev_state.value:14s} "
                f"new_state={r.p1.state.value:14s} roi_count={cur_count} "
                f"entry_trigger_start={ovd._entry_trigger_start_time:.3f} "
                f"prev_top={ovd._prev_top_ojama_count} "
                f"placement_exit_t={ovd._placement_override_exit_time:.3f}"
            )
        if t_sec > print_to_sec:
            break
        frame_idx += 1
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
