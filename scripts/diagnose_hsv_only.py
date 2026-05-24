"""HSV のみで cell classify (= CNN override 無効化) して結果を出力."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION, BOARD_COLS, HIDDEN_ROWS
from src.recognition_pipeline import RecognitionPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--sec", type=float, required=True)
    p.add_argument("--hsv-state", type=Path, default=None)
    p.add_argument("--cnn-model", type=Path,
                   default=Path("models/cnn_phase_b_finetuned.pt"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipe = RecognitionPipeline.load_default(
        cnn_model_path=args.cnn_model, force_in_match=True,
    )
    if args.hsv_state is not None:
        with args.hsv_state.open() as f:
            state = json.load(f)
        ranges = {int(k): tuple(int(x) for x in v) for k, v in state["per_video_ranges"].items()}
        from src.hybrid_classifier import HybridClassifier
        hc = pipe._reader._classifier
        if isinstance(hc, HybridClassifier) and ranges:
            hc._hsv.set_color_ranges_from_simple(ranges)

    # CNN を無効化 (override_prob を 1.01 = HSV のみ採用)
    hc = pipe._reader._classifier
    if hasattr(hc, "set_cnn_override_prob"):
        hc.set_cnn_override_prob(1.01)
        print("[diag] CNN disabled (override_prob=1.01)")

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pipe._reader.set_resolution_aware_s_min(src_h)
    target_frame = int(args.sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame - 60))
    fr = None
    for fi in range(max(0, target_frame - 60), target_frame + 1):
        ok, fr = cap.read()
        if not ok: break
        if fr.shape[:2] != (1080, 1920):
            fr = cv2.resize(fr, (1920, 1080))
        pipe.update(fi, fi / fps, fr)
    cap.release()
    if fr is None: return

    p1_board, p2_board = pipe._reader.read_both_boards(fr)
    for side, board in [("1P", p1_board), ("2P", p2_board)]:
        cnt = Counter()
        for r in range(HIDDEN_ROWS, 13):
            for c in range(BOARD_COLS):
                cnt[int(board.get(r, c))] += 1
        print(f"{side}: {dict(cnt)}")


if __name__ == "__main__":
    main()
