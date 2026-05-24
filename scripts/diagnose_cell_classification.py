"""指定 frame で cell ごとの classify 結果と patch HSV を出力 (診断用).

使い方:
    python scripts/diagnose_cell_classification.py \
        --video data/test_unknown/unknown_match2_75s.mp4 \
        --sec 20 \
        --hsv-state data/per_video_hsv_ranges/_merged_default.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.image_reader import (
    BOARD_COLS,
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
    HIDDEN_ROWS,
)
from src.recognition_pipeline import RecognitionPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--sec", type=float, required=True)
    p.add_argument("--hsv-state", type=Path, default=None)
    p.add_argument("--cnn-model", type=Path,
                   default=Path("models/cnn_phase_b_finetuned.pt"))
    p.add_argument("--out-png", type=Path, default=None)
    return p.parse_args()


COLOR_NAMES = {
    0: "EMPTY", 1: "RED", 2: "BLUE", 3: "GREEN",
    4: "YELLOW", 5: "PURPLE", 9: "OJAMA", 10: "UNKNOWN",
}


def main() -> None:
    args = parse_args()
    pipe = RecognitionPipeline.load_default(
        cnn_model_path=args.cnn_model, force_in_match=True,
    )
    if args.hsv_state is not None:
        with args.hsv_state.open() as f:
            state = json.load(f)
        ranges = {
            int(k): tuple(int(x) for x in v)
            for k, v in state["per_video_ranges"].items()
        }
        from src.hybrid_classifier import HybridClassifier
        hc = pipe._reader._classifier
        if isinstance(hc, HybridClassifier) and ranges:
            hc._hsv.set_color_ranges_from_simple(ranges)


    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if hasattr(pipe._reader, "set_resolution_aware_s_min"):
        pipe._reader.set_resolution_aware_s_min(src_height)
        print(f"[diag] resolution-aware applied for source height={src_height}")
    target_frame = int(args.sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame - 60))
    for fi in range(max(0, target_frame - 60), target_frame + 1):
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[:2] != (1080, 1920):
            fr = cv2.resize(fr, (1920, 1080))
        result = pipe.update(fi, fi / fps, fr)
    cap.release()

    if fr is None:
        print("frame read failed")
        return

    # 直接 image_reader で cnn_board 取得 (= classify 結果)
    p1_board, p2_board = pipe._reader.read_both_boards(fr)
    hsv_full = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)

    print(f"\n=== cell classification at t={args.sec}s ===")
    for side, region, board in [
        ("1P", DEFAULT_P1_REGION, p1_board),
        ("2P", DEFAULT_P2_REGION, p2_board),
    ]:
        print(f"\n--- {side} ---")
        cnt = Counter()
        for r in range(HIDDEN_ROWS, 13):
            row_str = []
            for c in range(BOARD_COLS):
                v = int(board.get(r, c))
                cnt[v] += 1
                # HSV 中央値も取得
                x1, y1, x2, y2 = region.cell_sample_rect(r, c)
                patch_hsv = hsv_full[y1:y2, x1:x2]
                h_med = int(np.median(patch_hsv[:, :, 0]))
                s_med = int(np.median(patch_hsv[:, :, 1]))
                v_med = int(np.median(patch_hsv[:, :, 2]))
                row_str.append(
                    f"{COLOR_NAMES.get(v, '?'):>6}({h_med:>3},{s_med:>3},{v_med:>3})",
                )
            print(f"  r={r}: " + " | ".join(row_str))
        print(f"  count: {dict(cnt)}")

    # 視覚化 PNG (cell 矩形 + classify 結果文字)
    if args.out_png is not None:
        viz = fr.copy()
        for side, region, board in [
            ("1P", DEFAULT_P1_REGION, p1_board),
            ("2P", DEFAULT_P2_REGION, p2_board),
        ]:
            for r in range(HIDDEN_ROWS, 13):
                for c in range(BOARD_COLS):
                    v = int(board.get(r, c))
                    x1, y1, x2, y2 = region.cell_sample_rect(r, c)
                    color_box = (
                        (0, 0, 255) if v == 0 else
                        (255, 255, 255) if v == 10 else
                        (0, 255, 255)
                    )
                    cv2.rectangle(viz, (x1, y1), (x2, y2), color_box, 2)
                    label = COLOR_NAMES.get(v, "?")[0]
                    cv2.putText(
                        viz, label, (x1 + 4, y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4,
                    )
                    cv2.putText(
                        viz, label, (x1 + 4, y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    )
        cv2.imwrite(str(args.out_png), viz)
        print(f"viz saved: {args.out_png}")


if __name__ == "__main__":
    main()
