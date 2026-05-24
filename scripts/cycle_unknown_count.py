"""viz mp4 を再走させて、 各 frame の confirmed_board の UNKNOWN セル数を集計.

各 cycle の overlay 上の "?" 出現量を numerical に比較できる metric.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import cv2

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_UNKNOWN, Board
from src.board_state_machine import BoardState
from src.recognition_pipeline import RecognitionPipeline


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--cnn-model", type=Path,
                   default=Path("models/cnn_phase_b_large_v3.pt"))
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_target = (
        min(n_total, args.max_frames) if args.max_frames > 0 else n_total
    )
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=args.cnn_model,
        load_next_detector=True,
        force_in_match=True,
    )
    if hasattr(pipe._reader, "set_resolution_aware_s_min"):
        pipe._reader.set_resolution_aware_s_min(h_src)

    total_unknown_cells = 0
    stable_frame_count = 0
    p1_unknown_sum = 0
    p2_unknown_sum = 0

    for fi in range(n_target):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        t_sec = fi / fps
        result = pipe.update(fi, t_sec, frame)
        if (result.p1.confirmed_board is not None
                and result.p1.state == BoardState.STABLE):
            cnt = 0
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    if int(result.p1.confirmed_board.get(r, c)) == COLOR_UNKNOWN:
                        cnt += 1
            p1_unknown_sum += cnt
            stable_frame_count += 1
        if (result.p2.confirmed_board is not None
                and result.p2.state == BoardState.STABLE):
            cnt = 0
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    if int(result.p2.confirmed_board.get(r, c)) == COLOR_UNKNOWN:
                        cnt += 1
            p2_unknown_sum += cnt
            stable_frame_count += 1

    print(f"[unknown-count] video={args.video.name}")
    print(f"  total_frames={n_target}")
    print(f"  STABLE side-frames={stable_frame_count}")
    print(f"  1P unknown cells sum={p1_unknown_sum}")
    print(f"  2P unknown cells sum={p2_unknown_sum}")
    print(f"  total unknown cells sum={p1_unknown_sum + p2_unknown_sum}")
    if stable_frame_count > 0:
        avg = (p1_unknown_sum + p2_unknown_sum) / stable_frame_count
        print(f"  avg unknown per side-frame={avg:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
