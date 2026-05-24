"""v97 frame 180 (= 3s @ 60fps) の認識挙動を診断 (cycle 71v debug).

ユーザー報告: 1P 1 列目 row 5,6 / 2P row 3,4 段目に明らかな誤認.
本スクリプトで該当 frame の cnn_board / confirmed_board / inferred / signals を dump.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU で十分

import cv2

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN, Board
from src.board_state_machine import BoardState
from src.recognition_pipeline import RecognitionPipeline


COLOR_NAMES = {
    0: "EMPTY", 1: "RED  ", 2: "BLUE ", 3: "GREEN",
    4: "YEL  ", 5: "PURP ", 9: "OJAMA", 10: "UNK  ",
}


def fmt_board(b: Board | None) -> str:
    if b is None:
        return "(None)"
    lines = []
    for r in range(BOARD_ROWS):
        row = []
        for c in range(BOARD_COLS):
            v = int(b.get(r, c))
            row.append(f"{v:2}")
        lines.append(f"r{r:2}: " + " ".join(row))
    return "\n".join(lines)


def main() -> int:
    video = "data/evaluation_videos/v97_match11_96s.mp4"
    target_frame = 180  # 3s @ 60fps

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[diag] video={video} fps={fps:.1f} frames={n_frames}")
    print(f"[diag] target_frame={target_frame} (t={target_frame/fps:.2f}s)")

    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=Path("models/cnn_phase_b_large_v3.pt"),
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
    )

    # 動画解像度を image_reader に通知 (= 720p)
    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if hasattr(pipe._reader, "set_resolution_aware_s_min"):
        pipe._reader.set_resolution_aware_s_min(h_src)

    last_p1_state = BoardState.MENU
    last_p2_state = BoardState.MENU
    last_p1_inferred = None
    last_p2_inferred = None

    for fi in range(target_frame + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t_sec = fi / fps
        result = pipe.update(fi, t_sec, frame)
        if fi == target_frame:
            print(f"\n=== frame {fi} (t={t_sec:.2f}s) ===")
            print(f"is_match_active={result.is_match_active}")
            print(f"\n[1P] state={result.p1.state.value}")
            print(f"  cnn_board (CNN 観測):")
            print(fmt_board(result.p1.cnn_board))
            print(f"\n  confirmed_board (overlay 表示元):")
            print(fmt_board(result.p1.confirmed_board))
            print(f"\n  inferred_board:")
            print(fmt_board(result.p1.inferred_board))
            print(f"  next_pair={result.p1.next_pair}")
            print(f"  dnext_pair={result.p1.dnext_pair}")
            print(f"  chain_event={result.p1.chain_event}")
            print(f"  score={result.p1.score}")
            print(f"  drift={result.p1.drift}")

            print(f"\n[2P] state={result.p2.state.value}")
            print(f"  cnn_board (CNN 観測):")
            print(fmt_board(result.p2.cnn_board))
            print(f"\n  confirmed_board (overlay 表示元):")
            print(fmt_board(result.p2.confirmed_board))
            print(f"\n  inferred_board:")
            print(fmt_board(result.p2.inferred_board))
            print(f"  next_pair={result.p2.next_pair}")
            print(f"  dnext_pair={result.p2.dnext_pair}")
            print(f"  chain_event={result.p2.chain_event}")
            print(f"  score={result.p2.score}")
            print(f"  drift={result.p2.drift}")

            # 内部 state 追加情報
            sm1 = pipe._sm_1p.context
            sm2 = pipe._sm_2p.context
            print(f"\n[1P sm] pending_count={sm1.pending_count} "
                  f"last_stable_idx={sm1.last_stable_idx} "
                  f"next_queue={sm1.next_queue}")
            print(f"[2P sm] pending_count={sm2.pending_count} "
                  f"last_stable_idx={sm2.last_stable_idx} "
                  f"next_queue={sm2.next_queue}")
            print(f"[1P landing_grace] {pipe._landing_grace_1p}")
            print(f"[2P landing_grace] {pipe._landing_grace_2p}")
            print(f"[1P tsumo_count] {dict(pipe._tsumo_count_1p)}")
            print(f"[2P tsumo_count] {dict(pipe._tsumo_count_2p)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
