"""動画再生して各 placement の色精度を auto-judge.

各 TSUMO_FALL→STABLE event で:
  - 物理推論 placement (= NEXT pair 色) を取得
  - その placement の cells が long-term vote 後にどう確定したか追跡
  - 初期色 vs 終局色 (= 数秒後の confirmed_board) の不一致を「色誤認」 としてカウント

出力:
  - misclassified_cells: 色変わった cells 数
  - placement_count: total placements
  - error_rate: misclassified / (placement_count * 2)
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

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN, Board,
)
from src.board_state_machine import BoardState
from src.recognition_pipeline import RecognitionPipeline


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--cnn-model", type=Path,
                   default=Path("models/cnn_phase_b_large_v3.pt"))
    p.add_argument("--max-frames", type=int, default=0,
                   help="0=全 frames")
    p.add_argument("--check-delay", type=int, default=60,
                   help="着地から N frame 後の最終確定色を比較")
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

    placements: list[dict] = []  # 各 placement イベント
    prev_p1_state = BoardState.MENU
    prev_p2_state = BoardState.MENU
    prev_p1_conf: Board | None = None
    prev_p2_conf: Board | None = None
    frame_history_1p: list[Board | None] = []
    frame_history_2p: list[Board | None] = []

    print(f"[audit] {args.video.name} fps={fps:.1f} frames={n_target}")

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
        # フレーム毎の confirmed snapshot を保存 (= 後で N frame 後を取れるよう)
        frame_history_1p.append(
            result.p1.confirmed_board.copy()
            if result.p1.confirmed_board is not None else None
        )
        frame_history_2p.append(
            result.p2.confirmed_board.copy()
            if result.p2.confirmed_board is not None else None
        )
        for side, cur_state, prev_state, prev_conf in [
            ("1P", result.p1.state, prev_p1_state, prev_p1_conf),
            ("2P", result.p2.state, prev_p2_state, prev_p2_conf),
        ]:
            res = result.p1 if side == "1P" else result.p2
            if (
                prev_state == BoardState.TSUMO_FALL
                and cur_state == BoardState.STABLE
                and prev_conf is not None
                and res.confirmed_board is not None
            ):
                # 着地イベント検出: prev_conf → cur_conf で変化した cells (= 配置 cells)
                cur_conf = res.confirmed_board
                diff_cells: list[tuple[int, int, int]] = []
                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        bv = int(prev_conf.get(r, c))
                        nv = int(cur_conf.get(r, c))
                        if bv == COLOR_EMPTY and nv not in (
                            COLOR_EMPTY, COLOR_UNKNOWN,
                        ):
                            diff_cells.append((r, c, nv))
                if diff_cells:
                    placements.append({
                        "frame": fi,
                        "t_sec": round(t_sec, 2),
                        "side": side,
                        "next_pair": res.next_pair,
                        "initial_cells": diff_cells,
                    })
        prev_p1_state = result.p1.state
        prev_p2_state = result.p2.state
        if result.p1.confirmed_board is not None:
            prev_p1_conf = result.p1.confirmed_board.copy()
        if result.p2.confirmed_board is not None:
            prev_p2_conf = result.p2.confirmed_board.copy()

    # 各 placement について N frame 後の確定色を取得
    misclass_total = 0
    unknown_total = 0
    placement_total = len(placements)
    for ev in placements:
        check_frame = ev["frame"] + args.check_delay
        if check_frame >= len(frame_history_1p):
            continue
        hist = (
            frame_history_1p if ev["side"] == "1P"
            else frame_history_2p
        )
        future_board = hist[check_frame]
        if future_board is None:
            continue
        for (r, c, initial_color) in ev["initial_cells"]:
            future_color = int(future_board.get(r, c))
            if future_color == COLOR_UNKNOWN:
                unknown_total += 1
            elif future_color == COLOR_EMPTY:
                # cell が消去された (= chain) ので評価対象外
                continue
            elif future_color != initial_color:
                misclass_total += 1

    print(f"[audit] placements_detected={placement_total}")
    print(f"[audit] misclassified_cells (初期色 != {args.check_delay} frame "
          f"後)={misclass_total}")
    print(f"[audit] unknown_cells ({args.check_delay} frame 後)={unknown_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
