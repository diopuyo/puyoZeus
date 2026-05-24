"""v97 で TSUMO_FALL→STABLE 遷移時の placement 確定挙動を全 trace.

frame 180 までの各 placement について:
  - state 遷移時刻
  - confirmed_before (= 直前 STABLE 盤面)
  - cnn_after (= 着地直後 CNN 観測)
  - inferred_landing (= 物理推論結果)
  - diff cells (= cnn で puyo 出現した cells)
  - next_pair / dnext_pair

これで「どの placement で column を間違えたか」 を特定可能.
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

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN, Board
from src.board_state_machine import BoardState
from src.recognition_pipeline import RecognitionPipeline


def get_diffs(base: Board | None, new: Board) -> list[tuple[int, int, int, int]]:
    """base empty / new non-empty cells を [(r, c, base_v, new_v)] で返す."""
    if base is None:
        return []
    out = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            bv = int(base.get(r, c))
            nv = int(new.get(r, c))
            if bv == COLOR_EMPTY and nv != COLOR_EMPTY and nv != COLOR_UNKNOWN:
                out.append((r, c, bv, nv))
    return out


def board_diff(a: Board | None, b: Board | None) -> list[tuple[int, int, int, int]]:
    if a is None or b is None:
        return []
    out = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            av = int(a.get(r, c))
            bv = int(b.get(r, c))
            if av != bv:
                out.append((r, c, av, bv))
    return out


def main() -> int:
    video = "data/evaluation_videos/v97_match11_96s.mp4"
    target_frame = 200
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[diag] fps={fps:.1f} target_frame={target_frame} (t={target_frame/fps:.2f}s)")

    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=Path("models/cnn_phase_b_large_v3.pt"),
        load_next_detector=True,
        force_in_match=True,
    )
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if hasattr(pipe._reader, "set_resolution_aware_s_min"):
        pipe._reader.set_resolution_aware_s_min(h_src)

    prev_p1_state = BoardState.MENU
    prev_p2_state = BoardState.MENU
    prev_p1_confirmed: Board | None = None
    prev_p2_confirmed: Board | None = None

    for fi in range(target_frame + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t_sec = fi / fps
        result = pipe.update(fi, t_sec, frame)
        # 1P side イベント検出
        for side, cur_state, prev_state, prev_conf in [
            ("1P", result.p1.state, prev_p1_state, prev_p1_confirmed),
            ("2P", result.p2.state, prev_p2_state, prev_p2_confirmed),
        ]:
            res = result.p1 if side == "1P" else result.p2
            # TSUMO_FALL→STABLE 検出 OR STABLE 中の confirmed_board 変化
            if (prev_state == BoardState.TSUMO_FALL
                    and cur_state == BoardState.STABLE):
                cur_conf = res.confirmed_board
                cnn_b = res.cnn_board
                diffs_from_prev = board_diff(prev_conf, cur_conf)
                diffs_cnn = get_diffs(prev_conf, cnn_b)
                print(f"\n--- frame {fi} (t={t_sec:.2f}s) {side} "
                      f"TSUMO_FALL→STABLE 遷移 ---")
                print(f"  next_pair={res.next_pair}")
                print(f"  dnext_pair={res.dnext_pair}")
                print(f"  confirmed_before→after diff cells:")
                for (r, c, av, bv) in diffs_from_prev:
                    print(f"    ({r},{c}): {av} → {bv}")
                print(f"  cnn_after が prev_confirmed 比で空→puyo になった cells:")
                for (r, c, _bv, nv) in diffs_cnn:
                    print(f"    ({r},{c}) = {nv}")
        # 状態保存
        prev_p1_state = result.p1.state
        prev_p2_state = result.p2.state
        if result.p1.confirmed_board is not None:
            prev_p1_confirmed = result.p1.confirmed_board.copy()
        if result.p2.confirmed_board is not None:
            prev_p2_confirmed = result.p2.confirmed_board.copy()

    return 0


if __name__ == "__main__":
    sys.exit(main())
