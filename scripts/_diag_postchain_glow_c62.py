"""段階(a) 診断専用スクリプト (2026-07-23, 連鎖後残像バグ根治タスク)。

目的: c62 game9 の t=908.4〜925.0 区間で、GRAVITY_SETTLE→STABLE 遷移フレームの
prev_state が実際に GRAVITY_SETTLE であり CHAIN でないこと (= Phase C-6 の C の
条件 `prev_state == BoardState.CHAIN` が発火しないこと) を数値確認する。

src/ は一切変更しない (読み取り専用の診断)。

使い方:
    PYTHONPATH=. python scripts/_diag_postchain_glow_c62.py
"""
from __future__ import annotations

import cv2

from src.board_state_machine import BoardState
from src.recognition_pipeline import RecognitionPipeline

VIDEO_PATH = "data/frames/video_c62.mp4"
# game9 は t=877.4〜949.2 (c62.npz game_idx==9 実測)。
# 状態機械/各種トラッカーの warmup を十分に取るため、game9 開始よりかなり前 (game8 尾部)
# から処理を開始し、診断対象区間 (908.4〜925.0) には広めのマージンを持たせる。
PROC_START_SEC = 850.0
DIAG_START_SEC = 903.0
DIAG_END_SEC = 930.0


def _fmt_state(state: BoardState) -> str:
    """BoardState を短縮表示する (診断ログの可読性用)。"""
    return state.value


def main() -> None:
    """c62 video を warmup 付きで処理し、診断区間の state 遷移を出力する。"""
    cv2.setNumThreads(1)  # 熱対策: 単一スレッドに制限 (feedback_thermal_safety_mandatory)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] open失敗: {VIDEO_PATH}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    proc_frame = int(PROC_START_SEC * fps)
    diag_end_frame = int(DIAG_END_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, proc_frame)
    print(
        f"[seek] fps={fps:.2f} proc_start={PROC_START_SEC}s "
        f"({proc_frame}frame) diag=[{DIAG_START_SEC},{DIAG_END_SEC}]s",
    )

    # 本番 (collect_indicators_v2.py / visualize_advantage_overlay.py) と同一構成。
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
    )
    pipe.set_video_id("c62")

    prev_p1_state: BoardState | None = None
    prev_p2_state: BoardState | None = None
    transitions: list[str] = []

    fi = proc_frame
    while fi < diag_end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = fi / fps
        r = pipe.update(fi, t, frame)
        if DIAG_START_SEC <= t <= DIAG_END_SEC:
            p1_puyo = r.p1.cnn_board.count_puyos() if r.p1.cnn_board is not None else -1
            p2_puyo = r.p2.cnn_board.count_puyos() if r.p2.cnn_board is not None else -1
            conf1_puyo = (
                r.p1.confirmed_board.count_puyos()
                if r.p1.confirmed_board is not None else -1
            )
            conf2_puyo = (
                r.p2.confirmed_board.count_puyos()
                if r.p2.confirmed_board is not None else -1
            )
            print(
                f"t={t:7.2f}  1P:{_fmt_state(r.p1.state):14s} "
                f"raw_puyo={p1_puyo:3d} conf_puyo={conf1_puyo:3d} | "
                f"2P:{_fmt_state(r.p2.state):14s} "
                f"raw_puyo={p2_puyo:3d} conf_puyo={conf2_puyo:3d}",
            )
            # 遷移検出 (prev_state → 現 state)
            if prev_p1_state is not None and prev_p1_state != r.p1.state:
                msg = (
                    f"[TRANSITION] 1P t={t:.2f} "
                    f"{_fmt_state(prev_p1_state)} -> {_fmt_state(r.p1.state)}"
                )
                print(msg)
                transitions.append(msg)
            if prev_p2_state is not None and prev_p2_state != r.p2.state:
                msg = (
                    f"[TRANSITION] 2P t={t:.2f} "
                    f"{_fmt_state(prev_p2_state)} -> {_fmt_state(r.p2.state)}"
                )
                print(msg)
                transitions.append(msg)
        prev_p1_state, prev_p2_state = r.p1.state, r.p2.state
        fi += 1

    cap.release()
    print("\n==== 診断サマリ ====")
    for msg in transitions:
        print(msg)
    settle_to_stable = [m for m in transitions if "GRAVITY_SETTLE -> STABLE" in m]
    chain_to_stable = [m for m in transitions if "CHAIN -> STABLE" in m]
    print(f"GRAVITY_SETTLE -> STABLE 遷移数: {len(settle_to_stable)}")
    print(f"CHAIN -> STABLE (直行) 遷移数: {len(chain_to_stable)}")
    if settle_to_stable and not chain_to_stable:
        print(
            "=> 確認: CHAIN は必ず GRAVITY_SETTLE を経由してから STABLE に"
            "遷移している (Phase C-6 の C は prev_state==CHAIN 条件で"
            "発火しない = dead code 化を実証)。",
        )


if __name__ == "__main__":
    main()
