"""W1.2: StatePipeline を動画で動かして 4 項目の精度を実測。

引数で指定した動画から、(start_sec, duration) の範囲を interval ごとに
StatePipeline で抽出し、得られた GameState を tsv に出力する。
ユーザー目視 / 簡易統計で精度確認できる。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_state_pipeline_eval \
        data/frames/video_01.mp4 --start 220 --duration 30 --interval 1.0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2
import numpy as np

from src.state_pipeline import StatePipeline


def _board_to_str(board) -> str:
    """6 列の各列のぷよ数を返す簡易表現 (height profile)。"""
    from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, HIDDEN_ROWS
    heights = []
    for col in range(BOARD_COLS):
        h = 0
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            if int(board.get(row, col)) != COLOR_EMPTY:
                h = (BOARD_ROWS - row)
                break
        heights.append(h)
    return ",".join(str(h) for h in heights)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--start", type=float, default=200.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--out-tsv", default="data/verify/phase_w_state_eval.tsv")
    parser.add_argument(
        "--match-start", type=float, default=-1.0,
        help="試合開始秒 (StatePipeline.reset で渡す)。負なら start を採用。",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"video open failed: {args.video}")
        return 1

    pipeline = StatePipeline()
    match_start = args.match_start if args.match_start >= 0 else args.start
    pipeline.reset(match_start_sec=match_start)
    print(f"video: {args.video}")
    print(f"start: {args.start}s, duration: {args.duration}s, interval: {args.interval}s")

    rows: list[str] = []
    rows.append(
        "t_sec\theights_p1\theights_p2\tnext_p1\tnext_p2\t"
        "score_p1\tscore_p2\tscore_conf_p1\tscore_conf_p2\t"
        "pending_ojama_p1\tpending_ojama_p2\ttelop\tlocked"
    )

    n_steps = int(args.duration / args.interval)
    for i in range(n_steps + 1):
        t = args.start + i * args.interval
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok or fr is None:
            print(f"  fetch failed at t={t}")
            continue
        state = pipeline.extract(fr, t_sec=t)

        next_p1_str = (
            f"{state.next_p1[0]},{state.next_p1[1]}"
            if state.next_p1 else "?"
        )
        next_p2_str = (
            f"{state.next_p2[0]},{state.next_p2[1]}"
            if state.next_p2 else "?"
        )
        score_p1 = state.score_p1 if state.score_p1 is not None else "?"
        score_p2 = state.score_p2 if state.score_p2 is not None else "?"
        rows.append(
            f"{t:.1f}\t{_board_to_str(state.board_p1)}\t"
            f"{_board_to_str(state.board_p2)}\t"
            f"{next_p1_str}\t{next_p2_str}\t"
            f"{score_p1}\t{score_p2}\t"
            f"{state.score_confidence_p1:.2f}\t{state.score_confidence_p2:.2f}\t"
            f"{state.pending_ojama_p1}\t{state.pending_ojama_p2}\t"
            f"{int(state.is_telop_visible)}\t{int(state.is_match_end_locked)}"
        )
        if i % 10 == 0:
            print(f"  [{i}/{n_steps}] t={t:.1f}s")

    cap.release()

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\nsaved: {to_windows_path(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
