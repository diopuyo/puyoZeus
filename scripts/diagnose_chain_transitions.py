"""連鎖発火前後の confirmed_board 変化を per-frame log で記録する診断 script.

目的 (= 2026-05-11 引継ぎ Cycle 71):
    仮説 A (置き誤認→freeze) vs 仮説 B (chain detection 遅延) を判別する。

出力 (JSONL, 1 行 1 frame):
    {
        "frame_idx": int, "time_sec": float, "is_match_active": bool,
        "p1": {
            "state": str,
            "cnn_total": int, "cnn_by_color": {color: count},
            "confirmed_total": int, "confirmed_by_color": {color: count},
            "confirmed_grid_hash": str (8 文字),
            "confirmed_diff_to_prev": int (cell 単位 mismatch 数),
            "score": int|None, "score_delta": int,
            "chain_event": null|{
                "chain_count": int, "trigger_sec": float,
                "before_count": int, "sim_chain_count_now": int,
                "sim_chain_count_confirmed_now": int (= 仮説 A 判別),
            }
        },
        "p2": 同上,
    }

判別ヒント:
    - 仮説 A (= 置いた直後の confirmed_board が誤認、 連鎖前から間違っている):
      連鎖発火 frame で sim_chain_count_confirmed_now == 0 なら、
      confirmed_board からは連鎖発生不可 = confirmed の色が間違っている可能性高い
      (置き誤認 → freeze で固定が起きている)。
    - 仮説 B (= chain detection 遅延、 発火後 N frame 遅れて event 受信):
      score_delta が増えてから chain_event 受信までの frame 数を見る。
      その間の confirmed_board が animation 由来色を吸ったか観測。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.diagnose_chain_transitions \\
        --video data/test_unknown/v91_match1_75s_720p.mp4 \\
        --output data/diagnostics/v91_match1_75s_diag.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console  # noqa: E402

init_console()

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN, Board,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402


def board_signature(board: Board | None) -> str:
    """盤面の 8 文字 hash (= 前 frame との同一性比較用)."""
    if board is None:
        return "_none_"
    h = hashlib.sha1(board._grid.tobytes()).hexdigest()
    return h[:8]


def board_color_counts(board: Board | None) -> dict[str, int]:
    """色別 cell count (key は色 int の文字列、 JSON 互換)."""
    if board is None:
        return {}
    grid = board._grid
    counts: Counter[int] = Counter()
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            v = int(grid[r, c])
            if v != COLOR_EMPTY:
                counts[v] += 1
    return {str(k): int(v) for k, v in counts.items()}


def board_total_puyos(board: Board | None) -> int:
    if board is None:
        return 0
    return int(board.count_puyos())


def board_diff_cell_count(a: Board | None, b: Board | None) -> int:
    """2 盤面の cell 単位 mismatch 数 (None なら -1)."""
    if a is None or b is None:
        return -1
    return int(np.sum(a._grid != b._grid))


def safe_chain_simulate(
    board: Board | None, sim: ChainSimulator,
) -> int:
    """board から連鎖を simulate して chain_count を返す。 None なら 0。"""
    if board is None:
        return 0
    try:
        result = sim.simulate(board)
        return int(result.chain_count)
    except Exception:
        return 0


def chain_event_to_dict(
    chain_event: Any, confirmed_board: Board | None, sim: ChainSimulator,
) -> dict[str, Any] | None:
    """ChainEvent を JSON 可能 dict に変換 + 仮説判別ヒント追加."""
    if chain_event is None:
        return None
    before = chain_event.before_board
    return {
        "chain_count": int(chain_event.chain_count),
        "trigger_sec": float(chain_event.trigger_sec),
        "before_count": int(before.count_puyos()) if before is not None else 0,
        # before_board (= tracker が連鎖発火直前として保持していた cnn) で
        # 何連鎖になるかは tracker 側で simulate 済 = chain_count と同値の想定。
        # 確認用に confirmed_board (= state machine の確定盤面) でも simulate.
        "sim_chain_count_confirmed_now": safe_chain_simulate(
            confirmed_board, sim,
        ),
    }


def side_snapshot(
    side_result: Any,
    prev_confirmed: Board | None,
    sim: ChainSimulator,
) -> dict[str, Any]:
    """SideResult を JSON 可能 dict に変換."""
    confirmed = side_result.confirmed_board
    cnn = side_result.cnn_board
    return {
        "state": side_result.state.value,
        "cnn_total": board_total_puyos(cnn),
        "cnn_by_color": board_color_counts(cnn),
        "confirmed_total": board_total_puyos(confirmed),
        "confirmed_by_color": board_color_counts(confirmed),
        "confirmed_grid_hash": board_signature(confirmed),
        "confirmed_diff_to_prev": board_diff_cell_count(
            confirmed, prev_confirmed,
        ),
        "score": int(side_result.score) if side_result.score is not None else None,
        "score_delta": int(side_result.score_delta),
        "chain_event": chain_event_to_dict(
            side_result.chain_event, confirmed, sim,
        ),
        "next_pair": (
            [int(c) for c in side_result.next_pair]
            if side_result.next_pair is not None else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample-interval", type=float, default=1.0 / 60.0,
        help="認識処理する frame 間隔 (秒)。 default は全 frame 処理。",
    )
    parser.add_argument(
        "--max-sec", type=float, default=0.0,
        help="入力動画の処理最大秒数 (0=全部)",
    )
    parser.add_argument("--cnn-model", type=str, default=None)
    parser.add_argument(
        "--hsv-state", type=Path, default=None,
        help="動画別 HSV ranges JSON (per_video_hsv_ranges DB)。",
    )
    parser.add_argument(
        "--progress-every", type=int, default=300,
        help="N frame ごとに progress ログを stderr に書く",
    )
    parser.add_argument(
        "--vote-mode", action="store_true",
        help="ColorClassifier を per-pixel 投票方式に切替 (cycle 71)",
    )
    parser.add_argument(
        "--cnn-override-prob", type=float, default=None,
        help="HybridClassifier CNN 採用閾値 (None で default 0.70). "
             "AB 比較用に 0.5 / 0.7 / 0.9 等を指定.",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[ERROR] cannot open: {args.video}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.max_sec > 0:
        n_frames = min(n_frames, int(args.max_sec * fps))
    print(
        f"[input] {args.video} fps={fps:.1f} frames={n_frames} height={height}",
        file=sys.stderr,
    )

    # Pipeline (visualize_recognition.py と同じ構成)
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=args.cnn_model,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
        vote_mode=args.vote_mode,
        cnn_override_prob=args.cnn_override_prob,
    )
    if args.vote_mode:
        print("[diag] vote_mode=ON (per-pixel HSV voting)", file=sys.stderr)
    if args.cnn_override_prob is not None:
        print(
            f"[diag] cnn_override_prob={args.cnn_override_prob}",
            file=sys.stderr,
        )
    if hasattr(pipeline._reader, "set_resolution_aware_s_min"):
        pipeline._reader.set_resolution_aware_s_min(height)
    if args.hsv_state is not None:
        try:
            with args.hsv_state.open("r", encoding="utf-8") as _f:
                _state = json.load(_f)
            _ranges = _state.get("per_video_ranges", {})
            _ranges_int = {
                int(k): tuple(int(x) for x in v) for k, v in _ranges.items()
            }
            from src.hybrid_classifier import HybridClassifier
            _hc = pipeline._reader._classifier
            if (
                isinstance(_hc, HybridClassifier)
                and hasattr(_hc._hsv, "set_color_ranges_from_simple")
                and _ranges_int
            ):
                _hc._hsv.set_color_ranges_from_simple(_ranges_int)
                if pipeline._online_hsv is not None and height >= 720:
                    pipeline._online_hsv_injected = True
        except Exception as _e:
            print(f"[diag] HSV pre-inject failed: {_e}", file=sys.stderr)

    sample_interval_frames = max(1, int(round(args.sample_interval * fps)))
    sim = ChainSimulator()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.output.open("w", encoding="utf-8")

    prev_p1_confirmed: Board | None = None
    prev_p2_confirmed: Board | None = None
    n_written = 0

    try:
        for fi in range(n_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(
                    frame, (1920, 1080), interpolation=cv2.INTER_AREA,
                )
            t_sec = fi / fps
            if fi % sample_interval_frames != 0:
                continue
            result = pipeline.update(fi, t_sec, frame)

            row = {
                "frame_idx": int(fi),
                "time_sec": float(t_sec),
                "is_match_active": bool(result.is_match_active),
                "p1": side_snapshot(result.p1, prev_p1_confirmed, sim),
                "p2": side_snapshot(result.p2, prev_p2_confirmed, sim),
            }
            out_f.write(json.dumps(row, ensure_ascii=False))
            out_f.write("\n")
            n_written += 1

            if result.p1.confirmed_board is not None:
                prev_p1_confirmed = result.p1.confirmed_board
            if result.p2.confirmed_board is not None:
                prev_p2_confirmed = result.p2.confirmed_board

            if fi % args.progress_every == 0:
                print(
                    f"  [progress] {fi}/{n_frames} "
                    f"({fi*100/max(n_frames,1):.1f}%) "
                    f"1P={result.p1.state.value} 2P={result.p2.state.value} "
                    f"1P_score={result.p1.score} 2P_score={result.p2.score}",
                    file=sys.stderr, flush=True,
                )
    finally:
        out_f.close()
        cap.release()

    print(
        f"[done] {args.output} wrote {n_written} rows",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
