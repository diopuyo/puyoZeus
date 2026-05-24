"""ネクスト履歴ベース landing 色精度評価 (manual label 不要).

NextDetector は v89 で 100% 精度確認済 (= ground truth 信頼).
NEXT pair の遷移 = ツモが置かれたタイミング. その後 confirmed_board に
新しく加わった 2 cell の色が, 消えた NEXT pair の色と一致するか検査.

使い方:
    python scripts/eval_landing_via_next_history.py \
        --video data/evaluation_videos/v89_match3_95s.mp4 \
        --hsv-state data/per_video_hsv_ranges/v89.json \
        --max-sec 90
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.image_reader import COLOR_EMPTY
from src.recognition_pipeline import RecognitionPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--hsv-state", type=Path, default=None)
    p.add_argument("--cnn-model", type=Path,
                   default=Path("models/cnn_phase_b_finetuned.pt"))
    p.add_argument("--max-sec", type=float, default=90.0)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def load_pipeline(cnn_model: Path, hsv_state: Path | None) -> RecognitionPipeline:
    pipe = RecognitionPipeline.load_default(
        cnn_model_path=cnn_model, force_in_match=True,
    )
    if hsv_state is not None:
        with hsv_state.open() as f:
            state = json.load(f)
        ranges = {
            int(k): tuple(int(x) for x in v)
            for k, v in state["per_video_ranges"].items()
        }
        from src.hybrid_classifier import HybridClassifier
        hc = pipe._reader._classifier
        if isinstance(hc, HybridClassifier) and ranges:
            hc._hsv.set_color_ranges_from_simple(ranges)
    return pipe


def board_to_grid(b) -> np.ndarray:
    """Board → 13x6 ndarray."""
    return np.asarray([[int(b.get(r, c)) for c in range(6)] for r in range(13)])


def diff_top_cells(
    prev_grid: np.ndarray | None, cur_grid: np.ndarray,
) -> list[tuple[int, int, int]]:
    """直前の confirmed_board と比較して新規追加 cell を返す.

    Returns: list of (row, col, color) for newly-occupied cells.
    """
    if prev_grid is None:
        return []
    new_cells: list[tuple[int, int, int]] = []
    for r in range(13):
        for c in range(6):
            prev_v = int(prev_grid[r, c])
            cur_v = int(cur_grid[r, c])
            if prev_v == COLOR_EMPTY and cur_v != COLOR_EMPTY and cur_v != 9:
                new_cells.append((r, c, cur_v))
    return new_cells


def score_match(
    new_cells: list[tuple[int, int, int]],
    next_pair: tuple[int, int],
) -> tuple[int, int]:
    """新規 cell の色が next_pair (top, bot) の 2 色と一致する数を返す.

    回転含めた色集合比較 (= 順序問わず).
    new_cells が 2 個ない場合: matched=0, total=0 (= skip サンプル).
    Returns: (matched, total)
    """
    # 通常 1 ツモ = 2 cell. 連鎖直前など 2 以外なら skip.
    if len(new_cells) != 2:
        return (0, 0)
    placed_colors = sorted([new_cells[0][2], new_cells[1][2]])
    next_colors = sorted(list(next_pair))
    matched = 0
    if placed_colors == next_colors:
        return (2, 2)
    # 部分一致 (ぞろ目との混同などに耐性)
    placed_counter = Counter(placed_colors)
    for c in next_colors:
        if placed_counter.get(c, 0) > 0:
            placed_counter[c] -= 1
            matched += 1
    return (matched, 2)


def evaluate(args: argparse.Namespace) -> dict:
    pipe = load_pipeline(args.cnn_model, args.hsv_state)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if hasattr(pipe._reader, "set_resolution_aware_s_min"):
        pipe._reader.set_resolution_aware_s_min(src_h)
    max_frames = int(args.max_sec * fps)

    # 各 side ごとの状態を track
    # board_driven: 確定 board に新規 cell が 2 個追加された frame で score.
    # その時点の NEXT history (= 直近の安定 next_pair) と新規 cell 色を比較.
    # next_history: list of (frame, next_pair) — 最近 200 frame 保持.
    state_1p = {
        "prev_next": None, "prev_board": None,
        "next_history": [],
        "matched": 0, "total": 0, "samples": 0, "details": [],
        "placements": 0, "skipped": [],
    }
    state_2p = {
        "prev_next": None, "prev_board": None,
        "next_history": [],
        "matched": 0, "total": 0, "samples": 0, "details": [],
        "placements": 0, "skipped": [],
    }

    fi = 0
    while fi < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[:2] != (1080, 1920):
            fr = cv2.resize(fr, (1920, 1080))
        result = pipe.update(fi, fi / fps, fr)
        for side_name, side_state, pside in [
            ("1P", state_1p, result.p1),
            ("2P", state_2p, result.p2),
        ]:
            cur_board = pside.confirmed_board
            cur_next = pside.next_pair
            if cur_board is None or cur_next is None:
                fi += 1
                continue
            cur_arr = board_to_grid(cur_board)
            # next_history を最大 200 frame 保持
            side_state["next_history"].append((fi, cur_next))
            if len(side_state["next_history"]) > 200:
                side_state["next_history"].pop(0)
            # board-driven: 確定 board に新規 cell が 2 個追加された frame で score
            # フィルタ: 同列 (col 一致) かつ隣接 row (= 縦置き 1 ツモ) のみ採用.
            # 横置き (隣接 col, 同 row) も採用. それ以外は連鎖中などのため skip.
            if side_state["prev_board"] is not None:
                new_cells = diff_top_cells(side_state["prev_board"], cur_arr)
                is_vertical = (
                    len(new_cells) == 2
                    and new_cells[0][1] == new_cells[1][1]
                    and abs(new_cells[0][0] - new_cells[1][0]) == 1
                )
                is_horizontal = (
                    len(new_cells) == 2
                    and new_cells[0][0] == new_cells[1][0]
                    and abs(new_cells[0][1] - new_cells[1][1]) == 1
                )
                if len(new_cells) == 2 and (is_vertical or is_horizontal):
                    side_state["placements"] += 1
                    # 直近 5-30 frame 前 (= placement 前) で最も多く出現した next_pair
                    history = side_state["next_history"][-31:-5] if len(side_state["next_history"]) >= 31 else side_state["next_history"][:-5]
                    if not history:
                        history = side_state["next_history"][:-1]
                    pair_counts: Counter = Counter()
                    for _, np_pair in history:
                        if np_pair is not None and all(c > 0 for c in np_pair):
                            pair_counts[np_pair] += 1
                    if not pair_counts:
                        side_state["skipped"].append({
                            "frame": fi, "time_sec": round(fi / fps, 2),
                            "reason": "no stable next_pair in history",
                            "new_cells": [list(x) for x in new_cells],
                        })
                    else:
                        consumed_next = pair_counts.most_common(1)[0][0]
                        m, t = score_match(new_cells, consumed_next)
                        side_state["matched"] += m
                        side_state["total"] += t
                        side_state["samples"] += 1
                        side_state["details"].append({
                            "frame": fi,
                            "time_sec": round(fi / fps, 2),
                            "consumed_next": list(consumed_next),
                            "history_count": pair_counts[consumed_next],
                            "new_cells": [list(x) for x in new_cells],
                            "matched": m,
                            "total": t,
                        })
            side_state["prev_board"] = cur_arr.copy()
            side_state["prev_next"] = cur_next
        fi += 1
    cap.release()

    return {
        "video": str(args.video),
        "hsv_state": str(args.hsv_state) if args.hsv_state else None,
        "max_sec": args.max_sec,
        "1P": {
            "placements": state_1p["placements"],
            "samples": state_1p["samples"],
            "matched": state_1p["matched"],
            "total": state_1p["total"],
            "accuracy": state_1p["matched"] / state_1p["total"] if state_1p["total"] else 0.0,
            "details": state_1p["details"],
            "skipped": state_1p["skipped"],
        },
        "2P": {
            "placements": state_2p["placements"],
            "samples": state_2p["samples"],
            "matched": state_2p["matched"],
            "total": state_2p["total"],
            "accuracy": state_2p["matched"] / state_2p["total"] if state_2p["total"] else 0.0,
            "details": state_2p["details"],
            "skipped": state_2p["skipped"],
        },
    }


def main() -> None:
    args = parse_args()
    r = evaluate(args)
    print("=== Landing color accuracy via NEXT history ===")
    print(f"video: {args.video.name}")
    print(f"hsv_state: {args.hsv_state}")
    print(f"max_sec: {args.max_sec}")
    for side in ("1P", "2P"):
        s = r[side]
        print(
            f"  {side}: placements={s['placements']} "
            f"samples={s['samples']} skipped={len(s['skipped'])} "
            f"matched={s['matched']}/{s['total']} acc={s['accuracy']:.3f}",
        )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as f:
            json.dump(r, f, indent=2)
        print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()
