"""R2 (浮きぷよ是正機構) 効果測定 (2026-08-17、計装専用・src/ の再変更なし)。

W13 (docs/KNOWN_WEAKNESSES.md) の実画面場面 (review_demo_2026-08-12.mp4、
試合3開始付近 t=276.7〜288秒、1P列0・2P列2 が実在ぷよのまま丸ごとEMPTY化) を、
**W13根治フラグ (enable_patch_fp_hsv_guard) を外した状態**で再現し、
R2 (enable_floating_gap_restore) 単体で列消失が復元されるかを確認する
(= 防御の多層化: hsv-guard が防ぎ損ねた場合の第二防衛線としての効果測定)。

ベース構成 = 本番採用構成 (RECOGNITION_ADOPTED) から enable_patch_fp_hsv_guard
のみ除外したもの。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, HIDDEN_ROWS
from src.fps_normalize import resolve_normalize_fps_30_stride
from src.production_config import recognition_load_default_kwargs
from src.recognition_pipeline import RecognitionPipeline

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT_DIR = Path("data/verify/diag_r2_floating_gap_restore_2026-08-17")

TARGET_1P_COL = 0
TARGET_2P_COL = 2


def build_pipeline(enable_floating_gap_restore: bool) -> RecognitionPipeline:
    """本番採用構成から enable_patch_fp_hsv_guard を除外 + R2 フラグのみ上書き。"""
    kwargs = dict(recognition_load_default_kwargs())
    kwargs.pop("enable_patch_fp_hsv_guard", None)  # W13根治を意図的に外す
    kwargs["enable_floating_gap_restore"] = enable_floating_gap_restore
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=False,
        **kwargs,
    )
    return pipe


def run_pass(
    enable_floating_gap_restore: bool, start_sec: float, end_sec: float,
) -> list[dict]:
    pipe = build_pipeline(enable_floating_gap_restore)
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = resolve_normalize_fps_30_stride(fps)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    records: list[dict] = []
    last_tick_reported = -1.0
    for fi in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if (fi - start_frame) % stride != 0:
            continue
        t = fi / fps
        recog_frame = (
            frame if frame.shape[:2] == (1080, 1920)
            else cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        )
        r = pipe.update(fi, t, recog_frame)
        tick = round(t, 1)
        if tick == last_tick_reported:
            continue
        last_tick_reported = tick

        def col_colors(board, col: int) -> list[int]:
            if board is None:
                return [-1] * (BOARD_ROWS - HIDDEN_ROWS)
            return [
                int(board.get(row, col))
                for row in range(HIDDEN_ROWS, BOARD_ROWS)
            ]

        def full_board_colors(board) -> list[list[int]] | None:
            if board is None:
                return None
            return [
                [int(board.get(row, col)) for col in range(BOARD_COLS)]
                for row in range(HIDDEN_ROWS, BOARD_ROWS)
            ]

        records.append({
            "t": tick, "fi": fi,
            "p1_col0": col_colors(r.p1.confirmed_board, TARGET_1P_COL),
            "p2_col2": col_colors(r.p2.confirmed_board, TARGET_2P_COL),
            "p1_full": full_board_colors(r.p1.confirmed_board),
            "p2_full": full_board_colors(r.p2.confirmed_board),
        })
    cap.release()
    return records


def detect_recovery_gaps(
    records: list[dict], key: str, col_len: int,
) -> dict[int, list[tuple[float, float, int]]]:
    gaps: dict[int, list[tuple[float, float, int]]] = {r: [] for r in range(col_len)}
    for row in range(col_len):
        seq = [(rec["t"], rec[key][row]) for rec in records if rec[key][row] != -1]
        i = 0
        n = len(seq)
        while i < n:
            t0, c0 = seq[i]
            if c0 == COLOR_EMPTY:
                i += 1
                continue
            j = i + 1
            empty_run = 0
            while j < n and seq[j][1] == COLOR_EMPTY:
                empty_run += 1
                j += 1
            if empty_run > 0 and j < n and seq[j][1] == c0:
                gaps[row].append((seq[i + 1][0], seq[j - 1][0], empty_run))
                i = j
            else:
                i += 1
    return gaps


def count_false_positive_cells(
    records_off: list[dict], records_on: list[dict],
) -> tuple[int, dict]:
    count = 0
    breakdown: Counter[tuple[str, int, int]] = Counter()
    by_t_on = {r["t"]: r for r in records_on}
    for rec_off in records_off:
        rec_on = by_t_on.get(rec_off["t"])
        if rec_on is None:
            continue
        for side_key, side_name in (("p1_full", "1P"), ("p2_full", "2P")):
            b_off, b_on = rec_off[side_key], rec_on[side_key]
            if b_off is None or b_on is None:
                continue
            for row in range(len(b_off)):
                for col in range(BOARD_COLS):
                    if b_off[row][col] == COLOR_EMPTY and b_on[row][col] != COLOR_EMPTY:
                        count += 1
                        breakdown[(side_name, row, col)] += 1
    top = {
        f"{side}_row{row}_col{col}": n
        for (side, row, col), n in breakdown.most_common(20)
    }
    return count, {"n_distinct_cells": len(breakdown), "top20": top}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-sec", type=float, default=250.0)
    ap.add_argument("--end-sec", type=float, default=300.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("OFF パス実行中 (hsv-guard 外し・R2 OFF、W13再現ベース)...")
    records_off = run_pass(False, args.start_sec, args.end_sec)
    print(f"  {len(records_off)} サンプル")

    print("ON パス実行中 (hsv-guard 外し・R2 ON: enable_floating_gap_restore=True)...")
    records_on = run_pass(True, args.start_sec, args.end_sec)
    print(f"  {len(records_on)} サンプル")

    col_len = BOARD_ROWS - HIDDEN_ROWS
    gaps_off_1p = detect_recovery_gaps(records_off, "p1_col0", col_len)
    gaps_off_2p = detect_recovery_gaps(records_off, "p2_col2", col_len)
    gaps_on_1p = detect_recovery_gaps(records_on, "p1_col0", col_len)
    gaps_on_2p = detect_recovery_gaps(records_on, "p2_col2", col_len)

    def summarize(gaps: dict) -> dict:
        out = {}
        for row, glist in gaps.items():
            if glist:
                out[row] = [
                    {"start_t": g[0], "end_t": g[1], "n_samples_empty": g[2]}
                    for g in glist
                ]
        return out

    fp_count, fp_breakdown = count_false_positive_cells(records_off, records_on)

    summary = {
        "video": str(VIDEO),
        "window": [args.start_sec, args.end_sec],
        "base_config": "recognition_load_default_kwargs() minus enable_patch_fp_hsv_guard",
        "target_cells": {"1P_col": TARGET_1P_COL, "2P_col": TARGET_2P_COL},
        "gaps_off_1p_col0": summarize(gaps_off_1p),
        "gaps_on_1p_col0": summarize(gaps_on_1p),
        "gaps_off_2p_col2": summarize(gaps_off_2p),
        "gaps_on_2p_col2": summarize(gaps_on_2p),
        "false_positive_cell_count_off_empty_on_nonempty": fp_count,
        "false_positive_breakdown": fp_breakdown,
        "n_records_off": len(records_off),
        "n_records_on": len(records_on),
    }

    out_json = OUT_DIR / "summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"サマリ保存先: {out_json}")
    print(f"副作用 (OFF=EMPTY→ON=非EMPTY) セル数: {fp_count}")
    print("1P col0 ギャップ (OFF):", summarize(gaps_off_1p))
    print("1P col0 ギャップ (ON):", summarize(gaps_on_1p))
    print("2P col2 ギャップ (OFF):", summarize(gaps_off_2p))
    print("2P col2 ギャップ (ON):", summarize(gaps_on_2p))


if __name__ == "__main__":
    main()
