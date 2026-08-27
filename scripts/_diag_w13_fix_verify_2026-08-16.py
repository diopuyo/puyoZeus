"""W13根治 案1 効果測定 (2026-08-16、計装専用・src/は無変更)。

docs/KNOWN_WEAKNESSES.md W13 の実画面場面 (review_demo_2026-08-12.mp4,
試合3開始付近 t=276.7〜288秒、1P列0・2P列2が実在ぷよのまま丸ごとEMPTY化) を
本番採用構成 (production_config.recognition_load_default_kwargs()) 上で再現し、
`enable_highlight_override` フラグ OFF/ON で該当セルの認識色を時系列 (0.1秒刻み)
で比較する。

やること:
  1. OFF (既定・従来挙動) と ON (W13根治 案1) の2パイプラインを同一フレーム列で
     独立に走らせる (state 汚染を避けるため各々フルパスで別実行)。
  2. 1P列0・2P列2 の各可視行 (visible_row 0..11) について、t=250〜300秒の間
     0.1秒刻みで confirmed_board の色を記録する。
  3. 「誤EMPTY」= 該当行が時系列内で非EMPTY色→EMPTY→同じ非EMPTY色に戻る
     (自然回復) パターンを機械的に検出し、そのEMPTY継続長 (フレーム数) を
     OFF/ON で比較する。
  4. 副作用チェック: 同じ2ラン全体 (全列全行) で「OFF=EMPTY だが ON=非EMPTY」
     というセルをカウントする (真の空セルが誤ってぷよ有りに倒れる方向のリスク)。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, HIDDEN_ROWS
from src.fps_normalize import resolve_normalize_fps_30_stride
from src.production_config import recognition_load_default_kwargs
from src.recognition_pipeline import RecognitionPipeline

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT_DIR = Path("data/verify/diag_w13_fix_2026-08-16")

# W13 該当セル (docs/KNOWN_WEAKNESSES.md 記載列)。
TARGET_1P_COL = 0
TARGET_2P_COL = 2


def build_pipeline(enable_highlight_override: bool) -> RecognitionPipeline:
    """本番採用構成 (RECOGNITION_ADOPTED) + W13根治フラグのみ上書きして構築する。"""
    kwargs = dict(recognition_load_default_kwargs())
    kwargs["enable_highlight_override"] = enable_highlight_override
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=False,
        **kwargs,
    )
    return pipe


def run_pass(
    enable_highlight_override: bool, start_sec: float, end_sec: float,
) -> list[dict]:
    """1パイプライン分をフルパス実行し、フレームごとの盤面スナップショットを返す。"""
    pipe = build_pipeline(enable_highlight_override)
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
        # 0.1秒刻みに間引いて記録 (要求仕様)
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
            "t": tick,
            "fi": fi,
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
    """各 visible_row について「非EMPTY→EMPTY→同色に復帰」ギャップを検出する.

    Returns: {visible_row: [(gap開始t, gap終了t, ギャップ長サンプル数), ...]}
    """
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
            # c0 は非EMPTY。この後 EMPTY が続き、同じ c0 に戻るか探す。
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
    """OFF=EMPTY だが ON=非EMPTY のセル数 (全列全行、副作用チェック)。

    Returns: (総サンプル数, {"side_row_col": count, ...} 上位内訳)
    """
    count = 0
    breakdown: Counter[tuple[str, int, int]] = Counter()
    by_t_on = {r["t"]: r for r in records_on}
    for rec_off in records_off:
        rec_on = by_t_on.get(rec_off["t"])
        if rec_on is None:
            continue
        if rec_off["p1_full"] is not None and rec_on["p1_full"] is not None:
            for row in range(len(rec_off["p1_full"])):
                for col in range(BOARD_COLS):
                    v_off = rec_off["p1_full"][row][col]
                    v_on = rec_on["p1_full"][row][col]
                    if v_off == COLOR_EMPTY and v_on != COLOR_EMPTY:
                        count += 1
                        breakdown[("1P", row, col)] += 1
        if rec_off["p2_full"] is not None and rec_on["p2_full"] is not None:
            for row in range(len(rec_off["p2_full"])):
                for col in range(BOARD_COLS):
                    v_off = rec_off["p2_full"][row][col]
                    v_on = rec_on["p2_full"][row][col]
                    if v_off == COLOR_EMPTY and v_on != COLOR_EMPTY:
                        count += 1
                        breakdown[("2P", row, col)] += 1
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

    print("OFF パス実行中 (既定・従来挙動)...")
    records_off = run_pass(False, args.start_sec, args.end_sec)
    print(f"  {len(records_off)} サンプル")

    print("ON パス実行中 (W13根治 案1: enable_highlight_override=True)...")
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

    # 生タイムライン (該当2列のみ) もCSV的に保存 (目視突合用)
    raw_out = OUT_DIR / "timeline_raw.json"
    with open(raw_out, "w", encoding="utf-8") as f:
        json.dump({
            "off": [{"t": r["t"], "p1_col0": r["p1_col0"], "p2_col2": r["p2_col2"]}
                    for r in records_off],
            "on": [{"t": r["t"], "p1_col0": r["p1_col0"], "p2_col2": r["p2_col2"]}
                   for r in records_on],
        }, f, ensure_ascii=False, indent=2)

    print(f"サマリ保存先: {out_json}")
    print(f"生タイムライン保存先: {raw_out}")
    print(f"副作用 (OFF=EMPTY→ON=非EMPTY) セル数: {fp_count}")
    print("1P col0 ギャップ (OFF):", summarize(gaps_off_1p))
    print("1P col0 ギャップ (ON):", summarize(gaps_on_1p))
    print("2P col2 ギャップ (OFF):", summarize(gaps_off_2p))
    print("2P col2 ギャップ (ON):", summarize(gaps_on_2p))


if __name__ == "__main__":
    main()
