"""持続誤認70件 (c13/c17/c21中心) の連続処理再検証 (2026-08-17)。

`scripts/_diag_w23_artifact_measure_2026-08-17.py` (c10/c23、W23機構限定計装) の
知見を踏まえ、残る70件の主要動画 (c13/c17/c21) について、統一測定 構成F
(= 構成E + --enable-next-history-starvation-fix) と全く同じ pipeline 構成で
「本番相当の連続処理 (force_in_match=False、実試合境界検知に任せる、
チャンク先頭から十分手前 (LOOKBACK_SEC 秒) から通し処理)」を実行し、
構成F のチャンク収集 (`data/indicators_v2/yardstick_v2_boards_f_2026-08-17/`、
force_in_match=True でチャンク先頭から新規pipeline構築) で観測された
持続誤認セルが同じ時刻窓で再現するかを比較する。

本体コード変更なし (フラグ経由の構成のみ、計装コードは追加なし)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_w23_continuous_verify_2026-08-17 --group c17_chunk0
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_w23_continuous_verify_2026-08-17 --all
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(_ROOT))

from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO_DIR = Path.home() / "frames"
OUT_DIR = _ROOT / "data" / "verify" / "diag_w23_continuous_2026-08-17"

CHUNK_SEC: float = 30.0
CHUNK_OFFSET_FRACTIONS: tuple[float, ...] = (0.08, 0.45, 0.80)
LOOKBACK_SEC: float = 300.0
END_MARGIN_SEC: float = 15.0
PERSIST_FRAME_THRESHOLD: float = 5.0  # 1次目標②と同じ閾値(effective 30fps換算)
EFFECTIVE_FPS: float = 30.0


@dataclass(frozen=True)
class TargetCell:
    sheet_id: str
    side: str  # "1P" / "2P"
    r: int
    c: int
    wrong_value: int
    correct_value: int
    t_start: float  # chunkモード(構成F実測)での破損区間開始秒
    t_end: float
    boundary_censored: bool


@dataclass(frozen=True)
class Group:
    name: str
    video_id: str
    chunk_idx: int
    targets: tuple[TargetCell, ...]


# _find_run(F実測npz) から抽出した target セル一覧 (video_id/chunk_idx別)。
GROUPS: dict[str, Group] = {
    "c17_chunk0": Group(
        "c17_chunk0", "c17", 0,
        (
            TargetCell("003_c17_2P_f17284", "2P", 1, 3, 0, 1, 286.233, 297.233, False),
            TargetCell("003_c17_2P_f17284", "2P", 1, 5, 0, 4, 286.233, 288.067, False),
            TargetCell("003_c17_2P_f17284", "2P", 2, 3, 0, 1, 286.233, 288.067, False),
            TargetCell("003_c17_2P_f17284", "2P", 2, 5, 0, 4, 286.233, 288.067, False),
            TargetCell("003_c17_2P_f17284", "2P", 3, 3, 0, 4, 286.233, 288.067, False),
            TargetCell("003_c17_2P_f17284", "2P", 3, 5, 0, 3, 286.233, 288.067, False),
            TargetCell("006_c17_2P_f17006", "2P", 2, 3, 0, 1, 283.300, 283.533, False),
            TargetCell("006_c17_2P_f17006", "2P", 2, 4, 0, 4, 282.133, 283.533, True),
            TargetCell("006_c17_2P_f17006", "2P", 2, 5, 0, 4, 282.133, 283.533, True),
            TargetCell("006_c17_2P_f17006", "2P", 3, 0, 0, 4, 282.133, 283.367, True),
            TargetCell("006_c17_2P_f17006", "2P", 3, 1, 0, 3, 282.133, 283.533, True),
            TargetCell("006_c17_2P_f17006", "2P", 3, 3, 0, 4, 283.300, 283.533, False),
            TargetCell("006_c17_2P_f17006", "2P", 3, 4, 0, 5, 282.133, 283.533, True),
            TargetCell("006_c17_2P_f17006", "2P", 3, 5, 0, 3, 283.300, 283.533, False),
            TargetCell("006_c17_2P_f17006", "2P", 4, 1, 0, 3, 282.133, 283.533, True),
            TargetCell("006_c17_2P_f17006", "2P", 4, 2, 0, 1, 283.300, 283.533, False),
            TargetCell("006_c17_2P_f17006", "2P", 4, 3, 0, 1, 283.300, 283.533, False),
            TargetCell("006_c17_2P_f17006", "2P", 4, 4, 0, 4, 283.300, 283.533, False),
            TargetCell("006_c17_2P_f17006", "2P", 5, 1, 0, 4, 282.133, 283.533, True),
            TargetCell("006_c17_2P_f17006", "2P", 5, 2, 0, 1, 283.300, 283.533, False),
            TargetCell("006_c17_2P_f17006", "2P", 5, 3, 0, 4, 283.300, 283.533, False),
            TargetCell("006_c17_2P_f17006", "2P", 5, 4, 0, 5, 283.300, 283.533, False),
            TargetCell("006_c17_2P_f17006", "2P", 6, 2, 0, 4, 283.300, 283.533, False),
            TargetCell("006_c17_2P_f17006", "2P", 6, 3, 0, 4, 283.300, 283.533, False),
            TargetCell("006_c17_2P_f17006", "2P", 7, 2, 0, 1, 283.300, 283.533, False),
            TargetCell("006_c17_2P_f17006", "2P", 7, 3, 0, 1, 283.300, 283.533, False),
            TargetCell("006_c17_2P_f17006", "2P", 8, 2, 0, 9, 283.300, 283.533, False),
            TargetCell("022_c17_1P_f17412", "1P", 1, 5, 2, 5, 289.500, 292.267, False),
            TargetCell("022_c17_1P_f17412", "1P", 5, 4, 2, 5, 289.500, 292.267, False),
            TargetCell("022_c17_1P_f17412", "1P", 6, 4, 2, 5, 289.500, 292.267, False),
        ),
    ),
    "c13_chunk0": Group(
        "c13_chunk0", "c13", 0,
        (
            TargetCell("013/026_c13_2P", "2P", 2, 2, 4, 9, 290.600, 291.133, False),
            TargetCell("013/026_c13_2P", "2P", 3, 2, 4, 9, 290.600, 291.133, False),
            TargetCell("013/026_c13_2P", "2P", 9, 1, 9, 4, 290.700, 291.067, False),
            TargetCell("026_c13_2P", "2P", 9, 3, 3, 9, 279.500, 292.500, True),
            TargetCell("013/026_c13_2P", "2P", 9, 4, 9, 4, 290.500, 291.100, False),
            TargetCell("013/026_c13_2P", "2P", 9, 5, 9, 4, 290.700, 291.100, False),
            TargetCell("013/026_c13_2P", "2P", 10, 1, 9, 3, 290.633, 291.100, False),
            TargetCell("013_c13_2P", "2P", 10, 3, 9, 2, 290.667, 290.933, False),
            TargetCell("013/026_c13_2P", "2P", 10, 4, 9, 2, 290.500, 291.067, False),
            TargetCell("013/026_c13_2P", "2P", 10, 5, 9, 5, 290.500, 291.100, False),
            TargetCell("013_c13_2P", "2P", 11, 0, 9, 3, 290.667, 290.933, False),
            TargetCell("013/026_c13_2P", "2P", 11, 4, 9, 4, 290.500, 291.067, False),
        ),
    ),
    "c13_chunk1": Group(
        "c13_chunk1", "c13", 1,
        (
            TargetCell("033_c13_2P_f91334", "2P", 4, 2, 2, 0, 1522.367, 1526.933, False),
            TargetCell("033_c13_2P_f91334", "2P", 5, 2, 1, 0, 1522.367, 1526.933, False),
        ),
    ),
    "c21_chunk1": Group(
        "c21_chunk1", "c21", 1,
        (
            TargetCell("051_c21_1P_f81796", "1P", 9, 5, 4, 0, 1363.600, 1371.533, True),
            TargetCell("051_c21_1P_f81796", "1P", 10, 5, 4, 0, 1363.600, 1371.533, True),
            TargetCell("057_c21_1P_f81890", "1P", 8, 3, 4, 0, 1365.067, 1371.533, True),
            TargetCell("057_c21_1P_f81890", "1P", 9, 3, 5, 0, 1365.067, 1371.533, True),
        ),
    ),
    "c21_chunk2": Group(
        "c21_chunk2", "c21", 2,
        (
            TargetCell("004_c21_1P_f144486", "1P", 4, 0, 9, 3, 2408.033, 2414.667, True),
            TargetCell("058_c21_2P_f143682", "2P", 5, 1, 1, 0, 2394.867, 2411.200, True),
            TargetCell("058_c21_2P_f143682", "2P", 6, 1, 2, 0, 2394.867, 2411.200, True),
        ),
    ),
    "c22_chunk1": Group(
        "c22_chunk1", "c22", 1,
        (
            TargetCell("043_c22_1P_f108676", "1P", 12, 2, 5, 2, 1811.300, 1819.333, True),
        ),
    ),
}


def video_filename_of(video_id: str) -> str:
    if video_id == "c96":
        return "_hold_video_c96.mp4"
    return f"video_{video_id}.mp4"


def probe_fps_duration(path: Path) -> tuple[float, float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fps, (n / fps if fps > 0 else 0.0)


def _build_pipeline(*, force_in_match: bool) -> RecognitionPipeline:
    """統一測定 構成F (= 構成E + starvation-fix) と同一構成。"""
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        load_next_detector=True,
        temporal_smoothing=1,
        force_in_match=force_in_match,
        enable_effect_gate=True,
        enable_burst_guard_v2=True,
        enable_transition_merge_guard=True,
        burst_gate_open_threshold=0.954,
        enable_hidden_row_burst_guard=True,
        enable_match_transition_debounce=True,
        enable_ojama_fall_placement_override=True,
        enable_patch_fp_hsv_guard=True,
        enable_chain_tracker=True,
        enable_floating_gap_restore=True,
        enable_landing_color_guard=True,
        enable_override_color_guard=True,
        enable_ojama_column_stack_fix=True,
        enable_next_history_starvation_fix=True,
    )


def _run_continuous(
    video_path: Path, fps: float, start_sec: float, end_sec: float,
    targets: tuple[TargetCell, ...],
) -> dict:
    """force_in_match=False で通し処理し、対象セルの値時系列を記録する。"""
    cap = cv2.VideoCapture(str(video_path))
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pipeline = _build_pipeline(force_in_match=False)
    cells_1p = sorted({(t.r, t.c) for t in targets if t.side == "1P"})
    cells_2p = sorted({(t.r, t.c) for t in targets if t.side == "2P"})
    series_1p: dict[tuple[int, int], list[tuple[float, int]]] = {rc: [] for rc in cells_1p}
    series_2p: dict[tuple[int, int], list[tuple[float, int]]] = {rc: [] for rc in cells_2p}
    match_started_hist: list[tuple[float, float]] = []
    frame_idx = start_frame
    t_sec = start_sec
    n_frames = 0
    while t_sec < end_sec:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        res = pipeline.update(frame_idx, t_sec, frame)
        # STABLE 確定盤面のみ記録 (CLAUDE.md 設計思想④: 評価は confirmed_board
        # のみ、NON-STABLE 中は評価対象外)。None のフレームは時系列から除外する
        # (=直前のSTABLE値が実質「凍結」として引き継がれる、本番の評価と同条件)。
        if res.p1.confirmed_board is not None:
            for rc in cells_1p:
                series_1p[rc].append((t_sec, int(res.p1.confirmed_board.get(*rc))))
        if res.p2.confirmed_board is not None:
            for rc in cells_2p:
                series_2p[rc].append((t_sec, int(res.p2.confirmed_board.get(*rc))))
        if n_frames % 300 == 0:
            match_started_hist.append((t_sec, pipeline._match_active_started_time))
        n_frames += 1
        frame_idx += 1
        t_sec = frame_idx / fps
    cap.release()
    return {
        "series_1p": series_1p, "series_2p": series_2p,
        "n_frames": n_frames, "match_started_hist": match_started_hist,
    }


def _max_run_duration(
    series: list[tuple[float, int]], value: int, t_lo: float, t_hi: float,
) -> tuple[float, bool]:
    """[t_lo,t_hi] 窓内で value が連続する最大区間長 (秒) と、窓境界で
    途切れているか (=真の長さ不明) を返す。"""
    best = 0.0
    censored = False
    i = 0
    n = len(series)
    while i < n:
        t, v = series[i]
        if t < t_lo or t > t_hi:
            i += 1
            continue
        if v != value:
            i += 1
            continue
        j = i
        while j + 1 < n and series[j + 1][1] == value and series[j + 1][0] <= t_hi:
            j += 1
        dur = series[j][0] - series[i][0]
        touches_lo = i == 0 or series[i - 1][0] < t_lo
        touches_hi = j == n - 1 or series[j + 1][0] > t_hi
        if dur > best:
            best = dur
            censored = touches_lo or touches_hi
        i = j + 1
    return best, censored


def run_group(name: str) -> dict:
    group = GROUPS[name]
    video_path = VIDEO_DIR / video_filename_of(group.video_id)
    fps, duration = probe_fps_duration(video_path)
    chunk_start_sec = CHUNK_OFFSET_FRACTIONS[group.chunk_idx] * duration
    start_sec = max(0.0, chunk_start_sec - LOOKBACK_SEC)
    end_sec = chunk_start_sec + CHUNK_SEC + END_MARGIN_SEC
    print(
        f"[{name}] video={video_path.name} fps={fps:.3f} dur={duration:.1f} "
        f"chunk_start={chunk_start_sec:.2f} continuous_start={start_sec:.2f} "
        f"end={end_sec:.2f} (window={end_sec - start_sec:.1f}s)"
    )
    t0 = time.monotonic()
    run = _run_continuous(video_path, fps, start_sec, end_sec, group.targets)
    elapsed = time.monotonic() - t0
    print(f"[{name}] 処理完了 n_frames={run['n_frames']} elapsed={elapsed:.1f}s")

    results = []
    for t in group.targets:
        series = (
            run["series_1p"] if t.side == "1P" else run["series_2p"]
        )[(t.r, t.c)]
        # chunkモードの破損窓 ± 2秒を対象窓とする (連続処理側での再現有無確認)。
        t_lo, t_hi = t.t_start - 2.0, t.t_end + 2.0
        wrong_dur, wrong_censored = _max_run_duration(series, t.wrong_value, t_lo, t_hi)
        correct_dur, _ = _max_run_duration(series, t.correct_value, t_lo, t_hi)
        wrong_frames_equiv = round(wrong_dur * EFFECTIVE_FPS, 2)
        reproduced = wrong_frames_equiv >= PERSIST_FRAME_THRESHOLD
        results.append({
            "sheet_id": t.sheet_id, "side": t.side, "r": t.r, "c": t.c,
            "wrong_value": t.wrong_value, "correct_value": t.correct_value,
            "chunk_mode_duration_sec": round(t.t_end - t.t_start, 4),
            "continuous_wrong_max_run_sec": round(wrong_dur, 4),
            "continuous_wrong_frames_equiv": wrong_frames_equiv,
            "continuous_reproduced_persistent": reproduced,
            "continuous_wrong_run_censored": wrong_censored,
            "continuous_correct_max_run_sec": round(correct_dur, 4),
        })
    n_reproduced = sum(1 for r in results if r["continuous_reproduced_persistent"])
    out = {
        "group": name, "video_id": group.video_id, "chunk_idx": group.chunk_idx,
        "fps": fps, "duration": duration, "chunk_start_sec": chunk_start_sec,
        "continuous_start_sec": start_sec, "continuous_end_sec": end_sec,
        "n_frames_processed": run["n_frames"], "elapsed_sec": round(elapsed, 1),
        "match_started_hist": run["match_started_hist"],
        "n_targets": len(results), "n_reproduced": n_reproduced,
        "cells": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{name}] 再現={n_reproduced}/{len(results)} -> {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS.keys()))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    names = list(GROUPS.keys()) if args.all else ([args.group] if args.group else [])
    if not names:
        ap.print_help()
        return
    for name in names:
        run_group(name)


if __name__ == "__main__":
    main()
