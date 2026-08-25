"""c13系19件+c22系1件「真の持続誤認」再検証 (2026-08-17)。

前任 (`scripts/_diag_c21_burst_2026-08-17.py`) が c21系7件で確立した手法
(書き込みframe特定 -> 実画面照合 -> 区間中の真値変化チェック) を c13/c22 に適用する。

本体コード変更なし (計装は全て外部から instance 属性を読むのみ、
src/ への一時挿入も行わない)。

対象: data/verify/diag_w23_continuous_2026-08-17/c13_chunk0.json (12セル),
      c13_chunk1.json (2セル), c22_chunk1.json (1セル)。
      (上記 json の GROUPS 定義そのものを「重複ラベルシート013/026を統合した
      一意セル集合」として引き継ぐ。KNOWN_WEAKNESSES.md の見出し数字
      「c13=19件」は 013/026 の2枚のラベルシートに同一セルが重複計上された
      値で、一意な盤面セルは 12(chunk0)+2(chunk1)=14 個であることを本計装で
      確認した上で report に明記する)。

start_sec/end_sec は c13_chunk0.json / c13_chunk1.json / c22_chunk1.json の
match_started_hist (既存計装ログの副産物) から、対象窓を含む直近の試合開始/
終了イベントの前後に十分な安全マージンを取って短縮 (300秒フルlookbackは
不要、根拠はコード内コメント参照)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_c13c22_recheck_2026-08-17 --run c13_chunk0
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_c13c22_recheck_2026-08-17 --all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402
from src.placement_inferrer import _extract_cell_patch_from_frame  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO_DIR = Path.home() / "frames"
OUT_DIR = _ROOT / "data" / "verify" / "diag_c13c22_recheck_2026-08-17"
FRAMES_DIR = OUT_DIR / "frames"


@dataclass(frozen=True)
class TargetCell:
    sheet_id: str
    side: str
    r: int
    c: int
    wrong: int
    correct: int
    t_lo: float  # 密ログ・変化検出を行う対象窓 (元 chunkモード実測±マージン)
    t_hi: float


@dataclass(frozen=True)
class RunSpec:
    video_id: str
    start_sec: float
    end_sec: float
    cells: tuple[TargetCell, ...]
    note: str


# start_sec/end_sec: data/verify/diag_w23_continuous_2026-08-17/{name}.json の
# match_started_hist から、対象窓の直近試合開始/終了イベントの前後に
# 安全マージンを取って算出 (c21_burst と同方式)。
RUNS: dict[str, RunSpec] = {
    "c13_chunk0": RunSpec(
        video_id="c13", start_sec=210.0, end_sec=296.0,
        cells=(
            TargetCell("013/026_c13_2P", "2P", 2, 2, 4, 9, 288.5, 293.0),
            TargetCell("013/026_c13_2P", "2P", 3, 2, 4, 9, 288.5, 293.0),
            TargetCell("013/026_c13_2P", "2P", 9, 1, 9, 4, 288.5, 293.0),
            TargetCell("026_c13_2P", "2P", 9, 3, 3, 9, 277.5, 294.5),
            TargetCell("013/026_c13_2P", "2P", 9, 4, 9, 4, 288.5, 293.0),
            TargetCell("013/026_c13_2P", "2P", 9, 5, 9, 4, 288.5, 293.0),
            TargetCell("013/026_c13_2P", "2P", 10, 1, 9, 3, 288.5, 293.0),
            TargetCell("013_c13_2P", "2P", 10, 3, 9, 2, 288.5, 293.0),
            TargetCell("013/026_c13_2P", "2P", 10, 4, 9, 2, 288.5, 293.0),
            TargetCell("013/026_c13_2P", "2P", 10, 5, 9, 5, 288.5, 293.0),
            TargetCell("013_c13_2P", "2P", 11, 0, 9, 3, 288.5, 293.0),
            TargetCell("013/026_c13_2P", "2P", 11, 4, 9, 4, 288.5, 293.0),
        ),
        note=(
            "直近試合開始 t=225.82 (実測、c13_chunk0.json match_started_hist)。"
            "start_sec はその16秒前。対象窓は元jsonのt_start/t_endを内包するよう拡張。"
        ),
    ),
    "c13_chunk1": RunSpec(
        video_id="c13", start_sec=1490.0, end_sec=1530.0,
        cells=(
            TargetCell("033_c13_2P_f91334", "2P", 4, 2, 2, 0, 1520.0, 1529.0),
            TargetCell("033_c13_2P_f91334", "2P", 5, 2, 1, 0, 1520.0, 1529.0),
        ),
        note=(
            "対象窓直前に試合終了(t~1494-1497、-1.0観測)+新試合開始 t=1504.12"
            "(実測) が挟まる。start_sec はその前の試合(開始1425.2)がまだ"
            "続いている t=1490 から通し処理し、試合境界の遷移そのものも計装する。"
        ),
    ),
    "c22_chunk1": RunSpec(
        video_id="c22", start_sec=1792.0, end_sec=1822.0,
        cells=(
            TargetCell("043_c22_1P_f108676", "1P", 12, 2, 5, 2, 1809.0, 1821.0),
        ),
        note=(
            "直近試合開始 t=1808.52 (実測、c22_chunk1.json match_started_hist)。"
            "start_sec はその前の試合終了(t~1798-1802)を含む t=1792 から通し処理。"
        ),
    ),
    "c22_chunk1_ext": RunSpec(
        # c22_chunk1 の初回計装で t=1808.75-1821 が「マスターリーグ 30本先取」
        # ラウンド間イントロ画面 (実盤面でない、装飾ぷよ紫アイコン単体) と判明。
        # 実ゲームプレイに戻り正解値2(青)へ解決するかを追加確認する延長窓。
        video_id="c22", start_sec=1820.0, end_sec=1850.0,
        cells=(
            TargetCell("043_c22_1P_f108676", "1P", 12, 2, 5, 2, 1820.0, 1850.0),
        ),
        note="c22_chunk1 の続き (延長窓、初回窓の状態を引き継がない独立起動)。",
    ),
}


def video_path_of(video_id: str) -> Path:
    return VIDEO_DIR / f"video_{video_id}.mp4"


def probe_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def build_pipeline() -> RecognitionPipeline:
    """統一測定 構成F と同一構成 (force_in_match=False)。"""
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        load_next_detector=True,
        temporal_smoothing=1,
        force_in_match=False,
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


def _hsv_classify_at(
    pipeline: RecognitionPipeline, frame_bgr: np.ndarray, side: str, r: int, c: int,
) -> int | None:
    """独立 HSV-only 分類 (cnn_board とは別経路、cross-check 用)。"""
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    classifier = getattr(pipeline._reader, "_classifier", None)
    hsv_clf = getattr(classifier, "_hsv", classifier)
    if hsv_clf is None or not hasattr(hsv_clf, "classify"):
        return None
    patch = _extract_cell_patch_from_frame(frame_bgr, region, r, c)
    if patch is None or patch.size == 0:
        return None
    try:
        return int(hsv_clf.classify(patch))
    except Exception:
        return None


def _hsv_mean_at(frame_bgr: np.ndarray, side: str, r: int, c: int) -> list[float] | None:
    """該当セルパッチの平均 HSV (H,S,V) を返す (実測材料)。"""
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    patch = _extract_cell_patch_from_frame(frame_bgr, region, r, c)
    if patch is None or patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float64)
    return [round(float(hsv[..., i].mean()), 2) for i in range(3)]


def run_spec(name: str, save_frames: bool) -> dict:
    spec = RUNS[name]
    video_path = video_path_of(spec.video_id)
    fps = probe_fps(video_path)
    print(f"[{name}] video={video_path.name} fps={fps:.3f} "
          f"start={spec.start_sec} end={spec.end_sec} note={spec.note}")

    cap = cv2.VideoCapture(str(video_path))
    start_frame = int(spec.start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pipeline = build_pipeline()

    per_cell_dense: dict[str, list[dict]] = {
        f"{t.side}_r{t.r}c{t.c}": [] for t in spec.cells
    }
    prev_conf: dict[str, int | None] = {k: None for k in per_cell_dense}
    transitions: dict[str, list[dict]] = {k: [] for k in per_cell_dense}
    match_started_hist: list[tuple[float, float]] = []

    frame_idx = start_frame
    t_sec = spec.start_sec
    n_frames = 0
    t0 = time.monotonic()
    saved_frame_ranges: set[int] = set()

    while t_sec < spec.end_sec:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        res = pipeline.update(frame_idx, t_sec, frame)

        if n_frames % 300 == 0:
            match_started_hist.append((round(t_sec, 3), pipeline._match_active_started_time))

        for t in spec.cells:
            key = f"{t.side}_r{t.r}c{t.c}"
            side_res = res.p1 if t.side == "1P" else res.p2
            cnn_val = int(side_res.cnn_board.get(t.r, t.c)) if side_res.cnn_board is not None else None
            conf_val = (
                int(side_res.confirmed_board.get(t.r, t.c))
                if side_res.confirmed_board is not None else None
            )
            state_name = getattr(side_res.state, "name", str(side_res.state))

            watch_list = (
                pipeline._landing_color_watch_1p if t.side == "1P"
                else pipeline._landing_color_watch_2p
            )
            watch_entry = next(
                (d for (cell, d) in watch_list if cell == (t.r, t.c)), None,
            )
            history = (
                pipeline._stable_cnn_history_1p if t.side == "1P"
                else pipeline._stable_cnn_history_2p
            ).get((t.r, t.c), [])
            hist_counter = Counter(history)
            hist_most_common, hist_mc_count = (
                hist_counter.most_common(1)[0] if history else (None, 0)
            )
            hist_ratio = hist_mc_count / len(history) if history else 0.0

            grace_state = (
                pipeline._landing_grace_1p if t.side == "1P"
                else pipeline._landing_grace_2p
            )
            chain_exit_until = (
                pipeline._chain_exit_until_1p if t.side == "1P"
                else pipeline._chain_exit_until_2p
            )

            changed = conf_val is not None and conf_val != prev_conf[key]
            in_dense_window = t.t_lo <= t_sec <= t.t_hi

            if changed or in_dense_window:
                hsv_val = _hsv_classify_at(pipeline, frame, t.side, t.r, t.c)
                hsv_mean = _hsv_mean_at(frame, t.side, t.r, t.c)
            else:
                hsv_val = None
                hsv_mean = None

            if changed:
                transitions[key].append({
                    "frame_idx": frame_idx, "t_sec": round(t_sec, 4),
                    "prev_conf": prev_conf[key], "new_conf": conf_val,
                    "cnn_val_this_frame": cnn_val,
                    "state": state_name,
                    "hist_len": len(history), "hist_most_common": hist_most_common,
                    "hist_ratio": round(hist_ratio, 3),
                    "hist_values": list(history),
                    "watch_entry_deadline": watch_entry,
                    "grace_active": grace_state is not None,
                    "chain_exit_warmup_active": t_sec < chain_exit_until,
                    "hsv_only_classify": hsv_val,
                    "hsv_mean_h_s_v": hsv_mean,
                })
                if save_frames and frame_idx not in saved_frame_ranges:
                    _save_evidence_frames(video_path, fps, frame_idx, t, name)
                    saved_frame_ranges.add(frame_idx)
                prev_conf[key] = conf_val

            if in_dense_window:
                per_cell_dense[key].append({
                    "frame_idx": frame_idx, "t_sec": round(t_sec, 4),
                    "cnn_val": cnn_val, "conf_val": conf_val, "state": state_name,
                    "hist_len": len(history), "hist_ratio": round(hist_ratio, 3),
                    "watch_entry_deadline": watch_entry,
                    "hsv_only_classify": hsv_val,
                    "hsv_mean_h_s_v": hsv_mean,
                })

        n_frames += 1
        frame_idx += 1
        t_sec = frame_idx / fps

    cap.release()
    elapsed = time.monotonic() - t0
    print(f"[{name}] 処理完了 n_frames={n_frames} elapsed={elapsed:.1f}s")

    out = {
        "run": name, "video_id": spec.video_id, "fps": fps,
        "start_sec": spec.start_sec, "end_sec": spec.end_sec,
        "n_frames_processed": n_frames, "elapsed_sec": round(elapsed, 1),
        "match_started_hist": match_started_hist,
        "cells": [
            {
                "sheet_id": t.sheet_id, "side": t.side, "r": t.r, "c": t.c,
                "wrong": t.wrong, "correct": t.correct,
                "dense_window": [t.t_lo, t.t_hi],
                "transitions": transitions[f"{t.side}_r{t.r}c{t.c}"],
                "dense_series": per_cell_dense[f"{t.side}_r{t.r}c{t.c}"],
            }
            for t in spec.cells
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{name}] -> {out_path}")
    return out


def _save_evidence_frames(
    video_path: Path, fps: float, center_frame_idx: int,
    target: TargetCell, run_name: str,
) -> None:
    """center_frame_idx の前後数フレームを、対象セルに枠を描いた PNG として保存する。

    現在の再生位置 (cap) を退避せずに新しい VideoCapture を開いて seek するため、
    メイン処理の連続性 (frame_idx/t_sec) には影響しない。
    """
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    region = DEFAULT_P1_REGION if target.side == "1P" else DEFAULT_P2_REGION
    cap2 = cv2.VideoCapture(str(video_path))
    for delta in range(-3, 4):
        f_idx = center_frame_idx + delta
        if f_idx < 0:
            continue
        cap2.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap2.read()
        if not ok:
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        x1, y1, x2, y2 = region.cell_sample_rect(target.r, target.c)
        vis = frame.copy()
        cv2.rectangle(vis, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 255), 2)
        cv2.rectangle(
            vis, (region.x, region.y),
            (region.x + region.width, region.y + region.height), (0, 255, 255), 1,
        )
        fname = (
            f"{run_name}_{target.sheet_id.replace('/', '-')}_{target.side}_"
            f"r{target.r}c{target.c}_f{f_idx}_d{delta:+d}.png"
        )
        cv2.imwrite(str(FRAMES_DIR / fname), vis)
    cap2.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(RUNS.keys()))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-frames", action="store_true", help="証拠フレームPNG保存をskip")
    args = ap.parse_args()
    names = list(RUNS.keys()) if args.all else ([args.run] if args.run else [])
    if not names:
        ap.print_help()
        return
    for name in names:
        run_spec(name, save_frames=not args.no_frames)


if __name__ == "__main__":
    main()
