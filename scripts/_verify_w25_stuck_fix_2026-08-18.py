"""W25固着対策 (2026-08-18) の直接効果確認: c10/c109 永久固着の再トレース。

`data/verify/diag_w25_regression2_2026-08-18/trace_c10_2P.json` /
`trace_c109_2P.json` の `target` フィールド (video/start_sec/max_sec/side/
cells/gt_frame_idx) をそのまま再利用し、同一箇所を現行実装 (タイムアウト
機構つき) で再トレースする。

各対象セルについて、生CNN観測値・直近安定色メモリ・フィルタ後confirmed値を
毎フレーム記録し、以下を確認する:
  - 固着解消: gt_frame_idx 付近で正規の9着弾が生じた後、
    OJAMA_REJECT_TIMEOUT_SEC (1.5秒) 以内に confirmed が9を受理するか
    (永久固着していないか)。
  - 固着再現 (対照): タイムアウト機構を無効化した場合 (旧第3弾のみ) は
    ウィンドウ終了まで固着し続けるか (対照実験、固着が実在したことの確認)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._verify_w25_stuck_fix_2026-08-18
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
REGRESSION_DIR = _ROOT / "data" / "verify" / "diag_w25_regression2_2026-08-18"
OUT_PATH = REGRESSION_DIR / "w25_stuck_fix_verify.json"

TARGETS: list[dict] = [
    json.loads((REGRESSION_DIR / "trace_c10_2P.json").read_text(encoding="utf-8"))["target"],
    json.loads((REGRESSION_DIR / "trace_c109_2P.json").read_text(encoding="utf-8"))["target"],
]


def build_pipeline(*, disable_timeout: bool = False):  # noqa: ANN201
    """構成F + enable_ojama_write_accounting_guard=True。

    disable_timeout=True の場合、タイムアウト定数を実質無効化した状態
    (対照実験用、旧第3弾のみ相当) で構築する。src/ の値そのものは変更
    せず、pipeline 構築後に monkeypatch でこのプロセス内だけ無効化する
    (他プロセス・他テストに影響しない)。
    """
    from src.recognition_pipeline import RecognitionPipeline
    pipe = RecognitionPipeline.load_default(
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
        enable_ojama_write_accounting_guard=True,
    )
    if disable_timeout:
        import src.ojama_write_accounting as owa
        # 対照実験専用: このプロセスに限り実質無限大にしてタイムアウト
        # 機構を無効化する (旧第3弾=固着したままの挙動を再現)。
        owa.OJAMA_REJECT_TIMEOUT_SEC = 1e9
    return pipe


def trace_target(target: dict, *, disable_timeout: bool) -> dict:
    video_path = Path(target["video"])
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_sec = target["start_sec"]
    max_sec = target["max_sec"]
    side = target["side"]
    cells = [tuple(c) for c in target["cells"]]

    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pipeline = build_pipeline(disable_timeout=disable_timeout)

    frame_idx = start_frame
    t_sec = start_sec
    n_frames = 0
    series: dict[str, list[dict]] = {f"r{r}c{c}": [] for (r, c) in cells}
    while t_sec < start_sec + max_sec:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        res = pipeline.update(frame_idx, t_sec, frame)
        side_res = res.p1 if side == "1P" else res.p2
        for (r, c) in cells:
            key = f"r{r}c{c}"
            cnn_v = int(side_res.cnn_board.get(r, c)) if side_res.cnn_board is not None else None
            conf_v = (
                int(side_res.confirmed_board.get(r, c))
                if side_res.confirmed_board is not None else None
            )
            series[key].append({
                "frame_idx": frame_idx, "t_sec": round(t_sec, 4),
                "cnn": cnn_v, "confirmed": conf_v,
            })
        n_frames += 1
        frame_idx += 1
        t_sec = frame_idx / fps

    cap.release()
    return {"n_frames": n_frames, "series": series}


def _analyze_cell(cell_series: list[dict], gt_frame_idx: int) -> dict:
    """gt_frame_idx 以降で confirmed が最初に 9 になった frame までの
    遅延 (秒) を計算する。ウィンドウ終了まで9にならなければ 固着 と判定。"""
    gt_entry = next((e for e in cell_series if e["frame_idx"] >= gt_frame_idx), None)
    if gt_entry is None:
        return {"status": "gt_frame_out_of_window"}
    accepted = next(
        (e for e in cell_series if e["frame_idx"] >= gt_frame_idx and e["confirmed"] == 9),
        None,
    )
    if accepted is None:
        last = cell_series[-1]
        return {
            "status": "STUCK (permanent, window末尾まで9を受理せず)",
            "last_confirmed": last["confirmed"], "last_t_sec": last["t_sec"],
        }
    delay_sec = accepted["t_sec"] - gt_entry["t_sec"]
    return {
        "status": "RESOLVED", "delay_sec": round(delay_sec, 3),
        "accept_frame_idx": accepted["frame_idx"], "accept_t_sec": accepted["t_sec"],
    }


def main() -> None:
    results = {}
    for target in TARGETS:
        name = target["name"]
        print(f"[{name}] video={Path(target['video']).name} start={target['start_sec']} "
              f"max_sec={target['max_sec']} cells={target['cells']}")

        trace_stuck = trace_target(target, disable_timeout=True)
        trace_fixed = trace_target(target, disable_timeout=False)

        entry = {"name": name, "gt_frame_idx": target["gt_frame_idx"], "cells": {}}
        for (r, c) in target["cells"]:
            key = f"r{r}c{c}"
            stuck_analysis = _analyze_cell(trace_stuck["series"][key], target["gt_frame_idx"])
            fixed_analysis = _analyze_cell(trace_fixed["series"][key], target["gt_frame_idx"])
            entry["cells"][key] = {
                "stuck_config(no_timeout)": stuck_analysis,
                "fixed_config(with_timeout)": fixed_analysis,
            }
            print(f"  {key}: 対照(タイムアウト無効)={stuck_analysis} / "
                  f"修正後(タイムアウト有効)={fixed_analysis}")
        results[name] = entry

    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] -> {OUT_PATH}")


if __name__ == "__main__":
    main()
