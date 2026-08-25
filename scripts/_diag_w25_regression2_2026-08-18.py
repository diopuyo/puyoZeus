"""W25第3弾 (会計整合フィルタ) の新規悪化2シート (c10_2P/c109_2P) 機構特定用計装。

src/ 無改修。RecognitionPipeline._apply_ojama_write_accounting_filter と
_step_side をランタイム monkeypatch し、対象セルの per-frame トレースを
記録する (フィルタ前 cnn 値 / stable_color_memory / フィルタ後 confirmed 値)。

対象:
    c10  2P frame_idx=15517 (t=258.617s, chunk0 start=237.76)  cells: (8,1),(10,2)
    c109 2P frame_idx=652546(t=10875.767s, chunk2 start=10854.4) cells: (3,2)

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_w25_regression2_2026-08-18
"""
from __future__ import annotations

import json
from pathlib import Path

VIDEO_DIR = Path.home() / "frames"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "verify" / "diag_w25_regression2_2026-08-18"

TARGETS = [
    {
        "name": "c10_2P",
        "video": VIDEO_DIR / "video_c10.mp4",
        "start_sec": 237.76,
        "max_sec": 30.0,
        "side": "2P",
        "cells": [(8, 1), (10, 2)],
        "gt_frame_idx": 15517,
    },
    {
        "name": "c109_2P",
        "video": VIDEO_DIR / "video_c109.mp4",
        "start_sec": 10854.4,
        "max_sec": 30.0,
        "side": "2P",
        "cells": [(3, 2)],
        "gt_frame_idx": 652546,
    },
]


def run_one(target: dict, *, enable_guard: bool) -> list[dict]:
    """1 target を 1 構成 (guard ON/OFF) で処理し per-frame trace を返す。"""
    import src.recognition_pipeline as rp

    trace: list[dict] = []
    side = target["side"]
    cells = target["cells"]

    orig_apply_filter = rp.RecognitionPipeline._apply_ojama_write_accounting_filter
    orig_step_side = rp.RecognitionPipeline._step_side

    filter_calls: list[dict] = []

    def wrapped_apply_filter(self, side_arg, cnn_board, forecast):  # noqa: ANN001
        memory = (
            self._stable_color_memory_1p if side_arg == "1P"
            else self._stable_color_memory_2p
        )
        pre = {
            f"r{r}c{c}": int(cnn_board.get(r, c)) for (r, c) in cells
        } if side_arg == side else None
        mem_before = {
            f"r{r}c{c}": memory.get((r, c)) for (r, c) in cells
        } if side_arg == side else None
        out = orig_apply_filter(self, side_arg, cnn_board, forecast)
        if side_arg == side:
            post = {f"r{r}c{c}": int(out.get(r, c)) for (r, c) in cells}
            filter_calls.append({
                "pre_cnn": pre, "mem_before": mem_before, "post_cnn": post,
                "forecast": forecast,
            })
        return out

    def wrapped_step_side(self, side_arg, frame_idx, time_sec, *args, **kwargs):  # noqa: ANN001
        res = orig_step_side(self, side_arg, frame_idx, time_sec, *args, **kwargs)
        if side_arg == side:
            memory = (
                self._stable_color_memory_1p if side_arg == "1P"
                else self._stable_color_memory_2p
            )
            row = {
                "frame_idx": frame_idx, "t_sec": time_sec,
                "state": res.state.name if res.state is not None else None,
            }
            for (r, c) in cells:
                key = f"r{r}c{c}"
                row[f"{key}_cnn"] = int(res.cnn_board.get(r, c)) if res.cnn_board is not None else None
                row[f"{key}_confirmed"] = (
                    int(res.confirmed_board.get(r, c))
                    if res.confirmed_board is not None else None
                )
                row[f"{key}_mem"] = memory.get((r, c))
            # 直前の filter_calls をこのフレームに紐付け (無ければ None)。
            row["filter_call"] = filter_calls.pop(0) if filter_calls else None
            trace.append(row)
        return res

    rp.RecognitionPipeline._apply_ojama_write_accounting_filter = wrapped_apply_filter
    rp.RecognitionPipeline._step_side = wrapped_step_side
    try:
        from scripts.collect_boards_lean import collect_lean
        tmp_npz = OUT_DIR / f"_tmp_{target['name']}_{'on' if enable_guard else 'off'}.npz"
        collect_lean(
            video_path=target["video"],
            out_npz=tmp_npz,
            start_sec=target["start_sec"],
            max_sec=target["max_sec"],
            capture_next=True,
            enable_chain_tracker=True,
            enable_effect_gate=True,
            enable_burst_guard_v2=True,
            enable_transition_merge_guard=True,
            burst_gate_open_threshold=0.954,
            enable_hidden_row_burst_guard=True,
            enable_match_transition_debounce=True,
            enable_ojama_fall_placement_override=True,
            enable_patch_fp_hsv_guard=True,
            enable_floating_gap_restore=True,
            enable_landing_color_guard=True,
            enable_override_color_guard=True,
            enable_ojama_column_stack_fix=True,
            enable_next_history_starvation_fix=True,
            enable_ojama_write_accounting_guard=enable_guard,
        )
        if tmp_npz.exists():
            tmp_npz.unlink()
    finally:
        rp.RecognitionPipeline._apply_ojama_write_accounting_filter = orig_apply_filter
        rp.RecognitionPipeline._step_side = orig_step_side
    return trace


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        print(f"=== {target['name']} ===")
        trace_off = run_one(target, enable_guard=False)
        trace_on = run_one(target, enable_guard=True)
        out_path = OUT_DIR / f"trace_{target['name']}.json"
        out_path.write_text(json.dumps({
            "target": {k: (str(v) if isinstance(v, Path) else v) for k, v in target.items()},
            "trace_off": trace_off,
            "trace_on": trace_on,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  n_frames off={len(trace_off)} on={len(trace_on)} -> {out_path}")


if __name__ == "__main__":
    main()
