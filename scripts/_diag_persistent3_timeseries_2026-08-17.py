"""持続誤認3件 (c23/c10/c109) の時系列計装 (2026-08-17)。本体コード変更なし。

目的:
  1. 対象セルの誤り区間が「他セルと同時に切り替わる (=イベント単位)」か
     「単独セルだけ (=セル単位ノイズ)」かを判定する。
  2. W10ガード (enable_landing_color_guard) の監視期限 (LANDING_VOTE_SEC=0.4秒)
     と実際の持続時間を突き合わせる。
  3. c109 は wrong_value=9 (おじゃま) なので、ガードの登録条件
     (`inferred_landing not in (EMPTY, UNKNOWN, COLOR_OJAMA)`) を計装確認する。

出力: data/verify/diag_persistent3_2026-08-17/timeseries_<sheet>.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
NPZ_DIR = _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_r2w10_2026-08-17"
OUT_DIR = _ROOT / "data" / "verify" / "diag_persistent3_2026-08-17"

TARGETS = {
    "007_c23_2P_f84251": {
        "npz": "c23_chunk1.npz", "video_id": "c23", "side": "2P", "frame_idx": 84251,
        "cells": [
            (7, 1), (8, 4), (9, 5), (10, 0), (10, 1), (10, 2), (10, 5), (11, 4), (12, 3), (12, 4),
        ],
        "wrong": 1, "correct": 5,
    },
    "009_c10_2P_f80448": {
        "npz": "c10_chunk1.npz", "video_id": "c10", "side": "2P", "frame_idx": 80448,
        "cells": [
            (2, 5), (3, 5), (4, 1), (4, 4), (5, 1), (5, 4), (6, 0), (7, 0), (7, 3),
            (8, 3), (8, 4), (10, 2), (11, 3), (11, 4), (12, 2),
        ],
        "wrong": 5, "correct": 2,
    },
    "000_c109_1P_f652064": {
        "npz": "c109_chunk2.npz", "video_id": "c109", "side": "1P", "frame_idx": 652064,
        "cells": [(4, 4)],
        "wrong": 9, "correct": 1,
    },
}


def load_series(npz_name: str, video_id: str, side: str) -> list[dict]:
    d = np.load(NPZ_DIR / npz_name, allow_pickle=True)
    n = len(d["frame_idx"])
    rows = []
    for i in range(n):
        vid = str(d["video_id"][i]).removeprefix("_hold_").removeprefix("video_")
        s = str(d["side"][i])
        if vid == video_id and s == side:
            rows.append({
                "frame_idx": int(d["frame_idx"][i]),
                "t_sec": float(d["t_sec"][i]),
                "grid": d["grids"][i],
            })
    rows.sort(key=lambda r: r["frame_idx"])
    return rows


def analyze(sheet_id: str, cfg: dict) -> dict:
    rows = load_series(cfg["npz"], cfg["video_id"], cfg["side"])
    anchor_idx = next((i for i, r in enumerate(rows) if r["frame_idx"] == cfg["frame_idx"]), None)
    assert anchor_idx is not None, f"anchor not found for {sheet_id}"

    # 各セルについて、誤り値(wrong)の区間の [lo,hi] を求める
    cell_runs = {}
    for (r, c) in cfg["cells"]:
        val = int(rows[anchor_idx]["grid"][r, c])
        lo = anchor_idx
        while lo - 1 >= 0 and int(rows[lo - 1]["grid"][r, c]) == val:
            lo -= 1
        hi = anchor_idx
        while hi + 1 < len(rows) and int(rows[hi + 1]["grid"][r, c]) == val:
            hi += 1
        cell_runs[f"r{r}c{c}"] = {
            "value_at_anchor": val,
            "t_start": rows[lo]["t_sec"], "t_end": rows[hi]["t_sec"],
            "frame_start": rows[lo]["frame_idx"], "frame_end": rows[hi]["frame_idx"],
            "duration_sec": rows[hi]["t_sec"] - rows[lo]["t_sec"],
            "lo_boundary": lo == 0, "hi_boundary": hi == len(rows) - 1,
        }

    # 同時性判定: 各セルの区間開始時刻(t_start)のばらつき
    starts = [v["t_start"] for v in cell_runs.values()]
    ends = [v["t_end"] for v in cell_runs.values()]
    spread_start = max(starts) - min(starts)
    spread_end = max(ends) - min(ends)

    # 前後の全盤面差分診断: anchor直前フレームと直後フレームで、
    # 対象外セルも含めて盤面全体が何セル変化したか (イベント規模の把握)
    lo_all = min(v["frame_start"] for v in cell_runs.values())
    before_idx = next((i for i, r in enumerate(rows) if r["frame_idx"] == lo_all), None)
    n_diff_at_flip = None
    if before_idx is not None and before_idx - 1 >= 0:
        g0 = rows[before_idx - 1]["grid"]
        g1 = rows[before_idx]["grid"]
        n_diff_at_flip = int(np.sum(g0 != g1))

    return {
        "sheet_id": sheet_id,
        "n_rows_in_series": len(rows),
        "anchor_frame_idx": cfg["frame_idx"],
        "n_target_cells": len(cfg["cells"]),
        "cell_runs": cell_runs,
        "spread_of_flip_start_sec": round(spread_start, 4),
        "spread_of_flip_end_sec": round(spread_end, 4),
        "n_cells_changed_board_wide_at_flip_frame": n_diff_at_flip,
        "note": "spread_start/end が小さい(<1フレーム相当)ほど単一イベントで一斉切替",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sheet_id, cfg in TARGETS.items():
        result = analyze(sheet_id, cfg)
        out_path = OUT_DIR / f"timeseries_{sheet_id}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"[{sheet_id}] spread_start={result['spread_of_flip_start_sec']}s "
              f"spread_end={result['spread_of_flip_end_sec']}s "
              f"board_wide_diff_at_flip={result['n_cells_changed_board_wide_at_flip_frame']} "
              f"-> {out_path}")


if __name__ == "__main__":
    main()
