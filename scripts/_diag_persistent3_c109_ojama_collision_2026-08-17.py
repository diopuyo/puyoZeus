"""件3 (c109 r4c4 おじゃま誤読) の物理検証 (2026-08-17)。本体コード変更なし。

対象フレーム前後の盤面全体を比較し、おじゃま一斉着地イベントに
既存の色ぷよセルが巻き込まれて上書きされている (= 列4 で二重着地・
既存puyoの上書きという物理違反) ことを計装確認する。

出力: data/verify/diag_persistent3_2026-08-17/c109_ojama_collision.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
NPZ_PATH = _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_r2w10_2026-08-17" / "c109_chunk2.npz"
OUT_DIR = _ROOT / "data" / "verify" / "diag_persistent3_2026-08-17"

TARGET_FRAME = 652064


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = np.load(NPZ_PATH, allow_pickle=True)
    n = len(d["frame_idx"])
    rows = []
    for i in range(n):
        vid = str(d["video_id"][i]).removeprefix("_hold_").removeprefix("video_")
        side = str(d["side"][i])
        if vid == "c109" and side == "1P":
            rows.append({
                "frame_idx": int(d["frame_idx"][i]),
                "t_sec": float(d["t_sec"][i]),
                "grid": d["grids"][i],
            })
    rows.sort(key=lambda r: r["frame_idx"])
    idx = next(i for i, r in enumerate(rows) if r["frame_idx"] == TARGET_FRAME)
    g_before = rows[idx - 1]["grid"]
    g_after = rows[idx]["grid"]

    diff_cells = []
    for r in range(g_before.shape[0]):
        for c in range(g_before.shape[1]):
            b, a = int(g_before[r, c]), int(g_after[r, c])
            if b != a:
                diff_cells.append({
                    "r": r, "c": c, "before": b, "after": a,
                    "is_legit_landing(empty->ojama)": (b == 0 and a == 9),
                    "is_overwrite_of_existing_puyo": (b not in (0, 9) and a == 9),
                })

    # 列4 (c=4) の縦列を before/after 両方出力し、物理的整合性 (連続性) を確認
    col4_before = [int(g_before[r, 4]) for r in range(g_before.shape[0])]
    col4_after = [int(g_after[r, 4]) for r in range(g_after.shape[0])]

    result = {
        "target_frame_idx": TARGET_FRAME,
        "t_before": rows[idx - 1]["t_sec"], "t_after": rows[idx]["t_sec"],
        "frame_before": rows[idx - 1]["frame_idx"], "frame_after": rows[idx]["frame_idx"],
        "diff_cells": diff_cells,
        "n_legit_landing": sum(1 for e in diff_cells if e["is_legit_landing(empty->ojama)"]),
        "n_overwrite_of_existing_puyo": sum(1 for e in diff_cells if e["is_overwrite_of_existing_puyo"]),
        "col4_before_row0to12": col4_before,
        "col4_after_row0to12": col4_after,
        "note": (
            "col4 は before で row2/row3 が赤(1)で連続占有中に row1 が空だった。"
            "after では row1 が正しく空->おじゃまに着地した一方、row4 の既存赤(1)も"
            "同時におじゃま(9)へ上書きされている。row2/row3 は赤のまま不変="
            "着地バッチが同一列に二重書き込みし、2回目の着地行計算が"
            "既存puyoの位置を考慮せず上書きした物理違反 (浮きおじゃま/非連続スタック)。"
        ),
    }
    out_path = OUT_DIR / "c109_ojama_collision.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] -> {out_path}")
    print(f"legit_landing={result['n_legit_landing']} overwrite_of_existing_puyo={result['n_overwrite_of_existing_puyo']}")
    print("col4 before:", col4_before)
    print("col4 after :", col4_after)


if __name__ == "__main__":
    main()
