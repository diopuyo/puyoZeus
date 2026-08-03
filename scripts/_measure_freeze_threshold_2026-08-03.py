"""表示側「凍結検知」閾値の物理根拠導出 (2026-08-03 main指示、シーン逆算禁止)。

既存 measure_placement_speed_by_row_2026-08-03.py の通常設置イベント抽出
(66動画70npz、24.6万件、再実装しない) をそのまま再利用し、全段プールの
dt (STABLE→STABLE間隔) 分布の上側パーセンタイルから「通常プレイでは
まず起きない沈黙秒数」を導出する。この値を超える沈黙は「連鎖中(または
それに類する非STABLE事象)以外に説明がつかない」という物理的根拠として
表示側の凍結検知閾値に使う。
"""
from __future__ import annotations

import importlib

import numpy as np

mod = importlib.import_module("scripts.measure_placement_speed_by_row_2026-08-03")

PERCENTILES = [95, 99, 99.5, 99.9, 99.95, 99.99, 100]


def main() -> None:
    events = mod.collect_all_events(mod.NPZ_DIR)
    all_dt = np.array([e.dt_sec for e in events], dtype=float)
    print(f"\n=== 通常設置dt分布の上側パーセンタイル (全段プール n={len(all_dt)}) ===")
    for p in PERCENTILES:
        print(f"  p{p}: {np.percentile(all_dt, p):.3f}秒")

    # 段別 (row_index別) の最大値も確認 (最も遅い段でも通常プレイはどこまでか)
    print(f"\n=== 段別 dt 最大値・p99.9 (通常プレイの範囲を段別に確認) ===")
    dt_by_row: dict[int, list[float]] = {}
    for e in events:
        dt_by_row.setdefault(e.row_index, []).append(e.dt_sec)
    for row in sorted(dt_by_row.keys()):
        arr = np.array(dt_by_row[row])
        print(f"  row={row:2d}: n={len(arr):6d} max={arr.max():.3f}秒 p99.9={np.percentile(arr, 99.9):.3f}秒")


if __name__ == "__main__":
    main()
