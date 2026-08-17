"""mc_counter_estimator.py のロールアウト内訳を手動計装で計測する
(2026-08-13、Rust拡張載せ替えタスクの Step1)。

cProfile は使わない (project規約: 速度計測は time.perf_counter の手動計装、
cProfile は overhead が大きく反復比較に不向き)。

計測対象:
    - src.chain.ChainSimulator.simulate の呼び出し回数・累積時間
    - src.indicators_v2.current_max_chain の呼び出し回数・累積時間
    - src.indicators_v2.potential_fire_power の呼び出し回数・累積時間
    - scripts.mc_counter_estimator._enumerate_placements (indicators_v2 側)
      の呼び出し回数・累積時間

計測方法: 各関数をラップして (呼び出し回数, 累積秒) を集計するデコレータを
一時的にモンキーパッチする (本体コードは変更しない、計測専用スクリプト)。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import src.indicators_v2 as iv
from src.board import Board
from scripts import mc_counter_estimator as mce

_DATA_PATH = Path("data/verify/fps_stride_ab_2026-08-12/review_demo_stride2.npz")

# 計装対象関数名 -> [呼び出し回数, 累積秒]
_STATS: "dict[str, list[float]]" = {}


def _wrap(name: str, func):
    """func をラップし _STATS[name] = [count, total_sec] に集計する。"""
    _STATS[name] = [0.0, 0.0]

    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        _STATS[name][0] += 1.0
        _STATS[name][1] += time.perf_counter() - t0
        return result
    return wrapped


def _load_large_chain_board() -> Board:
    """review_demo_stride2.npz から色ぷよ数指定パーセンタイルの盤面を選ぶ
    (MCE_PROFILE_PERCENTILE env で調整、既定100=最多=大連鎖相当)。"""
    import os
    percentile = float(os.environ.get("MCE_PROFILE_PERCENTILE", "100"))
    data = np.load(_DATA_PATH, allow_pickle=True)
    grids = data["grids"]
    color_counts = np.sum((grids != 0) & (grids != 9), axis=(1, 2))
    target = np.percentile(color_counts, percentile)
    idx = int(np.argmin(np.abs(color_counts - target)))
    board = Board()
    board._grid = grids[idx].astype(np.uint8)
    return board


def main() -> None:
    """native系呼び出し (simulate_chain/enumerate_placements/drop_one_puyo) を
    計装した状態で estimate_counter_distribution(use_native=True) を実行し、
    内訳を表示する (2026-08-13、native載せ替え後のプロファイル)。
    """
    import os
    from src import puyo_core_bridge as bridge

    board = _load_large_chain_board()
    print(f"色ぷよ数: {int(np.sum((board._grid != 0) & (board._grid != 9)))}", flush=True)

    orig_simulate_chain = bridge.simulate_chain
    orig_enumerate_placements = bridge.enumerate_placements
    orig_simulate_after_drops = bridge.simulate_after_drops

    wrapped_simulate_chain = _wrap("native.simulate_chain", orig_simulate_chain)
    wrapped_enumerate_placements = _wrap("native.enumerate_placements", orig_enumerate_placements)
    wrapped_simulate_after_drops = _wrap("native.simulate_after_drops", orig_simulate_after_drops)
    bridge.simulate_chain = wrapped_simulate_chain
    bridge.enumerate_placements = wrapped_enumerate_placements
    bridge.simulate_after_drops = wrapped_simulate_after_drops
    mce._native_simulate_chain = wrapped_simulate_chain
    mce._native_enumerate_placements = wrapped_enumerate_placements
    mce._native_simulate_after_drops = wrapped_simulate_after_drops

    try:
        n_rollouts = int(os.environ.get("MCE_PROFILE_N_ROLLOUTS", "50"))
        budget = float(os.environ.get("MCE_PROFILE_BUDGET_SEC", "5.0"))
        use_native = os.environ.get("MCE_PROFILE_USE_NATIVE", "1") == "1"
        t0 = time.perf_counter()
        dist = mce.estimate_counter_distribution(
            board, time_budget_sec=budget, n_rollouts=n_rollouts, use_native=use_native,
        )
        total_sec = time.perf_counter() - t0
    finally:
        bridge.simulate_chain = orig_simulate_chain
        bridge.enumerate_placements = orig_enumerate_placements
        bridge.simulate_after_drops = orig_simulate_after_drops
        mce._native_simulate_chain = orig_simulate_chain
        mce._native_enumerate_placements = orig_enumerate_placements
        mce._native_simulate_after_drops = orig_simulate_after_drops

    print(f"\n合計時間: {total_sec:.3f} 秒 (n_rollouts=50, time_budget=5.0秒)")
    print(f"mean_hands_used={dist.mean_hands_used:.2f}")
    print("\n内訳:")
    for name, (count, sec) in sorted(_STATS.items(), key=lambda kv: -kv[1][1]):
        pct = 100.0 * sec / total_sec if total_sec > 0 else 0.0
        print(f"  {name:30s} count={int(count):8d} sec={sec:8.3f} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
