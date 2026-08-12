"""mc_counter_estimator.py の Rust ネイティブ拡張載せ替え、旧(純Python) vs
新(native) の速度比較ベンチ (2026-08-13)。

cProfile は使わない (project規約: time.perf_counter の手動計装、反復必須)。
`use_native=False` で純Python経路 (旧実装相当)、`use_native=True` (既定)
で native puyo_core 経路を、同一盤面・同一 time_budget・同一 n_rollouts
で N_REPEATS 回ずつ計測し、中央値で比較する。

実行条件 (task指示): WSL側で他ジョブ (148収集等) が走行中の場合があるため
nice -n 19 かつ単一プロセスで実行すること (計測値に負荷の影響がある旨を
呼び出し側が注記すること)。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.board import Board
from src.puyo_core_bridge import NATIVE_AVAILABLE
from scripts.mc_counter_estimator import estimate_counter_distribution

_DATA_PATH = Path("data/verify/fps_stride_ab_2026-08-12/review_demo_stride2.npz")


def _env_int(name: str, default: int) -> int:
    import os
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    import os
    return float(os.environ.get(name, str(default)))


# 本番既定 (MC_COUNTER_DEFAULT_N_ROLLOUTS=200) は旧実装 (legacy) では極端に
# 大きい盤面で1回あたり1時間規模になり反復計測が非現実的なため、native/legacy
# を別々の n_rollouts で計測できるよう環境変数で分離する
# (task 実行条件: nice -n 19 単一プロセス、WSL側の148収集ジョブと同時走行)。
_N_REPEATS: int = _env_int("BENCH_N_REPEATS", 10)
_TIME_BUDGET_SEC: float = _env_float("BENCH_BUDGET_SEC", 5.0)
_N_ROLLOUTS_NATIVE: int = _env_int("BENCH_N_ROLLOUTS_NATIVE", 200)
_N_ROLLOUTS_LEGACY: int = _env_int("BENCH_N_ROLLOUTS_LEGACY", 200)


def _load_board_at_percentile(percentile: float) -> Board:
    """review_demo_stride2.npz から色ぷよ数指定パーセンタイルの実盤面を選ぶ。"""
    data = np.load(_DATA_PATH, allow_pickle=True)
    grids = data["grids"]
    color_counts = np.sum((grids != 0) & (grids != 9), axis=(1, 2))
    target = np.percentile(color_counts, percentile)
    idx = int(np.argmin(np.abs(color_counts - target)))
    board = Board()
    board._grid = grids[idx].astype(np.uint8)
    return board


def _time_n_calls(
    board: Board, use_native: bool, n_rollouts: int, n_repeats: int, label: str,
) -> "list[float]":
    """estimate_counter_distribution を n_repeats 回呼び、各回の秒数を返す。

    負荷変動下で進捗を追えるよう、各回の結果を即時 print する
    (flush=True、長時間反復のため途中経過を見えるようにする)。
    """
    times: "list[float]" = []
    for i in range(n_repeats):
        t0 = time.perf_counter()
        estimate_counter_distribution(
            board, time_budget_sec=_TIME_BUDGET_SEC, n_rollouts=n_rollouts,
            use_native=use_native,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"    [{label} {i + 1}/{n_repeats}] {elapsed:.3f}秒", flush=True)
    return times


def main() -> None:
    """複数の盤面フルネス (パーセンタイル) で旧/新の速度を比較する。

    native は本番既定 n_rollouts (200) で計測する (十分高速なため)。
    legacy は極端な盤面で1回あたり1時間規模になり得るため、環境変数
    BENCH_N_ROLLOUTS_LEGACY で個別に縮小できる (既定200、遅すぎる場合は
    呼び出し側が明示的に減らす。n_rollouts線形の1手あたりコストなので
    縮小しても「1手あたり」の速度比較としては有効)。
    """
    print(f"NATIVE_AVAILABLE={NATIVE_AVAILABLE}")
    print(
        f"time_budget={_TIME_BUDGET_SEC}秒 "
        f"n_rollouts(native)={_N_ROLLOUTS_NATIVE} n_rollouts(legacy)={_N_ROLLOUTS_LEGACY} "
        f"各{_N_REPEATS}回反復\n",
    )
    for percentile in (50.0, 80.0, 100.0):
        board = _load_board_at_percentile(percentile)
        n_color = int(np.sum((board._grid != 0) & (board._grid != 9)))
        print(f"--- percentile={percentile} 色ぷよ数={n_color} ---", flush=True)

        native_times = _time_n_calls(board, True, _N_ROLLOUTS_NATIVE, _N_REPEATS, "native")
        native_med = float(np.median(native_times))
        print(f"  native (n_rollouts={_N_ROLLOUTS_NATIVE}): 中央値={native_med:.3f}秒 "
              f"(全{_N_REPEATS}回: {[round(t, 3) for t in native_times]})", flush=True)

        legacy_times = _time_n_calls(board, False, _N_ROLLOUTS_LEGACY, _N_REPEATS, "legacy")
        legacy_med = float(np.median(legacy_times))
        print(f"  legacy (n_rollouts={_N_ROLLOUTS_LEGACY}): 中央値={legacy_med:.3f}秒 "
              f"(全{_N_REPEATS}回: {[round(t, 3) for t in legacy_times]})", flush=True)

        # n_rollouts が異なる場合は1手あたりコストで正規化した倍速も併記する。
        native_per_rollout = native_med / _N_ROLLOUTS_NATIVE
        legacy_per_rollout = legacy_med / _N_ROLLOUTS_LEGACY
        speedup = legacy_per_rollout / native_per_rollout if native_per_rollout > 0 else float("inf")
        print(f"  1rolloutあたり speedup: {speedup:.1f}x\n", flush=True)


if __name__ == "__main__":
    main()
