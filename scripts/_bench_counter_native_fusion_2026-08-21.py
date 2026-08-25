"""#24 MC反撃計算 選択ロジックのRust融合 (v3.3) 速度実測 (2026-08-21).

user発注: 「先にRust繋げた方が良くないか」への応答として実装した3項目
(① 盤面遅延生成、② 既知ツモ重複計算排除、③ 選択ロジック全体のRust融合) の
速度実測。cProfile は禁止 (measure with perf_counter のみ、CLAUDE.md/
memory project_counter_reach_cost_breakdown_2026-08-21 の指示通り)。

計測方法:
    1. `_select_build_placement` 単体: 融合後 (use_native=True) vs
       融合前を再現したリファレンス実装 (既存の個別バッチ呼び出し3回、
       `_enumerate_placements_dispatch`/`_current_max_chain_values_batch`/
       `_potential_fire_power_values_batch` はコード上まだ残っているため
       素直に呼べる) を同一プロセス内で交互に計測 (paired比較、
       feedback_paired_comparison_fixed_population_2026-08-20 準拠)。
    2. `estimate_counter_distribution` (既知ツモ2手あり): 既知ツモ重複計算
       排除 (enable_prefix_dedup=True/False) の比較。n_rollouts=200 (既定)
       と 60 (オーバーレイ実効) の両方で測る。
    3. 陽性対照: POTENTIAL_FIRE_POWER_BEAM_K を一時的に5→10に変更し、
       tie-break 区間だけ時間が約2倍に伸びることを確認 (計測後に必ず戻す)。

読み取り専用のベンチマーク (盤面を破壊しない)。CPU競合の実測値
(loadavg/nproc) を必ず併記する (feedback_speed_claims_need_parallelism_
2026-08-20 準拠、並列数を書かない速度主張は無効)。
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_UNKNOWN, Board  # noqa: E402
from src.console_init import init_console  # noqa: E402

init_console()

import scripts.mc_counter_estimator as mc  # noqa: E402
import src.indicators_v2 as iv  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402

COLORS = (1, 2, 3, 4)

# 実盤面サンプル (test_puyo_core_parity.py と同一データソース)。
# ⚠️ 完全ランダム合成盤面 (旧 _make_board、全列を同じ高さまで均等に乱数で
# 埋める) は「どこに置いても4連結が発生する」密度になりやすく、実測すると
# build_only (消去を起こさない配置) が常に0件になってしまい (`_select_
# build_placement` の tie-break 分岐 [実測上の主要コスト] を一度も通らない、
# 2026-08-21 診断で確認済み)。実際の中盤盤面 (空き列がある) では build_only
# は348件中282件で非0・tied件数中央値22件だったため、ベンチの代表性を
# 保つため実盤面サンプルを使う。
_DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-11"
)


def _load_real_boards(n: int = 20) -> "list[Board]":
    npz_files = sorted(_DATA_DIR.glob("*.npz"))[:3]
    boards: "list[Board]" = []
    for path in npz_files:
        data = np.load(path, allow_pickle=True)
        grids = data["grids"]
        for i in range(min(30, grids.shape[0])):
            grid = grids[i].astype(np.uint8)
            if np.any(grid == COLOR_UNKNOWN):
                continue
            b = Board()
            b._grid = grid
            boards.append(b)
            if len(boards) >= n:
                return boards
    return boards


def _old_select_build_placement(
    current: Board, pair: "tuple[int, int]", sim: ChainSimulator,
) -> "Board | None":
    """融合前 (v3.2時点) の `_select_build_placement` ロジックの再現
    (`scripts/mc_counter_estimator.py` の内部ヘルパーはまだ残っているため
    そのまま呼べる。融合後との速度比較専用、本番コードではない)。
    """
    candidates = mc._enumerate_placements_dispatch(current, pair, sim, True)
    build_only = [(c, p) for c, p in candidates if c == 0 and not p.is_dead()]
    if not build_only:
        non_dead = [(c, p) for c, p in candidates if not p.is_dead()]
        if not non_dead:
            return None
        _c, placed = min(non_dead, key=lambda cp: cp[0])
        return mc._native_simulate_chain(placed).final_board
    if len(build_only) == 1:
        return build_only[0][1]
    build_boards = [p for _c, p in build_only]
    chain_values = mc._current_max_chain_values_batch(build_boards, sim, True)
    scored = list(zip(chain_values, build_boards))
    best_potential = max(potential for potential, _p in scored)
    tied = [p for potential, p in scored if potential == best_potential]
    if len(tied) == 1:
        return tied[0]
    pfp_values = mc._potential_fire_power_values_batch(tied, sim, True)
    best_idx = max(range(len(tied)), key=lambda i: pfp_values[i])
    return tied[best_idx]


def _bench_select_build_placement(n_calls: int = 300) -> None:
    sim = ChainSimulator()
    boards = _load_real_boards(5)
    pairs = [(1, 2), (3, 4), (1, 1), (2, 3)]
    old_times: "list[float]" = []
    new_times: "list[float]" = []
    calls_per_rep = len(boards) * len(pairs)
    reps = max(1, n_calls // calls_per_rep)
    for _rep in range(reps):
        # 交互に計測 (paired比較、片方だけがシステム負荷の谷/山に当たる偏りを抑える)
        t0 = time.perf_counter()
        for b in boards:
            for p in pairs:
                _old_select_build_placement(b, p, sim)
        old_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for b in boards:
            for p in pairs:
                mc._select_build_placement(b, p, sim, use_native=True)
        new_times.append(time.perf_counter() - t0)

    old_ms = statistics.median(old_times) / calls_per_rep * 1000.0
    new_ms = statistics.median(new_times) / calls_per_rep * 1000.0
    print(f"  _select_build_placement 1回あたり: 旧(3回個別バッチ) {old_ms:.4f}ms  "
          f"新(1回融合) {new_ms:.4f}ms  倍率 {old_ms / new_ms:.2f}x  (rep={reps}件)")


def _bench_prefix_dedup(n_rollouts: int, budget_sec: float, board: Board) -> None:
    known_pairs = ((1, 2), (3, 4))
    reps = 3
    dedup_times: "list[float]" = []
    full_times: "list[float]" = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mc.estimate_counter_distribution(
            board, budget_sec, known_pairs=known_pairs, n_rollouts=n_rollouts,
        )
        dedup_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        mc.estimate_counter_distribution(
            board, budget_sec, known_pairs=known_pairs, n_rollouts=n_rollouts,
            enable_prefix_dedup=False,
        )
        full_times.append(time.perf_counter() - t0)

    dedup_ms = statistics.median(dedup_times) * 1000.0
    full_ms = statistics.median(full_times) * 1000.0
    print(f"  n_rollouts={n_rollouts} budget={budget_sec:.1f}s: "
          f"dedup {dedup_ms:.1f}ms  旧相当(毎回再計算) {full_ms:.1f}ms  "
          f"倍率 {full_ms / dedup_ms:.2f}x")


def _bench_positive_control() -> None:
    """POTENTIAL_FIRE_POWER_BEAM_K を5→10に一時変更し、tie-break区間だけ
    時間が約2倍に伸びることを確認する (計装が劣化を検出できることの証明、
    測って必ず戻す)。tie-break (pfp計算) の比重を上げるため、build_only
    全候補が確実にタイになりやすい平坦な盤面を使う。
    """
    sim = ChainSimulator()
    # 空盤面: 22配置全てが build_only かつ current_max_chain が同値タイ (22件)
    # になることを事前確認済み (診断: build_only22/tied22)。tied 全件に対して
    # potential_fire_power (beam_k依存) が呼ばれるため、BEAM_K依存性を
    # 最大限に検出できる構図。
    board = Board()
    pair = (1, 2)
    original = mc.POTENTIAL_FIRE_POWER_BEAM_K
    try:
        n = 400
        t0 = time.perf_counter()
        for _ in range(n):
            mc._select_build_placement(board, pair, sim, use_native=True)
        base_ms = (time.perf_counter() - t0) * 1000.0 / n

        mc.POTENTIAL_FIRE_POWER_BEAM_K = 10
        t0 = time.perf_counter()
        for _ in range(n):
            mc._select_build_placement(board, pair, sim, use_native=True)
        doubled_ms = (time.perf_counter() - t0) * 1000.0 / n
    finally:
        mc.POTENTIAL_FIRE_POWER_BEAM_K = original
    print(f"  陽性対照: BEAM_K=5 → {base_ms:.4f}ms/call, BEAM_K=10 → "
          f"{doubled_ms:.4f}ms/call (倍率 {doubled_ms / base_ms:.2f}x、"
          f"2倍前後なら計装は劣化を検出できている)")


def main() -> int:
    load1, load5, load15 = os.getloadavg()
    print(f"CPU競合の実測 (feedback_speed_claims_need_parallelism_2026-08-20 準拠):")
    print(f"  nproc={os.cpu_count()}  loadavg(1,5,15分)={load1:.2f},{load5:.2f},{load15:.2f}")
    print(f"  (このベンチ自体は単一プロセス・単一スレッドで実行)")
    print()
    print("[1] _select_build_placement 単体 (融合前3回バッチ vs 融合後1回)")
    _bench_select_build_placement()
    print()
    print("[2] estimate_counter_distribution (既知ツモ2手、重複計算排除)")
    dedup_board = _load_real_boards(1)[0]
    for n in (200, 60):
        for chain in (2.0, 9.0):
            budget = float(iv.estimate_chain_anim_duration_sec(chain))
            _bench_prefix_dedup(n, budget, dedup_board)
    print()
    print("[3] 陽性対照 (POTENTIAL_FIRE_POWER_BEAM_K 5→10)")
    _bench_positive_control()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
