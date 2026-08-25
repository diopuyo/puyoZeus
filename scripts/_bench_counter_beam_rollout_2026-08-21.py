"""応手ビームロールアウト方式 (v4、user決定) の実測 (2026-08-21)。

背景: `project_counter_beam_rollout_design_2026-08-21` (user決定)。
「ツモのみランダム・置き方はビームサーチ」への方式変更の検証で、以下4点を
この順で測る (順番が重要、①で近似の天井を先に切り離す):

    ① 厳密得点(exact_score) vs 近似得点(score_approx) の速度差・返せる量の差
    ② 幅 (10/30/100/250) の飽和点
    ③ ツモの木構造での共有 (重複率の実測)
    ④ 8スレッド並列が遅くなる現象の再測定 (CPU飽和 vs 設計欠陥の判別)

cProfile 禁止 (perf_counter のみ)。並列数を必ず併記する
(`feedback_speed_claims_need_parallelism_2026-08-20`)。実盤面で測る
(合成盤面は密度が均一すぎて選択則の通り方が実戦と異なる、
`project_counter_reach_cost_breakdown_2026-08-21` 系の前回発見)。

④ の並列数比較は `native/puyo_core/src/lib.rs::get_or_build_pool` が
プロセス全体で rayon スレッドプールを1個だけ再利用する制約があるため
(初回呼び出しのスレッド数で確定)、本スクリプトはスレッド数ごとに
**別プロセス** (`sys.executable` 再実行) を起動して計測する。
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import COLOR_UNKNOWN, Board  # noqa: E402
from src.console_init import init_console  # noqa: E402

init_console()

import scripts.mc_counter_estimator as mc  # noqa: E402
import src.indicators_v2 as iv  # noqa: E402
from src.puyo_core_bridge import beam_search  # noqa: E402

_DATA_DIR = (
    _ROOT / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-11"
)


def _load_real_boards(n: int) -> "list[Board]":
    """実盤面サンプル (test_puyo_core_parity.py と同一データソース)。

    窒息 (is_dead) 盤面はビームロールアウトが即座に0を返すだけで探索を
    一切行わないため、ベンチの母集団からは除外する (窒息判定は本番の
    `estimate_counter_distribution` 入口でも同じ扱い、除外してもベンチの
    代表性は損なわない)。
    """
    npz_files = sorted(_DATA_DIR.glob("*.npz"))
    rng = np.random.RandomState(20260821)
    chosen = rng.choice(npz_files, size=min(6, len(npz_files)), replace=False)
    boards: "list[Board]" = []
    for path in chosen:
        data = np.load(path, allow_pickle=True)
        grids = data["grids"]
        idxs = rng.choice(grids.shape[0], size=min(40, grids.shape[0]), replace=False)
        for i in idxs:
            grid = grids[i].astype(np.uint8)
            if np.any(grid == COLOR_UNKNOWN):
                continue
            b = Board()
            b._grid = grid
            if b.is_dead():
                continue
            boards.append(b)
            if len(boards) >= n:
                return boards
    return boards


def _report_load(label: str) -> None:
    load1, load5, load15 = os.getloadavg()
    print(f"  [{label}] nproc={os.cpu_count()}  loadavg(1,5,15分)="
          f"{load1:.2f},{load5:.2f},{load15:.2f}")


# ============================
# ① 厳密得点 vs 近似得点
# ============================


def _bench_exact_vs_approx(boards: "list[Board]") -> None:
    print("[1] exact_score vs score_approx — 速度差・返せる量の差")
    depth = 13
    width = 30
    reps = 3
    approx_times: "list[float]" = []
    exact_times: "list[float]" = []
    ratios: "list[float]" = []
    rng = np.random.RandomState(1)
    for board in boards:
        pairs = [(int(rng.choice((1, 2, 3, 4))), int(rng.choice((1, 2, 3, 4)))) for _ in range(depth)]

        t0 = time.perf_counter()
        for _ in range(reps):
            r_approx = beam_search(board, pairs, width, exclude_hidden_row_from_pop=True)
        approx_times.append((time.perf_counter() - t0) / reps)

        t0 = time.perf_counter()
        for _ in range(reps):
            r_exact = beam_search(
                board, pairs, width, exclude_hidden_row_from_pop=True, use_exact_score=True,
            )
        exact_times.append((time.perf_counter() - t0) / reps)

        if r_approx.best_score > 0:
            ratios.append(r_exact.best_score / r_approx.best_score)

    approx_ms = statistics.median(approx_times) * 1000.0
    exact_ms = statistics.median(exact_times) * 1000.0
    print(f"  速度: approx {approx_ms:.2f}ms/回  exact {exact_ms:.2f}ms/回  "
          f"倍率(exact/approx) {exact_ms / approx_ms:.3f}x  "
          f"(理論=同じ計算を読むだけなので≒1.0x が期待値)")
    if ratios:
        arr = np.array(ratios)
        print(f"  返せる量の比 (exact/approx、n={len(arr)}): "
              f"中央値={np.median(arr):.3f}  p25={np.percentile(arr, 25):.3f}  "
              f"p75={np.percentile(arr, 75):.3f}  最大={arr.max():.3f}")


# ============================
# ② 幅の飽和点
# ============================


def _bench_positive_control_width_one(boards: "list[Board]") -> None:
    """陽性対照: 幅1 (=1手ずつ最大素点を選ぶのと実質同義、v1の即時発火優先
    greedyを再現するはず) にすると過小評価が起きること、幅を広げると
    改善方向に動くことを実盤面で確認する
    (`scripts/mc_counter_estimator.py` docstring 冒頭の v1 過小評価
    [match_04: 35 vs 実測319] の検収アンカー、実盤面で確認する方針
    [`feedback_accuracy_claims_distribution_2026-08-07`])。
    """
    print("[2-0] 陽性対照: 幅1 vs 幅250 (実盤面、budget=9連鎖想定)")
    budget = float(iv.estimate_chain_anim_duration_sec(9.0))
    width1_values: "list[float]" = []
    width250_values: "list[float]" = []
    for board in boards:
        d1 = mc.estimate_counter_distribution(
            board, budget, known_pairs=((1, 2), (3, 4)), n_rollouts=60,
            rollout_mode="beam", beam_width=1,
        )
        d250 = mc.estimate_counter_distribution(
            board, budget, known_pairs=((1, 2), (3, 4)), n_rollouts=60,
            rollout_mode="beam", beam_width=250,
        )
        width1_values.append(d1.mean)
        width250_values.append(d250.mean)
    arr1 = np.array(width1_values)
    arr250 = np.array(width250_values)
    print(f"  幅1  : 中央値={np.median(arr1):.1f}  平均={arr1.mean():.1f}")
    print(f"  幅250: 中央値={np.median(arr250):.1f}  平均={arr250.mean():.1f}")
    improved = int(np.sum(arr250 > arr1))
    print(f"  幅250が幅1を上回った盤面: {improved}/{len(boards)} "
          f"(過小評価が幅拡大で改善する方向に動いているかの直接確認)")


def _bench_width_saturation(boards: "list[Board]") -> None:
    print("[2] 幅の飽和点 (10/30/100/250)")
    widths = (10, 30, 100, 250)
    for n_rollouts in (200, 60):
        print(f"  n_rollouts={n_rollouts}:")
        for chain in (2.0, 5.0, 9.0):
            budget = float(iv.estimate_chain_anim_duration_sec(chain))
            for width in widths:
                values: "list[float]" = []
                t0 = time.perf_counter()
                for board in boards:
                    dist = mc.estimate_counter_distribution(
                        board, budget, known_pairs=((1, 2), (3, 4)),
                        n_rollouts=n_rollouts, rollout_mode="beam", beam_width=width,
                    )
                    values.append(dist.mean)
                elapsed = time.perf_counter() - t0
                arr = np.array(values)
                print(f"    連鎖{chain:.0f}想定budget={budget:.1f}s 幅{width:4d}: "
                      f"mean中央値={np.median(arr):7.1f} p25={np.percentile(arr,25):7.1f} "
                      f"p75={np.percentile(arr,75):7.1f} 最大={arr.max():7.1f}  "
                      f"({len(boards)}盤面で{elapsed:.1f}s)")


# ============================
# ③ ツモの木構造での共有
# ============================


def _canonical_pair(pair: "tuple[int, int]") -> "tuple[int, int]":
    """(top,bot) と (bot,top) は盤面到達集合が同一なので同一視する
    (`native/puyo_core/src/bitboard.rs::place_pair` の回転対称性、
    モジュール docstring 参照)。"""
    return (min(pair), max(pair))


def _measure_tsumo_dedup(board: Board, n_rollouts: int) -> None:
    """`n_rollouts` 本のランダムツモ列を引き、深さ別の重複率を測る。

    既知ツモ (known_pairs) は乱数を消費しない決定論的な手なので、
    全ロールアウトで自明に100%重複する (既存 v3.3 `_compute_known_prefix_
    state` がこの区間の重複は既に解消済み)。本測定は「③ 見えていない
    (=未確定の乱数) ツモの木構造での共有」がテーマなので、
    known_pairs=() (全手ランダム) にして純粋にランダム部分の重複率だけを
    測る (既知部分と混ぜると自明な重複で相殺され、ランダム部分特有の
    重複率が見えなくなる)。
    """
    depth = 15
    colors = (1, 2, 3, 4)
    rng = mc.random.Random(mc._mc_counter_seed(board, 5.0))
    sequences = [
        mc._draw_beam_tsumo_sequence(depth, (), colors, rng)
        for _ in range(n_rollouts)
    ]
    print(f"  n_rollouts={n_rollouts}:")
    distinct_per_depth: "list[int]" = []
    for d in range(1, depth + 1):
        prefixes = [
            tuple(_canonical_pair(p) for p in seq[:d]) for seq in sequences
        ]
        distinct = len(set(prefixes))
        distinct_per_depth.append(distinct)
        if d in (1, 2, 3, 5, 10, 15):
            dedup_ratio = 1.0 - distinct / n_rollouts
            print(f"    深さ{d:2d}: 相異なる接頭辞={distinct:3d}/{n_rollouts}  "
                  f"重複排除できる余地={dedup_ratio * 100:5.1f}%")

    # 木の形で共有した場合の総ノード訪問数 (= 各深さの相異なる接頭辞数の和)
    # vs フラットに n_rollouts×depth 回訪問する現状との比較 (削減率の実測)。
    tree_units = sum(distinct_per_depth)
    flat_units = n_rollouts * depth
    reduction = 1.0 - tree_units / flat_units
    print(f"    [木構造で共有した場合] 総ノード訪問数: 現状={flat_units}  "
          f"共有後={tree_units}  削減率={reduction * 100:.1f}%")

    # 「木の形で引く」代替案: 1手目を10通り (同色4+異色6) に完全層化し、
    # 60/200本を均等割り当てした場合の1手目重複率 (理論値、運の要素を除去)。
    n_combos = 10
    per_combo = n_rollouts // n_combos
    print(f"    [参考] 1手目を{n_combos}通りに層化した場合: "
          f"1手目の相異なる接頭辞は必ず{min(n_combos, n_rollouts)}(既定通り)、"
          f"1手目あたり{per_combo}本均等 (現状のランダム引きでの1手目分布と比較する目的)")


def _bench_tsumo_dedup(boards: "list[Board]") -> None:
    print("[3] ツモの木構造での共有 (重複率の実測)")
    board = boards[0]
    for n_rollouts in (200, 60):
        _measure_tsumo_dedup(board, n_rollouts)


# ============================
# ④ 8スレッド並列の再測定 (別プロセス方式、スレッドプール1回制約回避)
# ============================


def _bench_parallel_subprocess(board_idx: int, depth: int, width: int, num_threads: "int | None") -> float:
    """このスクリプト自身を `--parallel-worker` モードで再実行し、
    指定スレッド数での実行時間 [ms] を1件返す (別プロセス方式)。
    """
    args = [sys.executable, __file__, "--parallel-worker",
             str(board_idx), str(depth), str(width), str(num_threads)]
    out = subprocess.run(args, capture_output=True, text=True, check=True)
    return float(out.stdout.strip().splitlines()[-1])


def _parallel_worker_main(argv: "list[str]") -> int:
    board_idx, depth, width, num_threads_raw = argv
    boards = _load_real_boards(int(board_idx) + 1)
    board = boards[int(board_idx)]
    rng = np.random.RandomState(2)
    pairs = [(int(rng.choice((1, 2, 3, 4))), int(rng.choice((1, 2, 3, 4)))) for _ in range(int(depth))]
    num_threads = None if num_threads_raw == "None" else int(num_threads_raw)
    t0 = time.perf_counter()
    beam_search(
        board, pairs, int(width), exclude_hidden_row_from_pop=True,
        num_threads=num_threads, use_exact_score=True,
    )
    print((time.perf_counter() - t0) * 1000.0)
    return 0


def _bench_parallel_scaling(width: int) -> None:
    print(f"[4] 8スレッド並列の再測定 (幅={width}、別プロセス方式)")
    for depth in (13, 16):
        print(f"  深さ{depth}:")
        for num_threads in (None, 1, 2, 4, 8):
            times = [
                _bench_parallel_subprocess(0, depth, width, num_threads) for _ in range(3)
            ]
            label = "単スレッド(None)" if num_threads is None else f"{num_threads}スレッド"
            print(f"    {label:>14s}: {statistics.median(times):7.2f}ms")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--parallel-worker":
        return _parallel_worker_main(sys.argv[2:])

    print("=== 応手ビームロールアウト方式 実測 (2026-08-21) ===")
    _report_load("計測開始前")
    print(f"  (①②③は単一プロセス・単一スレッド、④のみ別プロセス生成で計測)")
    print()

    boards = _load_real_boards(20)
    print(f"実盤面 {len(boards)} 件をロード ({_DATA_DIR.name})")
    print()

    _bench_exact_vs_approx(boards)
    print()
    _bench_positive_control_width_one(boards[:10])
    print()
    _bench_width_saturation(boards)
    print()
    _bench_tsumo_dedup(boards)
    print()
    _bench_parallel_scaling(width=30)

    print()
    _report_load("計測終了後")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
