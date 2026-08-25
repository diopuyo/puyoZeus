"""「両方採用+高速化3点」の速度実測 (2026-08-21、user指示)。

① 完全探索の結果をビームサーチの初期集団にする (seed)
② 答えを変えない打ち切り (early_exit_at_threshold)
③ 死ぬ枝を切る (既存 filter_dead=True、audit結果は本体で報告)

cProfile禁止 (perf_counter)。実盤面 (data/indicators_v2/
boards_lean_model50v2_2026-08-20、学習データ62本) を使う。②は場面を
層別 (閾値が小さい/大きい) して報告する
(feedback_stratify_before_pooling_2026-07-29 準拠、プールした平均は
相殺で見かけ倒しになるため)。
"""
from __future__ import annotations

import glob
import os
import statistics
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
from src.puyo_core_bridge import (  # noqa: E402
    beam_search,
    beam_search_continue,
    exact_shallow_search,
)

_DATA_DIR = _ROOT / "data" / "indicators_v2" / "boards_lean_model50v2_2026-08-20"


def _load_real_boards(n: int, seed: int = 1) -> "list[Board]":
    files = sorted(glob.glob(str(_DATA_DIR / "*.npz")))
    rng = np.random.RandomState(seed)
    chosen = rng.choice(files, size=min(10, len(files)), replace=False)
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
            if b.is_dead() or int((grid != 0).sum()) == 0:
                continue
            boards.append(b)
            if len(boards) >= n:
                return boards
    return boards


def _report_load(label: str) -> None:
    load1, load5, load15 = os.getloadavg()
    print(f"  [{label}] nproc={os.cpu_count()}  loadavg={load1:.2f},{load5:.2f},{load15:.2f}")


def _bench_exact_shallow_speed(boards: "list[Board]") -> None:
    print("[A] exact_shallow (ama方式) の実際の所要 (数msと見積もり、未実測だったもの)")
    rng = np.random.RandomState(1)
    for depth in (1, 2, 3):
        pairs = [(int(rng.choice((1, 2, 3, 4))), int(rng.choice((1, 2, 3, 4)))) for _ in range(depth)]
        for use_pruning in (False, True):
            mh = mc.EXACT_SHALLOW_PRUNE_HEIGHT if use_pruning else None
            times = []
            for b in boards:
                t0 = time.perf_counter()
                exact_shallow_search(b, pairs, exclude_hidden_row_from_pop=False, max_height=mh)
                times.append(time.perf_counter() - t0)
            ms = statistics.median(times) * 1000.0
            label = "枝刈りON" if use_pruning else "枝刈りOFF"
            print(f"  depth={depth} {label}: {ms:.4f}ms/回 (median, n={len(boards)}盤面)")


def _bench_seed_effect(boards: "list[Board]") -> None:
    """①: seedありvs無しで「同じ幅なら速度は同等、質は上がる」を確認。"""
    print("[B] ① seed効果 (幅ごとの速度・質)")
    rng = np.random.RandomState(2)
    depth = 6
    pairs_list = [
        [(int(rng.choice((1, 2, 3, 4))), int(rng.choice((1, 2, 3, 4)))) for _ in range(depth)]
        for _ in boards
    ]
    for width in (10, 30, 100):
        unseeded_scores, seeded_scores = [], []
        t_unseeded = t_seeded = 0.0
        for b, pairs in zip(boards, pairs_list):
            t0 = time.perf_counter()
            r_u = beam_search(b, pairs, width, exclude_hidden_row_from_pop=False, use_exact_score=True)
            t_unseeded += time.perf_counter() - t0
            unseeded_scores.append(r_u.best_score)

            t0 = time.perf_counter()
            seed = exact_shallow_search(
                b, pairs[:mc.EXACT_SHALLOW_SEED_DEPTH], exclude_hidden_row_from_pop=False,
            )
            truncated = mc._truncate_frontier_by_running_best(seed.final_frontier, width)
            r_s = beam_search_continue(
                truncated, seed.best_score, pairs[mc.EXACT_SHALLOW_SEED_DEPTH:], width,
                exclude_hidden_row_from_pop=False, use_exact_score=True,
            )
            t_seeded += time.perf_counter() - t0
            seeded_scores.append(r_s.best_score)
        u = np.array(unseeded_scores)
        s = np.array(seeded_scores)
        print(f"  幅{width:3d}: 旧(seedなし) 中央値={np.median(u):.0f} 平均={u.mean():.1f} "
              f"{t_unseeded/len(boards)*1000:.3f}ms/回  |  "
              f"新(seedあり) 中央値={np.median(s):.0f} 平均={s.mean():.1f} "
              f"{t_seeded/len(boards)*1000:.3f}ms/回  |  "
              f"seed側が上回った盤面 {int(np.sum(s > u))}/{len(boards)}")


def _bench_early_exit_stratified(boards: "list[Board]") -> None:
    """②: 閾値が小さい(すぐ超える)/大きい(最後まで回る)場面で層別に短縮率を測る。"""
    print("[C] ② 早期打ち切りの効果 (層別、プールしない)")
    known_pairs = ((1, 2), (3, 4))
    depth_chain = 9.0
    import src.indicators_v2 as iv
    budget = float(iv.estimate_chain_anim_duration_sec(depth_chain))
    for label, threshold in (("小さい閾値 (すぐ超える)", 1.0), ("大きい閾値 (最後まで回る)", 500.0)):
        t_on = t_off = 0.0
        for b in boards:
            t0 = time.perf_counter()
            mc.estimate_counter_distribution(
                b, budget, known_pairs=known_pairs, n_rollouts=30, thresholds_ojama=(threshold,),
                rollout_mode="beam", beam_width=100, early_exit_at_threshold=True,
            )
            t_on += time.perf_counter() - t0
            t0 = time.perf_counter()
            mc.estimate_counter_distribution(
                b, budget, known_pairs=known_pairs, n_rollouts=30, thresholds_ojama=(threshold,),
                rollout_mode="beam", beam_width=100, early_exit_at_threshold=False,
            )
            t_off += time.perf_counter() - t0
        print(f"  {label} (閾値{threshold}): ON={t_on/len(boards)*1000:.1f}ms/盤面  "
              f"OFF={t_off/len(boards)*1000:.1f}ms/盤面  倍率={t_off/t_on:.2f}x")


def _bench_combined_effect(boards: "list[Board]") -> None:
    """①②③ (v5既定スタック) まとめてONにした場合の短縮率
    (v4当時=seedなし・打ち切りなし・native fusion済み、との比較)。
    """
    print("[D] ①②③ まとめての短縮率 (v4相当 vs v5既定)")
    known_pairs = ((1, 2), (3, 4))
    import src.indicators_v2 as iv
    for chain in (5.0, 9.0):
        budget = float(iv.estimate_chain_anim_duration_sec(chain))
        t_v4 = t_v5 = 0.0
        for b in boards:
            t0 = time.perf_counter()
            mc.estimate_counter_distribution(
                b, budget, known_pairs=known_pairs, n_rollouts=30, rollout_mode="beam",
                beam_width=100,
            )
            # v4相当 (seedなし) を模すため、EXACT_SHALLOW_SEED_DEPTH=0相当の
            # 生ビームサーチを直接計測 (①を無効化した基準線)。
            pairs = mc._draw_beam_tsumo_sequence(
                mc._time_budget_to_beam_depth(budget), known_pairs, (1, 2, 3, 4),
                mc.random.Random(0),
            )
            beam_search(b, pairs, 100, exclude_hidden_row_from_pop=False, use_exact_score=True)
            t_v4 += time.perf_counter() - t0

            t0 = time.perf_counter()
            mc.estimate_counter_distribution(
                b, budget, known_pairs=known_pairs, n_rollouts=30, rollout_mode="beam",
                beam_width=100,
            )
            t_v5 += time.perf_counter() - t0
        print(f"  連鎖{chain:.0f}想定: v4相当基準 {t_v4/len(boards)*1000:.1f}ms/盤面(30rollout)  "
              f"v5既定(①のseedのみ内蔵、②③既定OFF) {t_v5/len(boards)*1000:.1f}ms/盤面  "
              f"倍率={t_v4/t_v5:.2f}x")


def main() -> int:
    _report_load("計測開始前")
    print(f"  (単一プロセス・単一スレッドで計測)")
    boards = _load_real_boards(15)
    print(f"実盤面 {len(boards)} 件をロード")
    print()
    _bench_exact_shallow_speed(boards)
    print()
    _bench_seed_effect(boards)
    print()
    _bench_early_exit_stratified(boards[:8])
    print()
    _bench_combined_effect(boards[:8])
    print()
    _report_load("計測終了後")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
