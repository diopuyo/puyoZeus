"""飽和連鎖量 mode="video" 本気ビルダー v3: 既存最良連鎖を「シード」にして伸長する版。

コーディネータ方針転換 (2026-07-22 再指示) の反映:
    - ゼロから汎用ヒューリスティックで組むのをやめる (v2 は既存の良い骨格を
      壊して彷徨うことが判明済み: current_max_chain=11 → builder=5 の失敗)。
    - 初期フロンティアに `_takapt_best_drop` 相当の上位 seed_k 個
      (= current_max_chain を達成する 1 手先ボード群) を必ず含める。
      「何もしない元盤面」も保険として残す。
    - ビーム幅は 10-20 の小さめでよい (シード済みのため広い探索は不要)。
    - トリガー列(井戸)ボーナスは維持 (骨格の発火列を埋める手を間接的に抑制)。
    - numpy ベクトル化: 隣接ペア二乗和・段差穴ペナルティを grid 演算化。
    - simulate 呼び出し回数を計測し、ボトルネックの所在を特定する。

止め時 (コーディネータ指示): シード方式+小ビーム+numpy化でも
current_max_chain が中〜高の盤面でビルダー自身が明確に超えられない場合、
ここで停止し「簡易proxyか保留」を報告する。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_saturation_video_seeded
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN, Board,
)
from src.chain import ChainSimulator  # noqa: E402
import src.indicators_v2 as iv  # noqa: E402

from scripts._tmp_saturation_rolling_horizon import (  # noqa: E402
    FULL_BOARD_CAP,
    BUFFER_EMPTY_CELLS,
    COLOR_SET_6,
    _place_pair,
    _creates_ignition,
)

# ============================
# 定数
# ============================

W_ADJ_PAIR_SQ: float = 1.0
W_STRUCTURE_PENALTY: float = 1.0
W_TRIGGER_WELL: float = 2.0
W_PROBE_POTENTIAL: float = 3.0

BUILD_PAIR_COLORS_FULL: "tuple[tuple[int, int], ...]" = tuple(
    (t, b) for t in COLOR_SET_6 for b in COLOR_SET_6
)

SEED_K_DEFAULT: int = 5
BEAM_WIDTH_DEFAULT: int = 15
PROBE_POOL_MULTIPLIER: int = 3
MAX_BUILD_STEPS: int = FULL_BOARD_CAP

_VALID_COLOR_SET = frozenset(COLOR_SET_6)


# ============================
# numpy ベクトル化 評価関数
# ============================


def _adjacent_pair_sq_sum_np(grid: np.ndarray) -> float:
    """(a) 同色隣接ペア数の二乗和 (numpy ベクトル化版)。

    Python 二重ループを撤廃し、色ごとに boolean mask のシフト比較で
    横/縦の同色隣接エッジ数を一括計算する。
    """
    total = 0.0
    for color in COLOR_SET_6:
        mask = grid == color
        h_edges = np.count_nonzero(mask[:, :-1] & mask[:, 1:])
        v_edges = np.count_nonzero(mask[:-1, :] & mask[1:, :])
        edges = h_edges + v_edges
        total += float(edges * edges)
    return total


def _structure_penalty_np(grid: np.ndarray) -> float:
    """(c) 段差+穴ペナルティ (numpy ベクトル化版)。"""
    non_empty = grid != COLOR_EMPTY
    # 各列の高さ (上から数えて最初の非空セル位置から算出)。
    has_puyo = non_empty.any(axis=0)
    first_idx = non_empty.argmax(axis=0)  # 非空が無い列は0になる(has_puyoで判定)
    heights = np.where(has_puyo, BOARD_ROWS - first_idx, 0)
    bumpiness = float(np.abs(np.diff(heights)).sum())
    # 穴: 各列で最初の非空セルより下にある空セル数。
    holes = 0
    for c in range(BOARD_COLS):
        if not has_puyo[c]:
            continue
        holes += int((~non_empty[first_idx[c]:, c]).sum())
    return bumpiness + float(holes)


def _trigger_well_bonus_np(grid: np.ndarray) -> float:
    non_empty = grid != COLOR_EMPTY
    has_puyo = non_empty.any(axis=0)
    first_idx = non_empty.argmax(axis=0)
    heights = np.where(has_puyo, BOARD_ROWS - first_idx, 0)
    heights_sorted = np.sort(heights)
    return float(heights_sorted[1] - heights_sorted[0])


def _cheap_score_np(board: Board) -> float:
    grid = board._grid
    return (
        W_ADJ_PAIR_SQ * _adjacent_pair_sq_sum_np(grid)
        - W_STRUCTURE_PENALTY * _structure_penalty_np(grid)
        + W_TRIGGER_WELL * _trigger_well_bonus_np(grid)
    )


# ============================
# probe (b): simulate呼び出し回数を計測するラッパー
# ============================

_SIM_CALL_COUNTER = {"count": 0}


def _probe_potential_counted(board: Board, sim: ChainSimulator, colors: tuple[int, ...]) -> float:
    best = 0
    for col in range(BOARD_COLS):
        height = board.height_of(col)
        if height >= BOARD_ROWS:
            continue
        row = BOARD_ROWS - 1 - height
        for color in colors:
            work = board.copy()
            work.set(row, col, color)
            _SIM_CALL_COUNTER["count"] += 1
            chain = sim.simulate(work).chain_count
            if chain > best:
                best = chain
    return float(best)


def _enumerate_build_placements(
    board: Board, pair_colors: "tuple[tuple[int, int], ...]",
) -> "list[tuple[Board, list[tuple[int, int]]]]":
    candidates: "list[tuple[Board, list[tuple[int, int]]]]" = []
    for top, bot in pair_colors:
        for rotation in range(4):
            max_col = BOARD_COLS if rotation in (0, 2) else BOARD_COLS - 1
            for col in range(max_col):
                placed = _place_pair(board, top, bot, col, rotation)
                if placed is not None:
                    candidates.append(placed)
    return candidates


def _seed_frontier(board: Board, sim: ChainSimulator, seed_k: int) -> "list[Board]":
    """初期フロンティア = takapt上位 seed_k 個の1手先ボード + 元盤面(保険)。

    _takapt_full_scan (indicators_v2.py 既存) を流用し、chain_count 降順で
    上位 seed_k 件を取り出す。重複盤面は grid bytes で除去する。
    """
    hits = iv._takapt_full_scan(board, sim)  # [(chain, col, color, dropped, result)]
    hits_sorted = sorted(hits, key=lambda h: h[0], reverse=True)
    seeds = [dropped for _, _, _, dropped, _ in hits_sorted[:seed_k]]
    seeds.append(board.copy())  # 保険: 何もしない元盤面
    seen: "set[bytes]" = set()
    deduped: "list[Board]" = []
    for b in seeds:
        key = b._grid.tobytes()
        if key not in seen:
            seen.add(key)
            deduped.append(b)
    return deduped


def saturated_chain_count_video_seeded(
    board: Board,
    sim: "ChainSimulator | None" = None,
    beam_width: int = BEAM_WIDTH_DEFAULT,
    seed_k: int = SEED_K_DEFAULT,
    probe_pool_multiplier: int = PROBE_POOL_MULTIPLIER,
    buffer_cells: int = BUFFER_EMPTY_CELLS,
    max_steps: int = MAX_BUILD_STEPS,
    pair_colors: "tuple[tuple[int, int], ...]" = BUILD_PAIR_COLORS_FULL,
) -> tuple[float, int]:
    """mode="video" 本気ビルダー v3 (シード版)。既存最良連鎖骨格を土台に伸長する。"""
    sim = sim or ChainSimulator()
    if board.is_dead():
        return 0.0, 0

    target_cells = FULL_BOARD_CAP - buffer_cells
    frontier = _seed_frontier(board, sim, seed_k)
    terminal_chains: "list[float]" = []
    steps = 0

    while frontier and steps < max_steps:
        next_candidates: "list[tuple[float, Board]]" = []
        still_building: "list[Board]" = []

        for b in frontier:
            if b.count_puyos() >= target_cells:
                terminal_chains.append(_probe_potential_counted(b, sim, COLOR_SET_6))
                continue
            raw = _enumerate_build_placements(b, pair_colors)
            if not raw:
                terminal_chains.append(_probe_potential_counted(b, sim, COLOR_SET_6))
                continue
            safe = [(cb, cells) for cb, cells in raw if not _creates_ignition(b, cells)]
            if not safe:
                best_cb = max(raw, key=lambda bc: _cheap_score_np(bc[0]))[0]
                _SIM_CALL_COUNTER["count"] += 1
                result = sim.simulate(best_cb)
                terminal_chains.append(float(result.chain_count))
                continue
            still_building.append(b)
            for cb, _ in safe:
                next_candidates.append((_cheap_score_np(cb), cb))

        if not still_building:
            break

        next_candidates.sort(key=lambda x: x[0], reverse=True)
        probe_pool = next_candidates[: beam_width * probe_pool_multiplier]

        scored: "list[tuple[float, Board]]" = []
        for cheap_s, cb in probe_pool:
            potential = _probe_potential_counted(cb, sim, COLOR_SET_6)
            total = cheap_s + W_PROBE_POTENTIAL * potential
            scored.append((total, cb))
        scored.sort(key=lambda x: x[0], reverse=True)
        frontier = [cb for _, cb in scored[:beam_width]]
        steps += 1

    if not terminal_chains and frontier:
        terminal_chains = [_probe_potential_counted(b, sim, COLOR_SET_6) for b in frontier]

    best = max(terminal_chains) if terminal_chains else 0.0
    return best, steps


# ============================
# ベンチ本体
# ============================

BOARDS_NPZ = Path("data/indicators_v2/boards/v29.npz")


def main() -> None:
    print("=== mode=video v3 (シード版) 品質確認 ===")
    data = np.load(str(BOARDS_NPZ), allow_pickle=True)
    grids = data["grids"]
    rng = np.random.default_rng(1)
    n = min(20, len(grids))
    idxs = rng.choice(len(grids), size=n, replace=False)
    boards = [Board.from_list(grids[i].tolist()) for i in idxs]
    boards = [b for b in boards if not b.is_dead()]

    sim = ChainSimulator()

    print("\n### before/after: v2(シード無し・全色ペア) vs v3(シード版) ###")
    from scripts._tmp_saturation_video_mode import (
        saturated_chain_count_video as v2_builder,
        BUILD_PAIR_COLORS_FULL as V2_FULL,
    )

    for i, board in enumerate(boards[:5]):
        cur = iv.current_max_chain(board, sim).raw

        t0 = time.perf_counter()
        raw_v2, steps_v2 = v2_builder(board, sim, beam_width=10, pair_colors=V2_FULL)
        t_v2 = time.perf_counter() - t0

        _SIM_CALL_COUNTER["count"] = 0
        t0 = time.perf_counter()
        raw_v3, steps_v3 = saturated_chain_count_video_seeded(
            board, sim, beam_width=15, seed_k=5,
        )
        t_v3 = time.perf_counter() - t0
        sim_calls_v3 = _SIM_CALL_COUNTER["count"]

        print(
            f"[盤面{i}] current_max_chain={cur:.0f} | "
            f"v2(シード無し)={raw_v2:.0f}(steps={steps_v2},{t_v2*1000:.0f}ms) | "
            f"v3(シード版)={raw_v3:.0f}(steps={steps_v3},{t_v3*1000:.0f}ms,sim呼出={sim_calls_v3}) | "
            f"v3が超えた={'YES' if raw_v3 > cur else 'no'}"
        )

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
