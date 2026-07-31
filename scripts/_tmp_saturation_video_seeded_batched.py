"""飽和連鎖量 mode="video" 本気ビルダー v4: シード版 + bitboardバッチ probe。

コーディネータ指示(2026-07-22 手順4.5・手順5)の反映:
    - probe(b) を1候補ずつ ChainSimulator.simulate するループから、
      候補を全部まとめて batch_from_boards -> chain_bitboard.simulate_batch
      するバッチ判定に置き換える (既存 ChainSimulator は触らない)。
    - これによりビーム幅 50/100/200・各手フルprobe を現実的コストで回す。
    - 探索骨格・シード方式 (初期フロンティア=_takapt_full_scan上位k+元盤面)
      は前回セッションのまま変更しない。

止め時 (継続): ビーム幅200・フルprobeでも current_max_chain が中〜高の
盤面でビルダー自身が明確に超えられない場合、ここで停止し報告する
(簡易proxy=build_ceiling_chain へ切り替える判断)。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_saturation_video_seeded_batched
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

from src.board import BOARD_COLS, BOARD_ROWS, Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.chain_bitboard import batch_from_boards, simulate_batch  # noqa: E402
import src.indicators_v2 as iv  # noqa: E402

from scripts._tmp_saturation_rolling_horizon import (  # noqa: E402
    FULL_BOARD_CAP,
    BUFFER_EMPTY_CELLS,
    COLOR_SET_6,
    _place_pair,
    _creates_ignition,
)
from scripts._tmp_saturation_video_seeded import (  # noqa: E402
    _cheap_score_np,
    _enumerate_build_placements,
    BUILD_PAIR_COLORS_FULL,
    W_PROBE_POTENTIAL,
)

SEED_K_DEFAULT: int = 5
MAX_BUILD_STEPS: int = FULL_BOARD_CAP
# probe(b) を適用する候補数の上限倍率 (beam_width * この値まで安価スコアで事前選抜)。
PROBE_POOL_MULTIPLIER: int = 3


# ============================
# バッチ化 probe(b)
# ============================


def _probe_potential_batch(
    boards: "list[Board]", colors: "tuple[int, ...]" = COLOR_SET_6,
) -> "list[float]":
    """複数盤面の probe(b) を1回の bitboard バッチ呼出でまとめて計算する。

    各盤面につき「各列に理想色を1個仮に落とす」変種 (最大 6列×5色=30通り) を
    生成し、全盤面分をまとめて1回 `simulate_batch` する。
    既存 ChainSimulator は使わない (chain_bitboard のみ使用)。

    Returns:
        list[float]: 各入力盤面に対応する最大到達連鎖数。
    """
    variant_boards: "list[Board]" = []
    owner_index: "list[int]" = []
    for i, b in enumerate(boards):
        for col in range(BOARD_COLS):
            height = b.height_of(col)
            if height >= BOARD_ROWS:
                continue
            row = BOARD_ROWS - 1 - height
            for color in colors:
                work = b.copy()
                work.set(row, col, color)
                variant_boards.append(work)
                owner_index.append(i)

    best = [0.0] * len(boards)
    if not variant_boards:
        return best

    batch = batch_from_boards(variant_boards)
    results = simulate_batch(batch)
    for owner, res in zip(owner_index, results):
        if res.chain_count > best[owner]:
            best[owner] = float(res.chain_count)
    return best


def _seed_frontier(board: Board, sim: ChainSimulator, seed_k: int) -> "list[Board]":
    hits = iv._takapt_full_scan(board, sim)
    hits_sorted = sorted(hits, key=lambda h: h[0], reverse=True)
    seeds = [dropped for _, _, _, dropped, _ in hits_sorted[:seed_k]]
    seeds.append(board.copy())
    seen: "set[bytes]" = set()
    deduped: "list[Board]" = []
    for b in seeds:
        key = b._grid.tobytes()
        if key not in seen:
            seen.add(key)
            deduped.append(b)
    return deduped


def saturated_chain_count_video_batched(
    board: Board,
    sim: "ChainSimulator | None" = None,
    beam_width: int = 50,
    seed_k: int = SEED_K_DEFAULT,
    buffer_cells: int = BUFFER_EMPTY_CELLS,
    max_steps: int = MAX_BUILD_STEPS,
    pair_colors: "tuple[tuple[int, int], ...]" = BUILD_PAIR_COLORS_FULL,
) -> "tuple[float, int, float]":
    """mode="video" v4: シード版 + bitboardバッチ probe。各手フルprobe。

    Returns:
        (raw_chain_count, steps_taken, elapsed_sec)
    """
    sim = sim or ChainSimulator()
    if board.is_dead():
        return 0.0, 0, 0.0

    t_start = time.perf_counter()
    target_cells = FULL_BOARD_CAP - buffer_cells
    frontier = _seed_frontier(board, sim, seed_k)
    terminal_chains: "list[float]" = []
    steps = 0

    while frontier and steps < max_steps:
        next_candidates: "list[Board]" = []
        cheap_scores: "list[float]" = []
        still_building: "list[Board]" = []
        terminal_now: "list[Board]" = []

        for b in frontier:
            if b.count_puyos() >= target_cells:
                terminal_now.append(b)
                continue
            raw = _enumerate_build_placements(b, pair_colors)
            if not raw:
                terminal_now.append(b)
                continue
            safe = [(cb, cells) for cb, cells in raw if not _creates_ignition(b, cells)]
            if not safe:
                # デッドロック: 安価スコア最良候補を「発火」させて打ち切り確定。
                best_cb = max(raw, key=lambda bc: _cheap_score_np(bc[0]))[0]
                terminal_now.append(best_cb)
                continue
            still_building.append(b)
            for cb, _ in safe:
                next_candidates.append(cb)
                cheap_scores.append(_cheap_score_np(cb))

        if terminal_now:
            terminal_chains.extend(_probe_potential_batch(terminal_now, COLOR_SET_6))

        if not still_building:
            break

        # 安価スコアで probe 対象を事前に絞る (全frontier×全safe候補は
        # beam_width=200 だと数万〜数十万件になり、バッチでも構築コストが
        # 過大になるため)。probe は上位 beam_width*PROBE_POOL_MULTIPLIER のみに適用。
        pre_order = sorted(
            range(len(next_candidates)), key=lambda i: cheap_scores[i], reverse=True,
        )
        probe_pool_idx = pre_order[: beam_width * PROBE_POOL_MULTIPLIER]
        probe_targets = [next_candidates[i] for i in probe_pool_idx]

        # 各手フル probe (バッチ化により probe_pool 全体に一括適用)。
        potentials = _probe_potential_batch(probe_targets, COLOR_SET_6)
        combined = [
            cheap_scores[probe_pool_idx[j]] + W_PROBE_POTENTIAL * potentials[j]
            for j in range(len(probe_targets))
        ]
        order = sorted(range(len(probe_targets)), key=lambda j: combined[j], reverse=True)
        frontier = [probe_targets[j] for j in order[:beam_width]]
        steps += 1

    if not terminal_chains and frontier:
        terminal_chains = _probe_potential_batch(frontier, COLOR_SET_6)

    best = max(terminal_chains) if terminal_chains else 0.0
    elapsed = time.perf_counter() - t_start
    return best, steps, elapsed


# ============================
# 品質テスト本体
# ============================

BOARDS_NPZ = Path("data/indicators_v2/boards/v29.npz")


def main() -> None:
    print("=== mode=video v4 (バッチprobe版) 品質テスト ===")
    data = np.load(str(BOARDS_NPZ), allow_pickle=True)
    grids = data["grids"]
    rng = np.random.default_rng(1)
    n = min(20, len(grids))
    idxs = rng.choice(len(grids), size=n, replace=False)
    boards = [Board.from_list(grids[i].tolist()) for i in idxs]
    boards = [b for b in boards if not b.is_dead()]

    sim = ChainSimulator()
    small_sample = boards[:5]

    # before (前回セッションのシード版、simulate個別呼出、beam_width=15) の参考値
    before_raw = {0: 2, 1: 5, 2: 1, 3: 1, 4: 4}  # 前回報告の v3(シード版) 結果を再掲

    for beam_width in (50, 100, 200):
        print(f"\n### beam_width={beam_width} (フルprobe・バッチ判定) ###")
        for i, board in enumerate(small_sample):
            cur = iv.current_max_chain(board, sim).raw
            raw, steps, elapsed = saturated_chain_count_video_batched(
                board, sim, beam_width=beam_width,
            )
            exceeded = "YES" if raw > cur else ("tie" if raw == cur else "no")
            print(
                f"[盤面{i}] current_max_chain={cur:.0f} | before(v3)={before_raw[i]} | "
                f"after(v4,beam={beam_width})={raw:.0f} | steps={steps} "
                f"time={elapsed*1000:.0f}ms | current超え={exceeded}"
            )

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
