"""飽和連鎖量 「本気ビルダー」(mode="video") プロトタイプ。

コーディネータ方針確定 (2026-07-22追加指示) の反映:
    - 複数フロンティア保持型ビームサーチ (rolling horizon の単一パス commit から昇格)。
    - ビーム幅 20/50/100 を試す (先行研究 traP=50, takapt=400 を参考に、
      まず 20→50 で品質と速度を確認してからスケールする)。
    - 評価3本柱を維持 + probe(b) を全手で実行 (終盤限定を撤廃)。
    - トリガー列 (井戸) ボーナスを追加: 最も低い列が突出して低いほど加点
      (先行研究 U字/列4重み付けの機能版)。
    - 絶対条件: 最終値 = max(current_max_chain, ビルダー結果) で下限を保証。
    - 物理制約: 2個1組配置 (22パターン) を維持。早期発火は打ち切り確定。

エンジニアリング簡略化 (計算量対策、正直に明記):
    - 構築時の色ペア候補は「同色ペアのみ」(5通り×22配置=110候補/フロンティア盤面)
      に制限する。異色ペア込み(最大25通り)は beam_width×フロンティア展開の
      組合せ爆発が実用外になるため。probe(b) 自体は5色フルスキャン(30通り)を維持。
    - 安全候補が0(デッドロック)のフロンティア枝は個別に打ち切り、他の枝は継続。

本スクリプトは検証専用 (_tmp_ prefix)。品質確認後に indicators_v2.py へ本実装。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_saturation_video_mode
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
import src.indicators_v2 as iv  # noqa: E402

from scripts._tmp_saturation_rolling_horizon import (  # noqa: E402
    FULL_BOARD_CAP,
    BUFFER_EMPTY_CELLS,
    COLOR_SET_6,
    W_ADJ_PAIR_SQ,
    W_STRUCTURE_PENALTY,
    _adjacent_same_color_pair_sq_sum,
    _structure_penalty,
    _probe_potential,
    _place_pair,
    _creates_ignition,
)

# ============================
# 定数
# ============================

# トリガー列ボーナスの重み (最も低い列と2番目に低い列の差)。
W_TRIGGER_WELL: float = 2.0
# probe(b) の重み (rolling horizon 版と統一)。
W_PROBE_POTENTIAL: float = 3.0

# 構築候補の色ペア: 同色ペアのみに制限 (計算量対策、docstring参照)。
BUILD_PAIR_COLORS_MONO: "tuple[tuple[int, int], ...]" = tuple((c, c) for c in COLOR_SET_6)
# 診断用: 全色ペア (異色込み、25通り) — 同色制限が品質悪化の主因か切り分ける。
BUILD_PAIR_COLORS_FULL: "tuple[tuple[int, int], ...]" = tuple(
    (t, b) for t in COLOR_SET_6 for b in COLOR_SET_6
)
BUILD_PAIR_COLORS: "tuple[tuple[int, int], ...]" = BUILD_PAIR_COLORS_MONO

# 安全弁
MAX_BUILD_STEPS: int = FULL_BOARD_CAP


def _trigger_well_bonus(board: Board) -> float:
    """トリガー列(井戸)ボーナス: 最低列が2番目に低い列よりどれだけ低いか。"""
    heights = sorted(board.height_of(c) for c in range(BOARD_COLS))
    return float(heights[1] - heights[0])


def _cheap_score(board: Board) -> float:
    """(a)+(c)+トリガー列 のみの安価スコア (simulate 不要)。"""
    return (
        W_ADJ_PAIR_SQ * _adjacent_same_color_pair_sq_sum(board)
        - W_STRUCTURE_PENALTY * _structure_penalty(board)
        + W_TRIGGER_WELL * _trigger_well_bonus(board)
    )


def _enumerate_build_placements(
    board: Board,
    pair_colors: "tuple[tuple[int, int], ...]" = BUILD_PAIR_COLORS,
) -> "list[tuple[Board, list[tuple[int, int]]]]":
    """指定した色ペア集合×22配置を列挙する。"""
    candidates: "list[tuple[Board, list[tuple[int, int]]]]" = []
    for top, bot in pair_colors:
        for rotation in range(4):
            max_col = BOARD_COLS if rotation in (0, 2) else BOARD_COLS - 1
            for col in range(max_col):
                placed = _place_pair(board, top, bot, col, rotation)
                if placed is not None:
                    candidates.append(placed)
    return candidates


def saturated_chain_count_video(
    board: Board,
    sim: "ChainSimulator | None" = None,
    beam_width: int = 50,
    probe_pool_multiplier: int = 3,
    buffer_cells: int = BUFFER_EMPTY_CELLS,
    max_steps: int = MAX_BUILD_STEPS,
    pair_colors: "tuple[tuple[int, int], ...]" = BUILD_PAIR_COLORS,
) -> tuple[float, int]:
    """mode="video" 本気ビルダー: 複数フロンティア保持ビームサーチ。

    Returns:
        (raw_chain_count, steps_taken) — raw は current_max_chain 未満なら
        呼出側で max(current_max_chain, raw) を取る (絶対条件は上位層で保証)。
    """
    sim = sim or ChainSimulator()
    if board.is_dead():
        return 0.0, 0

    target_cells = FULL_BOARD_CAP - buffer_cells
    frontier: "list[Board]" = [board.copy()]
    terminal_chains: "list[float]" = []
    steps = 0

    while frontier and steps < max_steps:
        next_candidates: "list[tuple[float, Board]]" = []
        still_building: "list[Board]" = []

        for b in frontier:
            if b.count_puyos() >= target_cells:
                terminal_chains.append(_probe_potential(b, sim, COLOR_SET_6))
                continue
            raw = _enumerate_build_placements(b, pair_colors)
            if not raw:
                terminal_chains.append(_probe_potential(b, sim, COLOR_SET_6))
                continue
            safe = [(cb, cells) for cb, cells in raw if not _creates_ignition(b, cells)]
            if not safe:
                # デッドロック: 安価スコア最良の候補を実際にsimulateして打ち切り確定。
                best_cb = max(raw, key=lambda bc: _cheap_score(bc[0]))[0]
                result = sim.simulate(best_cb)
                terminal_chains.append(float(result.chain_count))
                continue
            still_building.append(b)
            for cb, _ in safe:
                next_candidates.append((_cheap_score(cb), cb))

        if not still_building:
            break

        # 安価スコアで probe 対象を絞る (全candidateへのprobeはコスト過大)
        next_candidates.sort(key=lambda x: x[0], reverse=True)
        probe_pool = next_candidates[: beam_width * probe_pool_multiplier]

        scored: "list[tuple[float, Board]]" = []
        for cheap_s, cb in probe_pool:
            potential = _probe_potential(cb, sim, COLOR_SET_6)
            total = cheap_s + W_PROBE_POTENTIAL * potential
            scored.append((total, cb))
        scored.sort(key=lambda x: x[0], reverse=True)
        frontier = [cb for _, cb in scored[:beam_width]]
        steps += 1

    if not terminal_chains and frontier:
        terminal_chains = [_probe_potential(b, sim, COLOR_SET_6) for b in frontier]

    best = max(terminal_chains) if terminal_chains else 0.0
    return best, steps


# ============================
# ベンチ本体
# ============================

BOARDS_NPZ = Path("data/indicators_v2/boards/v29.npz")


def main() -> None:
    print("=== mode=video 本気ビルダー 品質確認 (小サンプル) ===")
    data = np.load(str(BOARDS_NPZ), allow_pickle=True)
    grids = data["grids"]
    rng = np.random.default_rng(1)
    n = min(20, len(grids))
    idxs = rng.choice(len(grids), size=n, replace=False)
    boards = [Board.from_list(grids[i].tolist()) for i in idxs]
    boards = [b for b in boards if not b.is_dead()]

    sim = ChainSimulator()

    print("\n### 診断: 同色ペア限定 vs 全色ペア (beam_width=10, 5盤面) ###")
    small_sample = boards[:5]
    for label, pair_colors in (
        ("同色ペアのみ(110候補/枝)", BUILD_PAIR_COLORS_MONO),
        ("全色ペア(550候補/枝)", BUILD_PAIR_COLORS_FULL),
    ):
        print(f"--- {label} ---")
        times: list[float] = []
        improved = 0
        for board in small_sample:
            cur = iv.current_max_chain(board, sim).raw
            t0 = time.perf_counter()
            raw, steps = saturated_chain_count_video(
                board, sim, beam_width=10, pair_colors=pair_colors,
            )
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            final = max(cur, raw)
            if raw > cur:
                improved += 1
            print(
                f"current_max_chain={cur:.0f} builder_raw={raw:.0f} "
                f"final(floor適用)={final:.0f} steps={steps} time={elapsed*1000:.0f}ms"
            )
        times_arr = np.array(times)
        print(
            f"  time mean={times_arr.mean()*1000:.0f}ms max={times_arr.max()*1000:.0f}ms "
            f"/ current超え={improved}/{len(times)}"
        )

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
