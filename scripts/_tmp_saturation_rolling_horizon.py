"""飽和連鎖量 v2 プロトタイプ: rolling horizon (1手ごと浅ビーム→最良1手確定) + 3本柱評価。

コーディネータからの先行研究反映指示 (2026-07-22) に基づくプロトタイプ:
    - MCTS不採用 (理想ツモ=色自由前提のため乱数プレイアウトの意義が薄い)。
    - 評価関数は機能3本柱のみ (形テンプレ無し):
        (a) 同色隣接ペア数の二乗和 (連結の伸ばしやすさ)
        (b) 1手打ち込みポテンシャル (takapt 30通り 1個落としで到達する連鎖数)
        (c) 段差 + 穴 (埋没空きマス) ペナルティ
    - 物理制約保持: 1手= 2個1組 (縦/横 × 6列 = 最大22パターン)。
      既存 _enumerate_placements / _place_pair_to_board (III-3) を流用。
    - 早期発火は「打ち切り」: 置いた瞬間に simulate して消去が起きたら
      その時点の chain_count を最終到達連鎖数として確定・build終了。
    - 色数バリアント (6色 vs 4色) を両方計測。
    - サニティ: raw > 20 が出たら実装バグを疑う。

本スクリープは検証専用 (_tmp_ prefix、indicators_v2.py へは未統合)。
ベンチ結果を見て統合方針を決める。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_saturation_rolling_horizon
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
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN,
    COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_PURPLE, Board,
)
from src.chain import ChainSimulator, MIN_ERASE_COUNT  # noqa: E402
import src.indicators_v2 as iv  # noqa: E402

# ============================
# 定数 (コーディネータ指示の初期値・チューニング対象)
# ============================

FULL_BOARD_CAP: int = BOARD_ROWS * BOARD_COLS  # = 78
BUFFER_EMPTY_CELLS: int = 5

# ビーム幅: 通常10、埋まり率85%超の終盤のみ30に拡大 (指示の出発点)。
BEAM_WIDTH_NORMAL: int = 10
BEAM_WIDTH_LATE: int = 30
LATE_GAME_FILL_THRESHOLD: float = 0.85

# 3本柱の重み (暫定・データ後決定)。
W_ADJ_PAIR_SQ: float = 1.0      # (a) 同色隣接ペア数の二乗和
W_PROBE_POTENTIAL: float = 3.0  # (b) 1手打ち込みポテンシャル (連鎖数なのでスケール大きめ)
W_STRUCTURE_PENALTY: float = 1.0  # (c) 段差+穴ペナルティ
# (d) 追試: 自手が即発火(total_erased>0)する場合のソフトペナルティ。
# 「打ち切りで確定」は許容するが、非発火の選択肢がある限りは避けたい
# (実験1で W_IGNITION_PENALTY=0 だと全盤面が1手目で即発火し過小評価と判明)。
W_IGNITION_PENALTY: float = 0.0  # 実験1=0 (指示通りそのまま) / 実験2で変更

COLOR_SET_6: tuple[int, ...] = (
    COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_PURPLE,
)
# ぷよは実質5色 (紫含む)。「6色/4色」= 5色フル vs 4色制限の意図で解釈。
COLOR_SET_4: tuple[int, ...] = (COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW)

MAX_BUILD_STEPS: int = FULL_BOARD_CAP  # 安全弁


# ============================
# 3本柱 評価関数
# ============================


def _adjacent_same_color_pair_sq_sum(board: Board) -> float:
    """(a) 同色隣接ペア数の二乗和。

    色ごとに「上下左右で同色隣接しているセル境界(エッジ)」の数を数え、
    色別にその数を2乗して合計する (連結が育つほど二乗で加速的に評価)。
    """
    grid = board._grid
    per_color_edges: dict[int, int] = {}
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            color = int(grid[r, c])
            if color in (COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN):
                continue
            # 右・下のみ見て二重カウントを防ぐ
            if c + 1 < BOARD_COLS and int(grid[r, c + 1]) == color:
                per_color_edges[color] = per_color_edges.get(color, 0) + 1
            if r + 1 < BOARD_ROWS and int(grid[r + 1, c]) == color:
                per_color_edges[color] = per_color_edges.get(color, 0) + 1
    return float(sum(v * v for v in per_color_edges.values()))


def _structure_penalty(board: Board) -> float:
    """(c) 段差 (bumpiness) + 穴 (埋没空きマス) ペナルティ。"""
    heights = [board.height_of(c) for c in range(BOARD_COLS)]
    bumpiness = sum(abs(heights[c + 1] - heights[c]) for c in range(BOARD_COLS - 1))
    holes = 0
    grid = board._grid
    for c in range(BOARD_COLS):
        seen_puyo = False
        for r in range(BOARD_ROWS):
            color = int(grid[r, c])
            if color not in (COLOR_EMPTY,):
                seen_puyo = True
            elif seen_puyo:
                holes += 1
    return float(bumpiness + holes)


def _probe_potential(board: Board, sim: ChainSimulator, colors: tuple[int, ...]) -> float:
    """(b) 1手打ち込みポテンシャル。各列に理想色を1個仮に落として発火する

    連鎖数の最大値 (takapt 30通り相当、色数は colors に従う)。
    """
    best = 0
    for col in range(BOARD_COLS):
        row = _drop_row_local(board, col)
        if row is None:
            continue
        for color in colors:
            work = board.copy()
            work.set(row, col, color)
            chain = sim.simulate(work).chain_count
            if chain > best:
                best = chain
    return float(best)


def _drop_row_local(board: Board, col: int) -> "int | None":
    height = board.height_of(col)
    if height >= BOARD_ROWS:
        return None
    return BOARD_ROWS - 1 - height


def _evaluate_candidate(board: Board, sim: ChainSimulator, colors: tuple[int, ...]) -> float:
    """3本柱の加重合計スコア (大きいほど良い)。"""
    a = _adjacent_same_color_pair_sq_sum(board)
    b = _probe_potential(board, sim, colors)
    c = _structure_penalty(board)
    return W_ADJ_PAIR_SQ * a + W_PROBE_POTENTIAL * b - W_STRUCTURE_PENALTY * c


# ============================
# 2個1組 配置列挙 (III-3 _enumerate_placements と同じ物理制約)
# ============================


def _drop_one_inplace(board: Board, col: int, color: int) -> bool:
    row = _drop_row_local(board, col)
    if row is None:
        return False
    board.set(row, col, color)
    return True


def _drop_two_in_column(board: Board, col: int, upper: int, lower: int) -> bool:
    if board.height_of(col) > BOARD_ROWS - 2:
        return False
    if not _drop_one_inplace(board, col, lower):
        return False
    return _drop_one_inplace(board, col, upper)


def _place_pair(
    board: Board, top: int, bot: int, col: int, rotation: int,
) -> "tuple[Board, list[tuple[int, int]]] | None":
    """rotation: 0=縦TOP上,1=横TOP左,2=縦BOT上,3=横BOT左 (III-3 と同一仕様)。

    Returns:
        (配置後盤面, 新規設置セル座標リスト[(row,col),...]) または None (不可)。
    """
    work = board.copy()
    if rotation in (0, 2):
        if not (0 <= col < BOARD_COLS):
            return None
        upper, lower = (top, bot) if rotation == 0 else (bot, top)
        height = board.height_of(col)
        if height > BOARD_ROWS - 2:
            return None
        row_lower = BOARD_ROWS - 1 - height
        row_upper = row_lower - 1
        if not _drop_two_in_column(work, col, upper, lower):
            return None
        return work, [(row_lower, col), (row_upper, col)]
    if not (0 <= col < BOARD_COLS - 1):
        return None
    left, right = (top, bot) if rotation == 1 else (bot, top)
    row_left = _drop_row_local(work, col)
    if row_left is None:
        return None
    if not _drop_one_inplace(work, col, left):
        return None
    row_right = _drop_row_local(work, col + 1)
    if row_right is None:
        return None
    if not _drop_one_inplace(work, col + 1, right):
        return None
    return work, [(row_left, col), (row_right, col + 1)]


def _group_size_from_cell(board: Board, row: int, col: int, color: int, cap: int) -> int:
    """(row,col) を含む同色連結成分のサイズを cap で早期打切りして返す。"""
    visited: "set[tuple[int, int]]" = {(row, col)}
    stack: "list[tuple[int, int]]" = [(row, col)]
    size = 0
    while stack:
        r, c = stack.pop()
        size += 1
        if size >= cap:
            return size
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS):
                continue
            if (nr, nc) in visited:
                continue
            if int(board._grid[nr, nc]) == color:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return size


def _creates_ignition(board: Board, new_cells: "list[tuple[int, int]]") -> bool:
    """新規設置セルのいずれかが即座に4連結以上を作るか (物理simulate不要の軽量判定)。"""
    for row, col in new_cells:
        color = int(board._grid[row, col])
        if color in (COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN):
            continue
        if _group_size_from_cell(board, row, col, color, MIN_ERASE_COUNT) >= MIN_ERASE_COUNT:
            return True
    return False


def _enumerate_all_pair_placements(
    board: Board, colors: tuple[int, ...],
) -> "list[tuple[Board, list[tuple[int, int]]]]":
    """全色ペア (colors×colors) × 22 配置パターンを列挙する (物理制約⑤準拠)。

    候補数: len(colors)^2 × 22 (満杯列は自動除外)。
    5色なら最大 550、4色なら最大 352。
    """
    candidates: "list[tuple[Board, list[tuple[int, int]]]]" = []
    for top in colors:
        for bot in colors:
            for rotation in range(4):
                max_col = BOARD_COLS if rotation in (0, 2) else BOARD_COLS - 1
                for col in range(max_col):
                    placed = _place_pair(board, top, bot, col, rotation)
                    if placed is not None:
                        candidates.append(placed)
    return candidates


# ============================
# rolling horizon 本体
# ============================


def saturated_chain_count_v2(
    board: Board,
    sim: "ChainSimulator | None" = None,
    colors: tuple[int, ...] = COLOR_SET_6,
    buffer_cells: int = BUFFER_EMPTY_CELLS,
    ignition_penalty: float = W_IGNITION_PENALTY,
) -> tuple[float, int, str]:
    """rolling horizon (1手ごと浅ビーム→最良1手確定) による飽和連鎖量。

    Returns:
        (raw_chain_count, steps_taken, stop_reason)
        stop_reason: "target_reached" (buffer到達→最終takapt発火)
                     "early_ignition" (構築中に意図せず発火→その時点で確定)
                     "no_candidates" (デッドロック)
                     "max_steps" (安全弁)
    """
    sim = sim or ChainSimulator()
    if board.is_dead():
        return 0.0, 0, "dead"
    target_cells = FULL_BOARD_CAP - buffer_cells
    current = board.copy()
    steps = 0
    while True:
        if current.count_puyos() >= target_cells:
            # 目標到達: 最終発火 (takapt 1個落とし相当で最大連鎖を測定)
            final_chain = _probe_potential(current, sim, colors)
            return final_chain, steps, "target_reached"
        if steps >= MAX_BUILD_STEPS:
            final_chain = _probe_potential(current, sim, colors)
            return final_chain, steps, "max_steps"

        fill_ratio = current.count_puyos() / float(FULL_BOARD_CAP)
        beam_width = BEAM_WIDTH_LATE if fill_ratio > LATE_GAME_FILL_THRESHOLD else BEAM_WIDTH_NORMAL

        raw_candidates = _enumerate_all_pair_placements(current, colors)
        if not raw_candidates:
            final_chain = _probe_potential(current, sim, colors)
            return final_chain, steps, "no_candidates"

        # 第0段 (軽量・simulate不要): 自手が即4連結以上を作る候補を除外する。
        # 局所BFS判定のみ (_creates_ignition) で ChainSimulator.simulate は呼ばない。
        # 非発火候補が1つも無い (デッドロック=終盤で埋まっている) 場合のみ
        # 発火候補もプールに含める (重要④: 回避不能な早期発火は測定に採用)。
        safe = [(b, cells) for b, cells in raw_candidates if not _creates_ignition(current, cells)]
        pool = safe if safe else raw_candidates
        forced_ignition = not safe

        # 第1段 (安価): (a)+(c) のみで上位 beam_width に絞る
        cheap_scored = [
            (
                W_ADJ_PAIR_SQ * _adjacent_same_color_pair_sq_sum(cand)
                - W_STRUCTURE_PENALTY * _structure_penalty(cand),
                cand,
            )
            for cand, _ in pool
        ]
        cheap_scored.sort(key=lambda x: x[0], reverse=True)
        pruned = [cand for _, cand in cheap_scored[:beam_width]]

        # 第2段 (高価): (b) 1手打ち込みポテンシャルは終盤 (beam_width拡大と同条件、
        # fill_ratio > LATE_GAME_FILL_THRESHOLD) のみ適用する。
        # 序盤〜中盤は (a)+(c) のみ (simulate 呼び出しゼロ) で確定し、
        # ステップ数が多くなりがちな低埋まり率盤面でのコスト爆発を防ぐ
        # (実験3で全ステップ probe した結果、平均1.5秒/盤面まで悪化したための対策)。
        use_probe = fill_ratio > LATE_GAME_FILL_THRESHOLD
        best_score = float("-inf")
        best_board: "Board | None" = None
        for cand in pruned:
            b = _probe_potential(cand, sim, colors) if use_probe else 0.0
            score = (
                W_ADJ_PAIR_SQ * _adjacent_same_color_pair_sq_sum(cand)
                + W_PROBE_POTENTIAL * b
                - W_STRUCTURE_PENALTY * _structure_penalty(cand)
            )
            if score > best_score:
                best_score = score
                best_board = cand

        assert best_board is not None
        if forced_ignition:
            # 回避不能な早期発火: 物理simulateして到達連鎖数を確定測定。
            result = sim.simulate(best_board)
            return float(result.chain_count), steps + 1, "early_ignition"

        current = best_board
        steps += 1


# ============================
# ベンチ本体
# ============================


BOARDS_NPZ = Path("data/indicators_v2/boards/v29.npz")
N_SAMPLE = 20


def main() -> None:
    print("=== saturated_chain_count v2 (rolling horizon, 3本柱, 2個1組) ベンチ ===")
    data = np.load(str(BOARDS_NPZ), allow_pickle=True)
    grids = data["grids"]
    rng = np.random.default_rng(1)
    n = min(N_SAMPLE, len(grids))
    idxs = rng.choice(len(grids), size=n, replace=False)
    boards = [Board.from_list(grids[i].tolist()) for i in idxs]
    boards = [b for b in boards if not b.is_dead()]
    print(f"サンプル盤面数: {len(boards)}")

    sim = ChainSimulator()

    print("\n### 実験3: 非発火候補ハード優先(局所BFSで安全候補プール限定) ###")
    for label, colors in (("6色(実質5色)", COLOR_SET_6), ("4色制限", COLOR_SET_4)):
        times: list[float] = []
        raws: list[float] = []
        reasons: dict[str, int] = {}
        for board in boards:
            t0 = time.perf_counter()
            raw, steps, reason = saturated_chain_count_v2(board, sim, colors=colors)
            times.append(time.perf_counter() - t0)
            raws.append(raw)
            reasons[reason] = reasons.get(reason, 0) + 1
        times_arr = np.array(times)
        raws_arr = np.array(raws)
        print(f"--- {label} ---")
        print(
            f"時間: mean={times_arr.mean()*1000:.1f}ms median={np.median(times_arr)*1000:.1f}ms "
            f"max={times_arr.max()*1000:.1f}ms"
        )
        print(
            f"raw: mean={raws_arr.mean():.2f} max={raws_arr.max():.0f} "
            f"分位点={np.percentile(raws_arr, [50, 75, 90, 95]).round(2)}"
        )
        print(f"停止理由内訳: {reasons}")
        n_over20 = int((raws_arr > 20).sum())
        print(f"サニティ(raw>20件数): {n_over20}")

    print("\n=== current_max_chain / 旧saturation_chain(単puyo版) / v2(安全優先) との比較 ===")
    for board in boards[:10]:
        cur = iv.current_max_chain(board, sim)
        old_sat = iv.saturation_chain(board, simulator=sim)
        raw_v2, steps, reason = saturated_chain_count_v2(board, sim, colors=COLOR_SET_6)
        print(
            f"current_max_chain={cur.raw:.0f} 旧saturation_chain(単puyo)={old_sat.raw:.0f} "
            f"v2(2個1組rolling,安全優先)={raw_v2:.0f} steps={steps} reason={reason}"
        )

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
