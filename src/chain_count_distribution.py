"""隠し段の可能性を展開して連鎖数の「分布」を求める (2026-08-08).

設計: docs/HIDDEN_ROW_PROBABILISTIC_DESIGN_2026-08-08.md

現行のシミュレーションは隠し段の UNKNOWN を連結対象から外すため、 連鎖数が
常に 1 通りしか出ない。 本モジュールは隠し段の色確率
(src/hidden_row_probability.py) から組み合わせを展開し、 各組み合わせを
シミュレートして **連鎖数の確率分布** を返す。

    {5: 0.52, 9: 0.43, 6: 0.05}   ← 「5連鎖の可能性52% / 9連鎖の可能性43%」

## 打ち切りを黙って行わない
隠し段は最大 6 セル。 各セル 5 通りなら 15,625 通りになりリアルタイムに
乗らないため枝刈りするが、 **打ち切った事実と網羅できた確率を必ず結果に持たせる**。
「全部見た」と誤解させないため (measure した範囲を明示する規律)。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from src.board import BOARD_COLS, COLOR_EMPTY, Board
from src.chain_bitboard import simulate_single
from src.hidden_row_probability import (
    HiddenRowProbabilities,
    TOP_VISIBLE_ROW,
)

# 展開する隠し段セル数の上限 (これを超えたら確率の高いセルのみ展開する)。
# 6 セル全展開は最悪 5^6=15,625 通りで実時間に乗らないための保険。
MAX_EXPAND_CELLS: int = 4
# 累積確率がこの値を下回る枝は打ち切る (寄与が無視できるため)。
PRUNE_THRESHOLD: float = 0.01
# 1 セルあたり展開する色の上限 (確率降順)。
MAX_COLORS_PER_CELL: int = 5


@dataclass(frozen=True)
class ChainCountDistribution:
    """連鎖数の可能性分布.

    単一値しか扱えない既存コードには `most_likely` を渡せば従来通り動く
    (既存 API を壊さないための逃げ道)。
    """

    probabilities: dict[int, float]
    truncated: bool
    covered_probability: float
    n_expanded_cells: int

    @property
    def most_likely(self) -> int:
        """最も確からしい連鎖数 (分布が空なら 0)。"""
        if not self.probabilities:
            return 0
        return max(self.probabilities.items(), key=lambda kv: kv[1])[0]

    @property
    def expected(self) -> float:
        """連鎖数の期待値。"""
        if not self.probabilities:
            return 0.0
        total = sum(self.probabilities.values())
        if total <= 0:
            return 0.0
        return sum(k * v for k, v in self.probabilities.items()) / total

    @property
    def value_range(self) -> tuple[int, int]:
        """(最小連鎖数, 最大連鎖数)。分布が空なら (0, 0)。"""
        if not self.probabilities:
            return 0, 0
        keys = self.probabilities.keys()
        return min(keys), max(keys)

    def probability_of(self, chain_count: int) -> float:
        """指定連鎖数に割り当てられた確率を返す (検証用)。"""
        return self.probabilities.get(chain_count, 0.0)

    @property
    def is_single_valued(self) -> bool:
        """可能性が 1 通りに定まっているか。"""
        return len(self.probabilities) == 1


def _candidates_for_cell(
    cell_probs: dict[int, float],
) -> list[tuple[int, float]]:
    """1 セルの (色, 確率) を確率降順で返す (上限・閾値で刈る)。"""
    items = sorted(cell_probs.items(), key=lambda kv: -kv[1])
    items = [(c, p) for c, p in items if p > 0][:MAX_COLORS_PER_CELL]
    return items


def _select_expand_cols(
    hidden: HiddenRowProbabilities,
) -> tuple[list[int], bool]:
    """展開対象の列と、上限で打ち切ったかを返す.

    確定セルは展開しない。 上限を超える場合は「最も不確かなセル」
    (エントロピーが高い順) を優先して展開する — 連鎖数への影響が大きい
    のは確率が割れているセルだから。
    """
    uncertain = [c for c in hidden.cells if not c.is_certain]
    if len(uncertain) <= MAX_EXPAND_CELLS:
        return [c.col for c in uncertain], False
    uncertain.sort(key=lambda c: -c.cell.entropy())
    return [c.col for c in uncertain[:MAX_EXPAND_CELLS]], True


def _apply_assignment(
    board: Board, assignment: dict[int, int],
) -> Board:
    """隠し段に色を割り当てた盤面のコピーを返す。"""
    b = board.copy()
    for col, color in assignment.items():
        for row in range(TOP_VISIBLE_ROW):
            b.set(row, col, color)
    return b


def compute_chain_count_distribution(
    board: Board,
    hidden: HiddenRowProbabilities,
) -> ChainCountDistribution:
    """隠し段の可能性を展開して連鎖数の分布を返す。

    Args:
        board: 可視段が読み取り済みの盤面 (隠し段の値は本関数が上書きする)。
        hidden: 隠し段の色確率 (src/hidden_row_probability.py で構成)。

    Returns:
        ChainCountDistribution。 確率は展開できた分の合計で正規化する。
    """
    expand_cols, truncated_by_limit = _select_expand_cols(hidden)
    # 確定セルは先に固定する (展開対象外)
    fixed: dict[int, int] = {}
    for c in hidden.cells:
        if c.is_certain:
            color, _ = c.cell.most_likely()
            fixed[c.col] = color

    if not expand_cols:
        # 全セル確定 → 従来と同じ単一値
        result = simulate_single(_apply_assignment(board, fixed))
        return ChainCountDistribution(
            probabilities={int(result.chain_count): 1.0},
            truncated=False,
            covered_probability=1.0,
            n_expanded_cells=0,
        )

    per_cell: list[list[tuple[int, float]]] = []
    for col in expand_cols:
        cell = hidden.get(col)
        per_cell.append(_candidates_for_cell(cell.cell.probs) if cell else [])

    dist: dict[int, float] = {}
    covered = 0.0
    truncated = truncated_by_limit
    for combo in itertools.product(*per_cell):
        prob = 1.0
        for _, p in combo:
            prob *= p
        if prob < PRUNE_THRESHOLD:
            truncated = True
            continue
        assignment = dict(fixed)
        for col, (color, _) in zip(expand_cols, combo):
            assignment[col] = color
        result = simulate_single(_apply_assignment(board, assignment))
        n = int(result.chain_count)
        dist[n] = dist.get(n, 0.0) + prob
        covered += prob

    if covered <= 0:
        # 全枝が刈られた場合は最尤の 1 通りだけ計算して返す (無出力を避ける)
        assignment = dict(fixed)
        for col in expand_cols:
            cell = hidden.get(col)
            if cell is not None:
                assignment[col] = cell.cell.most_likely()[0]
        result = simulate_single(_apply_assignment(board, assignment))
        return ChainCountDistribution(
            probabilities={int(result.chain_count): 1.0},
            truncated=True,
            covered_probability=0.0,
            n_expanded_cells=len(expand_cols),
        )

    normalized = {k: v / covered for k, v in dist.items()}
    return ChainCountDistribution(
        probabilities=normalized,
        truncated=truncated,
        covered_probability=covered,
        n_expanded_cells=len(expand_cols),
    )


def empty_hidden_row_board(board: Board) -> Board:
    """隠し段を空にした盤面を返す (従来挙動の再現・比較用)。"""
    b = board.copy()
    for col in range(BOARD_COLS):
        for row in range(TOP_VISIBLE_ROW):
            b.set(row, col, COLOR_EMPTY)
    return b
