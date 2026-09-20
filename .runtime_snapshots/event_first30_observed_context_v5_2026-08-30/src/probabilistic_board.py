"""W3.0: 確率的盤面表現 - 各セルに色の確率分布を保持。

既存 Board は各セルが単一の色コード (int) を持つ。本モジュールは
「100% 確定セル」と「量子的セル (複数色の確率分布)」を統一的に扱う。

主な用途:
    - 隠し段 (row 0, 13 段目): 画面外で観測不能、ネクスト+落下位置から推論
    - お邪魔ぷよの落下分散: 予告 N 個 vs 画面内 M 個の差を、列間で確率分布
    - CNN 信頼度低いセル: argmax だけでなく 2 位 3 位の色も保持

設計:
    - ProbabilisticCell: dict[color_code, prob] で 1 セルの確率分布
    - ProbabilisticBoard: 13 行 × 6 列 の ProbabilisticCell 配列
    - to_board(threshold) -> Board: 高信頼度セルは確定、それ以外 UNKNOWN
    - from_board(board) -> ProbabilisticBoard: 既存 Board → 全セル確率 1.0

確率分布は CERTAIN_THRESHOLD 以上で「確定」扱い、それ未満で量子的扱い。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
    HIDDEN_ROWS,
    Board,
)

# 確率分布対象の色 (UNKNOWN は確率対象外、未観測の意味で別管理)
PROB_COLORS: tuple[int, ...] = (
    COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
    COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA,
)
# 「確定」とみなす確率閾値
CERTAIN_THRESHOLD: float = 0.95


@dataclass
class ProbabilisticCell:
    """1 セルの確率分布。

    probs[color] = P(セルの色 = color)、Σ = 1.0 が原則。
    色が PROB_COLORS のいずれかに収まる前提。
    """
    probs: dict[int, float] = field(default_factory=dict)

    @classmethod
    def certain(cls, color: int) -> "ProbabilisticCell":
        """単一色 100% の確定セル。"""
        return cls(probs={color: 1.0})

    @classmethod
    def uniform(
        cls, colors: tuple[int, ...] = PROB_COLORS,
    ) -> "ProbabilisticCell":
        """指定色集合で均等分布 (完全に未確定の状態)。"""
        n = len(colors)
        if n == 0:
            return cls(probs={COLOR_EMPTY: 1.0})
        p = 1.0 / n
        return cls(probs={c: p for c in colors})

    def normalize(self) -> None:
        """確率の和を 1.0 に正規化。"""
        s = sum(self.probs.values())
        if s <= 0:
            self.probs = {COLOR_EMPTY: 1.0}
            return
        for k in self.probs:
            self.probs[k] /= s

    def most_likely(self) -> tuple[int, float]:
        """最尤色とその確率。空 dict の場合は (EMPTY, 0)."""
        if not self.probs:
            return COLOR_EMPTY, 0.0
        c, p = max(self.probs.items(), key=lambda kv: kv[1])
        return c, p

    def is_certain(self, threshold: float = CERTAIN_THRESHOLD) -> bool:
        """最尤色の確率が threshold 以上なら確定扱い。"""
        _, p = self.most_likely()
        return p >= threshold

    def get(self, color: int) -> float:
        """指定色の確率を返す (デフォルト 0)。"""
        return self.probs.get(color, 0.0)

    def entropy(self) -> float:
        """確率分布のエントロピー (確信度の逆指標)。0=確定、log(7) ≈ 1.95=完全未確定。"""
        s = 0.0
        for p in self.probs.values():
            if p > 0:
                s -= p * np.log(p)
        return float(s)

    def __repr__(self) -> str:
        items = sorted(self.probs.items(), key=lambda kv: -kv[1])[:3]
        parts = [f"{c}:{p:.2f}" for c, p in items]
        return f"PCell({', '.join(parts)})"


class ProbabilisticBoard:
    """13×6 セルの確率分布配列。

    Usage:
        pboard = ProbabilisticBoard()
        pboard.set_certain(12, 2, COLOR_RED)
        pboard.set_distribution(0, 3, {COLOR_RED: 0.6, COLOR_BLUE: 0.4})
        board = pboard.to_board()  # 確定セルだけ反映、不確定は UNKNOWN
    """

    def __init__(self) -> None:
        self._cells: list[list[ProbabilisticCell]] = [
            [ProbabilisticCell.certain(COLOR_EMPTY) for _ in range(BOARD_COLS)]
            for _ in range(BOARD_ROWS)
        ]

    @classmethod
    def from_board(cls, board: Board) -> "ProbabilisticBoard":
        """既存 Board → 全セル確率 1.0 (UNKNOWN は均等分布)."""
        pb = cls()
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                color = int(board.get(r, c))
                if color == COLOR_UNKNOWN:
                    pb._cells[r][c] = ProbabilisticCell.uniform()
                else:
                    pb._cells[r][c] = ProbabilisticCell.certain(color)
        return pb

    def cell(self, row: int, col: int) -> ProbabilisticCell:
        return self._cells[row][col]

    def set_certain(self, row: int, col: int, color: int) -> None:
        self._cells[row][col] = ProbabilisticCell.certain(color)

    def set_distribution(
        self, row: int, col: int, probs: dict[int, float],
    ) -> None:
        cell = ProbabilisticCell(probs=dict(probs))
        cell.normalize()
        self._cells[row][col] = cell

    def to_board(
        self, certain_threshold: float = CERTAIN_THRESHOLD,
    ) -> Board:
        """確定セルは確定色、それ以外は UNKNOWN として既存 Board に変換。"""
        b = Board()
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                cell = self._cells[r][c]
                if cell.is_certain(certain_threshold):
                    color, _ = cell.most_likely()
                    b.set(r, c, color)
                else:
                    b.set(r, c, COLOR_UNKNOWN)
        return b

    def total_uncertainty(self) -> float:
        """全セルのエントロピー合計 (盤面全体の不確実性)。"""
        return sum(
            self._cells[r][c].entropy()
            for r in range(BOARD_ROWS) for c in range(BOARD_COLS)
        )

    def n_uncertain(
        self, certain_threshold: float = CERTAIN_THRESHOLD,
    ) -> int:
        """確定でないセル数。"""
        return sum(
            1 for r in range(BOARD_ROWS) for c in range(BOARD_COLS)
            if not self._cells[r][c].is_certain(certain_threshold)
        )

    def iter_cells(self) -> Iterator[tuple[int, int, ProbabilisticCell]]:
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                yield r, c, self._cells[r][c]

    # ============================
    # Phase G: 最尤盤面 / Monte Carlo サンプリング
    # ============================

    def to_max_likelihood_board(self) -> Board:
        """各セルの最尤色を採用して Board を構築する.

        to_board() と異なり is_certain 判定を行わない。低確率分布でも
        最尤色を採用するため UNKNOWN セルは生成されない (UNIFORM 均等
        分布の場合は dict 順序最初の色が選ばれる)。
        確率版 indicator 計算で「最尤代表盤面」として使う。
        """
        b = Board()
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                color, _ = self._cells[r][c].most_likely()
                b.set(r, c, color)
        return b

    @property
    def mle_board(self) -> Board:
        """最尤盤面のキャッシュ版 (lazy 初期化、内部辞書を変更したら無効化されない点に注意)."""
        cache = getattr(self, "_mle_cache", None)
        if cache is None:
            cache = self.to_max_likelihood_board()
            self._mle_cache = cache
        return cache.copy()

    def sample_board(
        self, rng: np.random.Generator | None = None,
    ) -> Board:
        """各セルの確率分布から色を 1 個ずつ抽選し Board を構築する.

        Monte Carlo サンプリング用。確率 0 の色は選ばれない。
        rng=None なら numpy のデフォルト RNG を使う。
        """
        if rng is None:
            rng = np.random.default_rng()
        b = Board()
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                cell = self._cells[r][c]
                color = self._sample_cell(cell, rng)
                b.set(r, c, color)
        return b

    @staticmethod
    def _sample_cell(
        cell: ProbabilisticCell, rng: np.random.Generator,
    ) -> int:
        """1 セルの確率分布から色を 1 個抽選する."""
        if not cell.probs:
            return COLOR_EMPTY
        items = list(cell.probs.items())
        colors = np.array([c for c, _ in items], dtype=np.int64)
        probs = np.array([p for _, p in items], dtype=np.float64)
        s = probs.sum()
        if s <= 0:
            return COLOR_EMPTY
        probs /= s
        idx = int(rng.choice(len(colors), p=probs))
        return int(colors[idx])


__all__ = [
    "CERTAIN_THRESHOLD",
    "PROB_COLORS",
    "ProbabilisticBoard",
    "ProbabilisticCell",
]
