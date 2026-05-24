"""
隠し段トラッカー

連続するフレーム間の変化から、隠し段 (row 0) の状態を時系列で推論する。

基本原理:
    - 物理ルール: 可視最上段が空の列 → 隠し段も空 (ImageReader で処理済)
    - 時系列ルール: 前フレームから増えた puyo 数と、隣接フレームで検出された
      「次のぷよ」表示から、隠し段に入った puyo を推定できる
    - 回し入れ検出: 操作フレーム内で puyo が可視領域から消えた場合、
      同列の隠し段に移動したと推定

本モジュールは基礎的な時系列推論を提供し、将来的に:
    - 次ぷよ OCR による色予測
    - 操作puyo 軌跡追跡
    - 回し入れパターン認識
    と組み合わせて精度向上させる設計。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    HIDDEN_ROWS,
    Board,
)


# ============================
# 定数定義
# ============================

# 1 回の落下で同時に置かれるぷよ数 (通常 2, ちょうど積みなら 1 や 4 の可能性)
DEFAULT_PLACEMENT_COUNT: int = 2

# 状態不定の初期値
UNKNOWN_PROB: float = 0.5


# ============================
# データクラス
# ============================


@dataclass
class HiddenRowHypothesis:
    """
    隠し段の状態仮説。

    Attributes:
        definitely_empty: (col 単位) 空が確定している列 (重力ルール)。
        possibly_occupied: (col, 推定色) 回し入れの可能性がある列。
        confidence: 仮説全体の信頼度 (0.0〜1.0)。
    """
    definitely_empty: set[int] = field(default_factory=set)
    possibly_occupied: dict[int, int | None] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class BoardDiff:
    """
    2 フレーム間の盤面差分。

    Attributes:
        added_cells: 追加されたセル (row, col, color) のリスト。
        removed_cells: 消去されたセル。
        hidden_unchanged: 隠し段で変化なしの列。
    """
    added_cells: list[tuple[int, int, int]] = field(default_factory=list)
    removed_cells: list[tuple[int, int, int]] = field(default_factory=list)
    hidden_unchanged: set[int] = field(default_factory=set)


# ============================
# HiddenRowTracker
# ============================


class HiddenRowTracker:
    """
    隠し段の状態を時系列で追跡するクラス。

    連続する Board スナップショットを受け取り、各列の隠し段状態について
    仮説を蓄積する。回し入れの検出や次ぷよ情報との統合は段階的に拡張。

    Usage:
        tracker = HiddenRowTracker()
        for frame in frames:
            board = reader.read_board(frame, region)
            tracker.observe(board)
            hypothesis = tracker.current_hypothesis()
    """

    def __init__(self) -> None:
        self._last_board: Board | None = None
        self._hypothesis = HiddenRowHypothesis()

    # ============================
    # 公開メソッド
    # ============================

    def observe(self, board: Board) -> HiddenRowHypothesis:
        """
        新しい盤面を観測して仮説を更新する。

        Args:
            board: 最新の観測盤面。

        Returns:
            HiddenRowHypothesis: 更新後の仮説。
        """
        if self._last_board is None:
            self._hypothesis = self._physics_only_hypothesis(board)
        else:
            self._hypothesis = self._update_with_history(
                previous=self._last_board, current=board,
            )
        self._last_board = board.copy()
        return self._hypothesis

    def current_hypothesis(self) -> HiddenRowHypothesis:
        """現在の仮説を取得する。"""
        return self._hypothesis

    def reset(self) -> None:
        """履歴をリセットする (新試合開始時など)。"""
        self._last_board = None
        self._hypothesis = HiddenRowHypothesis()

    def compute_diff(self, previous: Board, current: Board) -> BoardDiff:
        """
        2 盤面の差分を計算する。

        Args:
            previous: 前フレーム盤面。
            current: 現フレーム盤面。

        Returns:
            BoardDiff: 追加/消去されたセル。
        """
        added: list[tuple[int, int, int]] = []
        removed: list[tuple[int, int, int]] = []
        hidden_unchanged: set[int] = set()
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                before = previous.get(row, col)
                after = current.get(row, col)
                if before != after:
                    if before == COLOR_EMPTY:
                        added.append((row, col, after))
                    elif after == COLOR_EMPTY:
                        removed.append((row, col, before))
                    else:
                        removed.append((row, col, before))
                        added.append((row, col, after))
                elif row < HIDDEN_ROWS:
                    hidden_unchanged.add(col)
        return BoardDiff(
            added_cells=added,
            removed_cells=removed,
            hidden_unchanged=hidden_unchanged,
        )

    # ============================
    # 内部メソッド
    # ============================

    @staticmethod
    def _physics_only_hypothesis(board: Board) -> HiddenRowHypothesis:
        """初回観測: 物理ルールのみで仮説を構築。"""
        hypothesis = HiddenRowHypothesis()
        top_visible = HIDDEN_ROWS
        for col in range(BOARD_COLS):
            if board.get(top_visible, col) == COLOR_EMPTY:
                hypothesis.definitely_empty.add(col)
            else:
                hypothesis.possibly_occupied[col] = None
        total = BOARD_COLS
        known = len(hypothesis.definitely_empty)
        hypothesis.confidence = known / total if total > 0 else 0.0
        return hypothesis

    def _update_with_history(
        self, previous: Board, current: Board,
    ) -> HiddenRowHypothesis:
        """履歴を使って仮説を更新する。"""
        hypothesis = self._physics_only_hypothesis(current)
        diff = self.compute_diff(previous, current)

        # 追加 puyo が可視最上段に現れた場合、新規置かれた puyo の痕跡
        # 可視最上段が非空かつ前フレーム空だった列は回し入れの有力候補
        top_visible = HIDDEN_ROWS
        new_top_cols: dict[int, int] = {}
        for (row, col, color) in diff.added_cells:
            if row == top_visible and color not in (COLOR_EMPTY, COLOR_UNKNOWN):
                new_top_cols[col] = color

        # 回し入れ推定: 前フレームで同列の隠し段が既に非空 (UNKNOWN) かつ
        # 追加で top_visible に別色が出現 → 隠し段に 1 puyo 確定、色は不明
        # (より精密な色推定は次ぷよ OCR が必要)
        for col, color in new_top_cols.items():
            if col in hypothesis.possibly_occupied:
                # 既に物理ルールで UNKNOWN 扱い、そのまま
                pass

        # 簡易確信度: 確定列 + 占有予想列数を分母に占める確定率
        total = BOARD_COLS
        known = len(hypothesis.definitely_empty)
        hypothesis.confidence = known / total if total > 0 else 0.0
        return hypothesis
