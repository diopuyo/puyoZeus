"""W3.0: お邪魔ぷよの量子的位置推論。

予告お邪魔 N 個 vs 画面内増加 M 個の差から、隠し段 (12 段目満杯時) や
画面外列に積まれた残り N-M 個の確率分布を推論する。

入力:
    prev_board: t-1 確定盤面 (お邪魔個数 known)
    cur_board: t 観測盤面
    expected_ojama_count: 予告で落下が確定した個数 (OjamaScoreInferrer 等から)

ロジック:
    1. cur_board でお邪魔ぷよの新規出現セル数 M を数える
    2. expected_ojama_count = N、残り N-M 個は画面内に観測されてない
    3. 「12 段目に既にぷよがある列」は隠し段 (row 0) に O が積まれた可能性
       それ以外の列は単純に画面下方向に積まれていない (= 落下中、別フレームで観測)
    4. 候補列 K 個に N-M 個を分散 → 各列 (N-M)/K 確率で row 0 に O

設計上の注意:
    - 1 列に 2 個以上の隠し段 O は実装的に複雑 (row 0 の 1 セルしか無いので)
      → 1 列 1 個までと近似 (実際の試合でも稀)
    - お邪魔の確率は EMPTY と OJAMA の二値分布
"""
from __future__ import annotations

from dataclasses import dataclass

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    HIDDEN_ROWS,
    Board,
)
from src.probabilistic_board import ProbabilisticBoard


@dataclass(frozen=True)
class OjamaPositionResult:
    """推論結果。"""
    expected_ojama: int           # 期待される落下お邪魔個数
    observed_in_visible: int      # 画面内 (visible) で観測されたお邪魔の新規出現数
    inferred_in_hidden: int       # 隠し段にあると推論される個数
    candidate_cols: tuple[int, ...]  # 隠し段に O がある可能性のある列
    skipped_reason: str | None


def _count_visible_ojama_increase(prev: Board, cur: Board) -> int:
    """prev → cur で OJAMA が新規出現した可視セル数。"""
    cnt = 0
    for r in range(HIDDEN_ROWS, BOARD_ROWS):
        for c in range(BOARD_COLS):
            if (
                int(prev.get(r, c)) != COLOR_OJAMA
                and int(cur.get(r, c)) == COLOR_OJAMA
            ):
                cnt += 1
    return cnt


def _column_visible_top_filled(board: Board, col: int) -> bool:
    """指定列で 12 段目 (画面最上段、row=HIDDEN_ROWS) が非 EMPTY なら True。"""
    return int(board.get(HIDDEN_ROWS, col)) != COLOR_EMPTY


def infer_ojama_positions(
    prev_board: Board,
    cur_board: Board,
    expected_ojama: int,
) -> tuple[ProbabilisticBoard, OjamaPositionResult]:
    """予告 vs 画面内差分から隠し段 OJAMA の確率分布を推論。

    Args:
        prev_board: t-1 確定盤面
        cur_board: t 観測盤面
        expected_ojama: 予告で「落下する」と確定したお邪魔個数

    Returns:
        ProbabilisticBoard: 隠し段 OJAMA 確率を反映、可視領域は cur_board そのまま
        OjamaPositionResult: 推論統計
    """
    pboard = ProbabilisticBoard.from_board(cur_board)

    if expected_ojama <= 0:
        return pboard, OjamaPositionResult(
            expected_ojama=expected_ojama,
            observed_in_visible=0, inferred_in_hidden=0,
            candidate_cols=(), skipped_reason="no_expected_ojama",
        )

    observed = _count_visible_ojama_increase(prev_board, cur_board)
    inferred = max(0, expected_ojama - observed)

    if inferred == 0:
        return pboard, OjamaPositionResult(
            expected_ojama=expected_ojama,
            observed_in_visible=observed, inferred_in_hidden=0,
            candidate_cols=(), skipped_reason="all_visible",
        )

    # 候補列: 12 段目 (row=HIDDEN_ROWS) に既に何か積まれている列
    # = ぷよが画面外に積み上がった可能性のある列
    candidates: list[int] = [
        c for c in range(BOARD_COLS)
        if _column_visible_top_filled(cur_board, c)
    ]
    n_cand = len(candidates)
    if n_cand == 0:
        return pboard, OjamaPositionResult(
            expected_ojama=expected_ojama,
            observed_in_visible=observed, inferred_in_hidden=inferred,
            candidate_cols=(),
            skipped_reason="no_candidate_columns",
        )

    # 候補列に inferred 個を分散
    # 1 列あたり 0〜1 個 (近似)、確率 P = min(1, inferred / n_cand)
    p_per_col = min(1.0, inferred / n_cand)
    for c in range(BOARD_COLS):
        if c in candidates:
            pboard.set_distribution(0, c, {
                COLOR_OJAMA: p_per_col,
                COLOR_EMPTY: 1.0 - p_per_col,
            })
        else:
            pboard.set_certain(0, c, COLOR_EMPTY)

    return pboard, OjamaPositionResult(
        expected_ojama=expected_ojama,
        observed_in_visible=observed, inferred_in_hidden=inferred,
        candidate_cols=tuple(candidates),
        skipped_reason=None,
    )


__all__ = [
    "OjamaPositionResult",
    "infer_ojama_positions",
]
