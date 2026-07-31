"""全消し演出検出 (Phase R)。

ユーザ仕様 (2026-04-27):
    - フィールド上部 25% に「全消し」テキスト (オレンジ白抜き)
    - score が 0 でないかつフィールドにぷよが 1 つもなければ全消し状態
    - 全消し演出は次にぷよを消すまで持続

user確定 (2026-07-22, docs/PUYO_RULES_CONFIRMED_2026-07-22.md):
    全消し = 「おじゃま含め盤面が完全に空」。色ぷよが 0 でもおじゃまぷよが
    1 つでも残っていれば全消しではない (旧実装は色ぷよ 0 個のみで判定しており誤り)。
    2100 点ボーナスは即座には加算されず「次の連鎖発火時」に上乗せされる
    (`src/scoring.py` ALL_CLEAR_BONUS / `src/chain_detector.py` VideoChainTracker
    が正しくこの仕様で実装済み)。

判定方式:
    1. **シンプル方式 (推奨)**: `is_all_clear(board, score)` で score>0 かつ board 完全に空
    2. 視覚方式: テンプレ NCC で「全消し」テキストを検出 (将来拡張)

利用例:
    if is_all_clear(board, score):
        all_clear_pending = True
        # 次の連鎖発火時に ALL_CLEAR_BONUS=2100 を加算
"""
from __future__ import annotations

from dataclasses import dataclass

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    Board,
)


@dataclass(frozen=True)
class AllClearResult:
    """全消し判定結果。

    Attributes:
        is_all_clear: 全消し状態か
        score: 判定に使った score
        n_puyo_on_field: フィールド上のぷよ数 (おじゃま含む)
        n_color_puyo: 色ぷよ数 (おじゃま除く)
        reason: 判定理由
    """
    is_all_clear: bool
    score: int
    n_puyo_on_field: int
    n_color_puyo: int
    reason: str = ""


def count_puyos_on_field(
    board: Board, exclude_ojama: bool = False,
    exclude_unknown: bool = True,
) -> int:
    """フィールド上のぷよ数をカウントする。

    Args:
        board: 評価する盤面
        exclude_ojama: True ならおじゃまぷよを除外
        exclude_unknown: True なら unknown セル (隠し段) を除外
    """
    n = 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            v = int(board.get(r, c))
            if v == COLOR_EMPTY:
                continue
            if exclude_unknown and v == COLOR_UNKNOWN:
                continue
            if exclude_ojama and v == COLOR_OJAMA:
                continue
            n += 1
    return n


def is_all_clear(
    board: Board,
    score: int,
    require_score_nonzero: bool = True,
    require_no_color_puyo: bool = True,
    require_no_ojama: bool = True,
) -> AllClearResult:
    """全消し状態を判定する。

    判定ロジック (user確定仕様 2026-07-22):
        - score == 0 → 試合冒頭、全消しではない
        - フィールドが完全に空 (色ぷよ 0 個 かつ おじゃまぷよ 0 個) + score > 0 → 全消し
        - おじゃまぷよが 1 つでも残っていれば全消しではない
          (旧実装は「色ぷよ 0 個」のみで判定しており誤りだった → 修正)

    Args:
        board: 評価する盤面
        score: 現在のスコア (画面 OCR or ChainResult から取得)
        require_score_nonzero: True なら score > 0 を要求
        require_no_color_puyo: True なら色ぷよ 0 個を要求
        require_no_ojama: True ならおじゃまぷよ 0 個も要求 (2026-07-22 新設、
            既定 True = 正しい全消し定義。後方互換のため optional 引数として追加)

    Returns:
        AllClearResult
    """
    n_total = count_puyos_on_field(board, exclude_ojama=False)
    n_color = count_puyos_on_field(board, exclude_ojama=True)
    n_ojama = n_total - n_color

    if require_score_nonzero and score <= 0:
        return AllClearResult(
            is_all_clear=False,
            score=score,
            n_puyo_on_field=n_total,
            n_color_puyo=n_color,
            reason="score=0 (試合冒頭または初期状態)",
        )
    if require_no_color_puyo and n_color > 0:
        return AllClearResult(
            is_all_clear=False,
            score=score,
            n_puyo_on_field=n_total,
            n_color_puyo=n_color,
            reason=f"色ぷよが {n_color} 個残っている",
        )
    if require_no_ojama and n_ojama > 0:
        return AllClearResult(
            is_all_clear=False,
            score=score,
            n_puyo_on_field=n_total,
            n_color_puyo=n_color,
            reason=f"おじゃまぷよが {n_ojama} 個残っている (全消しではない)",
        )
    return AllClearResult(
        is_all_clear=True,
        score=score,
        n_puyo_on_field=n_total,
        n_color_puyo=n_color,
        reason=f"全消し (盤面完全に空、score={score})",
    )


__all__ = [
    "AllClearResult",
    "count_puyos_on_field",
    "is_all_clear",
]
