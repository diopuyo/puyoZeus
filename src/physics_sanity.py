"""
ぷよぷよの物理/ゲームルールに基づく色分類サニティチェック。

CNN で読み取った盤面に対して「物理的/論理的にありえない状態」を検出する。
違反セルは色分類誤認の候補として人手レビューや学習フィードバックの対象にする。

検出対象:
    1. 空中浮遊 (AIRBORNE): 非空セルの真下が空で落下中でもない
        → 直下セルが空と誤認された、または当該セルが色と誤認された
    2. 未消去 4+ 連結 (UNRESOLVED_CHAIN): 4 個以上の同色連結が存在するのに
       連鎖が始まっていない (単一フレーム判定、弱い信号)
        → 連結内のどれかが他色との混同 or 連結外のどれかが同色誤認

将来拡張:
    - 時間方向不整合 (色 A → 色 B 直接遷移、空挟まず) は TemporalSmoother に分離予定
    - 「連鎖中」フラグ検出 (消去モーション) → 違反を抑制
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    Board,
    HIDDEN_ROWS,
)
from src.chain import ChainSimulator, MIN_ERASE_COUNT


class ViolationKind(str, Enum):
    """物理サニティ違反の種類。"""
    AIRBORNE = "airborne"                       # 空中浮遊
    UNRESOLVED_CHAIN = "unresolved_chain"       # 4+ 連結が消えていない


@dataclass(frozen=True)
class PhysicsViolation:
    """1 件の違反情報。"""
    kind: ViolationKind
    row: int
    col: int
    color: int
    detail: str                                 # 人間向け説明


class PhysicsSanityChecker:
    """
    盤面 1 枚に対する物理/論理ルール違反を列挙する。

    Usage:
        checker = PhysicsSanityChecker()
        violations = checker.check(board)
        for v in violations:
            print(v.kind, v.row, v.col, v.detail)
    """

    def __init__(self, simulator: ChainSimulator | None = None) -> None:
        self._simulator = simulator or ChainSimulator()

    def check(self, board: Board) -> list[PhysicsViolation]:
        """盤面をチェックして違反リストを返す。"""
        violations: list[PhysicsViolation] = []
        violations.extend(self._check_airborne(board))
        violations.extend(self._check_unresolved_chain(board))
        return violations

    def _check_airborne(self, board: Board) -> Iterator[PhysicsViolation]:
        """
        空中浮遊ぷよを検出する。

        ルール: 非空セル (r, c) があり、かつ真下のセル (r+1, c) が空で
        r+1 が盤面内である場合 → 落下により埋まっているはずなので違反。

        例外:
            - 隠し段 (row < HIDDEN_ROWS) はカメラ上部の不安定ゾーンなのでスキップ。
            - 最下段 (row == BOARD_ROWS - 1) は床なので問題なし。
        """
        for row in range(HIDDEN_ROWS, BOARD_ROWS - 1):
            for col in range(BOARD_COLS):
                here = board.get(row, col)
                if here == COLOR_EMPTY:
                    continue
                below = board.get(row + 1, col)
                if below != COLOR_EMPTY:
                    continue
                yield PhysicsViolation(
                    kind=ViolationKind.AIRBORNE,
                    row=row,
                    col=col,
                    color=here,
                    detail=(
                        f"row={row} col={col} color={here} は浮いている "
                        f"(直下 row={row+1} col={col} が空)"
                    ),
                )

    def _check_unresolved_chain(self, board: Board) -> Iterator[PhysicsViolation]:
        """
        4+ 連結が残っている状態を検出する (単一フレームなので弱い信号)。

        通常ぷよぷよでは 4 個以上の同色連結が接地した瞬間に連鎖が開始する。
        単一フレームでこの状態が見えている場合、以下のいずれか:
          a) 本当に連鎖開始直前の 1 フレーム (真陽性 false positive)
          b) 連結内の 1 個が他色との混同 (色分類誤認)
          c) 連結外の 1 個が同色誤認で大連結に見えている

        真陽性判定は時間方向で要確認なので、ここでは「要目視」として列挙する。
        """
        groups = self._simulator.find_groups(board)
        for group in groups:
            # おじゃまは消えないので除外
            if group.color == COLOR_OJAMA:
                continue
            if group.size < MIN_ERASE_COUNT:
                continue
            # 代表セル（最上段側）を違反位置にする
            top_cell = min(group.cells, key=lambda rc: (rc[0], rc[1]))
            yield PhysicsViolation(
                kind=ViolationKind.UNRESOLVED_CHAIN,
                row=top_cell[0],
                col=top_cell[1],
                color=group.color,
                detail=(
                    f"size={group.size} color={group.color} グループが未消去 "
                    f"(4+ 連結、連鎖開始直前 or 色誤認の可能性)"
                ),
            )


def summarize_violations(violations: list[PhysicsViolation]) -> str:
    """違反リストの人間可読サマリ。"""
    if not violations:
        return "物理サニティ: 違反なし"
    counts: dict[ViolationKind, int] = {}
    for v in violations:
        counts[v.kind] = counts.get(v.kind, 0) + 1
    parts = [f"{k.value}={c}" for k, c in counts.items()]
    return f"物理サニティ違反 {len(violations)} 件: {' '.join(parts)}"
