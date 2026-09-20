"""PuyoErasureMonitor — STABLE 中の「色→EMPTY」遷移監視 (fail-silent 自動検知)。

設計原則:
    STABLE 中に「色あり → 空」遷移が発生することは物理ルール上あり得ない。
    (= ぷよは消えない、 連鎖中は NON-STABLE なので STABLE には来ない)
    この遷移を alert として記録し、 eval スクリプトが自動 REJECT できる
    fail-silent 構造的禁止 indicator として機能する。

stateless 実装原則:
    外部 (RecognitionPipeline など) から frame ごとに update() を呼ぶ。
    本クラス自体は state を持たない (= alerts は外部から reset() で消去)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN, Board
from src.board_state_machine import BoardState


@dataclass
class PuyoErasureMonitor:
    """STABLE 中の「色→EMPTY」遷移を自動カウントする監視クラス。

    Attributes:
        alerts: (frame_idx, row, col) のタプルリスト。
            STABLE 中に「色→EMPTY」遷移が起きた全 cell を記録。
    """

    alerts: list[tuple[int, int, int]] = field(default_factory=list)

    def update(
        self,
        frame_idx: int,
        state: BoardState,
        prev_confirmed: Board | None,
        curr_confirmed: Board | None,
    ) -> None:
        """1 frame 分の遷移を評価し、 alert があれば記録する。

        物理ルール:
            STABLE 中に「非 EMPTY/非 UNKNOWN → EMPTY」は存在しない。
            (= ぷよは重力で落下するが消えない; 連鎖は NON-STABLE 経由)

        Args:
            frame_idx: 現在の frame インデックス (ログ用)。
            state: 現在の BoardState。STABLE 以外なら skip。
            prev_confirmed: 直前の確定盤面。None なら skip。
            curr_confirmed: 現在の確定盤面。None なら skip。
        """
        if state != BoardState.STABLE:
            return
        if prev_confirmed is None or curr_confirmed is None:
            return
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                prev_v = int(prev_confirmed.get(r, c))
                curr_v = int(curr_confirmed.get(r, c))
                # 「色あり → 空」= fail-silent 構造的禁止ケース
                if (
                    prev_v not in (COLOR_EMPTY, COLOR_UNKNOWN)
                    and curr_v == COLOR_EMPTY
                ):
                    self.alerts.append((frame_idx, r, c))

    def reset(self) -> None:
        """alert リストを初期化する (試合切替時など)。"""
        self.alerts.clear()

    def count(self) -> int:
        """alert 件数を返す。"""
        return len(self.alerts)

    def to_dict(self) -> dict:
        """eval スクリプト向けシリアライズ形式を返す。

        Returns:
            dict: {
                "p_to_e_count": int,
                "alerts": [[frame_idx, row, col], ...]
            }
        """
        return {
            "p_to_e_count": self.count(),
            "alerts": [list(a) for a in self.alerts],
        }


__all__ = ["PuyoErasureMonitor"]
