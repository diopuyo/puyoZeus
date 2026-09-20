"""W15-A: 「cell は連鎖 or ツモ着地以外で変わらない」物理拘束を強制。

ロジック (ユーザー要望):
    - ツモは必ず設置される (next_pair が変化 = アクティブツモが落下した)
    - 一度置いたぷよは連鎖が発生しない限り動かない (= score 変化なし時)

これを board-level で強制:
    1. stable_board を保持 (前回の確定盤面)
    2. event 判定:
        - chain_event: score が大きく増加
        - tsumo_event: next_pair が変化
        - ojama_event: pending_ojama が増加
    3. event なし (= 安定状態): 各 cell の色は stable と一致するはず
        - 現フレームで stable.color != EM だが cur == EM → 元の色を維持 (誤消失補正)
        - 現フレームで stable.color != EM だが cur != stable.color → 元の色を維持 (誤色補正)
        - cur == stable: 何もしない
    4. event あり (= 変化許容): cur をそのまま採用 (新着 puyo を許可)、stable 更新

注意:
    - 初回フレームは stable=None なので cur をそのまま採用 + stable 更新
    - 隠し段は stable 維持しない (推論不確定のため)
    - OJAMA は降下時 chain_event=False でも変化するので tsumo_event=True に近い扱い
      → ojama_event を別途検出
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.board import (
    BOARD_COLS, COLOR_EMPTY, COLOR_OJAMA,
    COLOR_UNKNOWN, HIDDEN_ROWS, Board,
)


# score 増加でこの値超なら chain
CHAIN_DELTA_THRESHOLD: int = 30


@dataclass
class PuyoStabilityRefiner:
    """安定状態 (event なし) で cell 色変化を抑制。"""

    chain_delta_threshold: int = CHAIN_DELTA_THRESHOLD

    stable: dict[str, Board | None] = field(
        default_factory=lambda: {"1P": None, "2P": None}
    )
    prev_score: dict[str, int | None] = field(
        default_factory=lambda: {"1P": None, "2P": None}
    )
    prev_next: dict[str, tuple[int, int] | None] = field(
        default_factory=lambda: {"1P": None, "2P": None}
    )
    prev_pending_ojama: dict[str, int] = field(
        default_factory=lambda: {"1P": 0, "2P": 0}
    )

    n_em_recovered: int = 0
    n_color_held: int = 0

    def reset(self) -> None:
        self.stable = {"1P": None, "2P": None}
        self.prev_score = {"1P": None, "2P": None}
        self.prev_next = {"1P": None, "2P": None}
        self.prev_pending_ojama = {"1P": 0, "2P": 0}
        self.n_em_recovered = 0
        self.n_color_held = 0

    def refine(
        self,
        side: str,
        board: Board,
        score: int | None,
        cur_next: tuple[int, int] | None,
        pending_ojama: int = 0,
    ) -> Board:
        """1 side の board を物理拘束で refine。"""
        out = board.copy()
        stable = self.stable.get(side)
        prev_s = self.prev_score.get(side)
        prev_n = self.prev_next.get(side)
        prev_oj = self.prev_pending_ojama.get(side, 0)

        # event 判定
        chain_event = (
            prev_s is not None and score is not None
            and score > prev_s + self.chain_delta_threshold
        )
        tsumo_event = (
            prev_n is not None and cur_next is not None
            and prev_n != cur_next
        )
        ojama_event = pending_ojama > prev_oj + 5

        any_event = chain_event or tsumo_event or ojama_event

        # 初回 or event あり: cur をそのまま採用、stable 更新
        if stable is None or any_event:
            self.stable[side] = board.copy()
        else:
            # 安定状態: cell 不変を強制
            for vrow in range(12):
                row = vrow + HIDDEN_ROWS
                for col in range(BOARD_COLS):
                    cur = int(out.get(row, col))
                    st = int(stable.get(row, col))
                    if cur == st:
                        continue
                    # stable が色付き、cur が EM → 元の色維持 (誤消失補正)
                    if st not in (COLOR_EMPTY, COLOR_UNKNOWN) and cur == COLOR_EMPTY:
                        out.set(row, col, st)
                        self.n_em_recovered += 1
                        continue
                    # stable が色付き、cur が別の色 → 元の色維持 (誤色補正)
                    if (
                        st not in (COLOR_EMPTY, COLOR_UNKNOWN)
                        and cur not in (COLOR_EMPTY, COLOR_UNKNOWN)
                        and cur != st
                    ):
                        out.set(row, col, st)
                        self.n_color_held += 1
                        continue
                    # stable が EM、cur が色 → 安定中なのに新着、CNN 過敏
                    # ただし保守的に許容 (盤面外要因あるかもしれない)
                    pass
            # stable も current の保持側で更新 (UNKNOWN 等は更新せず)
            self.stable[side] = out.copy()

        # 状態保存
        if score is not None:
            self.prev_score[side] = score
        if cur_next is not None:
            self.prev_next[side] = cur_next
        self.prev_pending_ojama[side] = pending_ojama
        return out


__all__ = [
    "CHAIN_DELTA_THRESHOLD",
    "PuyoStabilityRefiner",
]
