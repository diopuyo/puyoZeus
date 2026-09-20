"""W12-B: score 増加検出 → 直前盤面の 4+ cluster を強制 EM (animation 期間)。

旧 ScorePhysicsRefiner は streak ベースで、ペア着地時に新ぷよを潰すバグあり。
本クラスはそれと違い:
    - score 増加を要件にする (chain 発火時のみ動作)
    - prev frame の 4+ cluster cell を特定
    - 該当 cell を current frame で N フレーム間 EM 強制 (animation duration)

メリット:
    - chain 由来の animation residue (まだ消えてないように見える puyo) を抑制
    - 通常時は動作しないのでペア着地に干渉しない
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA,
    HIDDEN_ROWS, Board,
)
from src.chain import ChainSimulator, MIN_ERASE_COUNT


# score 差分でこれ以上なら chain 発火扱い
CHAIN_DELTA_THRESHOLD: int = 30
# 消えた cell を EM 強制するフレーム数 (BOARD_INTERVAL_SEC=0.2 なら 1 sec)
ERASURE_HOLD_FRAMES: int = 5


@dataclass
class ScoreBasedEraser:
    """score 増加検出時に 4+ cluster だった cell を EM 強制。"""

    chain_delta_threshold: int = CHAIN_DELTA_THRESHOLD
    hold_frames: int = ERASURE_HOLD_FRAMES

    prev_board: dict[str, Board | None] = field(
        default_factory=lambda: {"1P": None, "2P": None}
    )
    prev_score: dict[str, int | None] = field(
        default_factory=lambda: {"1P": None, "2P": None}
    )
    # (side, vrow, col) -> 残り EM 強制フレーム数
    pending: dict[tuple[str, int, int], int] = field(
        default_factory=dict,
    )
    n_forced_em: int = 0

    _simulator: ChainSimulator | None = None

    def reset(self) -> None:
        self.prev_board = {"1P": None, "2P": None}
        self.prev_score = {"1P": None, "2P": None}
        self.pending = {}
        self.n_forced_em = 0

    def _get_simulator(self) -> ChainSimulator:
        if self._simulator is None:
            self._simulator = ChainSimulator()
        return self._simulator

    def _find_4plus_clusters(
        self, side: str, board: Board,
    ) -> list[tuple[str, int, int]]:
        """prev board の 4+ same-color cluster を構成する全 cell を返す。"""
        sim = self._get_simulator()
        groups = sim.find_groups(board)
        out: list[tuple[str, int, int]] = []
        for g in groups:
            if g.color == COLOR_OJAMA or g.color == COLOR_EMPTY:
                continue
            if g.size < MIN_ERASE_COUNT:
                continue
            for r, c in g.cells:
                # (側, vrow, col) として保持。隠し段は除外
                if r >= HIDDEN_ROWS:
                    out.append((side, r - HIDDEN_ROWS, c))
        return out

    def refine(
        self, side: str, board: Board, current_score: int | None,
    ) -> Board:
        """score 増加検出時に 4+ cluster cell を EM 強制。"""
        out = board.copy()

        # pending decrement (このフレームで適用しつつ残カウント減らす)
        applied = 0
        new_pending: dict[tuple[str, int, int], int] = {}
        for (s, vrow, col), remaining in self.pending.items():
            if s != side:
                # 別 side は変更なし、保持
                new_pending[(s, vrow, col)] = remaining
                continue
            row = vrow + HIDDEN_ROWS
            cur = int(out.get(row, col))
            if cur != COLOR_EMPTY:
                out.set(row, col, COLOR_EMPTY)
                applied += 1
                self.n_forced_em += 1
            if remaining > 1:
                new_pending[(s, vrow, col)] = remaining - 1
            # remaining=1 なら次フレームから消す
        self.pending = new_pending

        # score 増加判定
        prev_b = self.prev_board.get(side)
        prev_s = self.prev_score.get(side)
        if (
            prev_b is not None
            and prev_s is not None
            and current_score is not None
            and current_score > prev_s + self.chain_delta_threshold
        ):
            # chain 発火、prev board の 4+ cluster cell を pending 登録
            erased = self._find_4plus_clusters(side, prev_b)
            for cell_key in erased:
                self.pending[cell_key] = self.hold_frames

        # 状態更新
        self.prev_board[side] = board.copy()
        if current_score is not None:
            self.prev_score[side] = current_score
        return out


__all__ = [
    "CHAIN_DELTA_THRESHOLD",
    "ERASURE_HOLD_FRAMES",
    "ScoreBasedEraser",
]
