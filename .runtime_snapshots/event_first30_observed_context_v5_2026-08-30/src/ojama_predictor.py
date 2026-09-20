"""C-2 (W-γ): score OCR 差分 → 予告おじゃま個数 → 落下スケジュール追跡.

incoming_ojama_pressure 指標の活性化のための簡易 predictor。
recognition_pipeline の SideResult.score_delta を時系列で蓄積し、
各サイドの「予告おじゃま」個数を推定する。

設計:
    - 1P が連鎖発火 → score 増加 → 2P への予告おじゃまが発生
    - score_delta // OJAMA_DIVISOR_GAME (=70) でおじゃま個数換算
    - 自サイドが連鎖発火すると、相手の予告分とキャンセルし合う
    - 連鎖終了後、未キャンセル分が盤面に落下
    - 自盤面の COLOR_OJAMA セル数を観測して「既に落下した」分を引く

簡易性:
    - 細かい時間遅延 (アニメーション) は無視 → score 加算と同タイミングで発生扱い
    - 連鎖の同時発火比較 (打ち消し) も score 加算同時で減算

import の循環を避けるため、Board は型アノテーションでのみ参照する。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_OJAMA, Board

# おじゃま換算: 公式式と一致 (1 おじゃま = 70 点)
OJAMA_DIVISOR_GAME: float = 70.0
# 落下遅延 (秒)。score 加算から実際の落下までの時間。
# 公式ぷよぷよでは 2 連鎖以上で予告 → 次のツモ操作後に落下。
# 簡略のため一定値。
OJAMA_DROP_DELAY_SEC: float = 1.5


@dataclass
class _SideOjamaState:
    """1 サイドの予告おじゃま蓄積状態."""
    # 「相手が score 増加 → 自分への予告おじゃま蓄積」
    pending: int = 0
    # 自分の累積 score (キャンセル分計算用、未使用なら 0)
    self_score_acc: int = 0
    # 直前盤面のおじゃま数 (落下検知用)
    last_ojama_count: int = 0


class OjamaPredictor:
    """両サイドの予告おじゃま個数を時系列追跡する.

    Usage:
        predictor = OjamaPredictor()
        for frame in stream:
            result = pipe.update(...)
            predictor.update(
                p1_score_delta=result.p1.score_delta,
                p2_score_delta=result.p2.score_delta,
                p1_board=result.p1.confirmed_board,
                p2_board=result.p2.confirmed_board,
            )
            inc_1p = predictor.pending_for("1P")
            inc_2p = predictor.pending_for("2P")
    """

    def __init__(self) -> None:
        self._states: dict[str, _SideOjamaState] = {
            "1P": _SideOjamaState(),
            "2P": _SideOjamaState(),
        }

    def reset(self) -> None:
        """試合切替時など、両サイド状態をクリアする."""
        for k in self._states:
            self._states[k] = _SideOjamaState()

    def update(
        self,
        p1_score_delta: int,
        p2_score_delta: int,
        p1_board: Board | None = None,
        p2_board: Board | None = None,
    ) -> None:
        """1 frame 分の更新.

        score_delta:
            score_delta は「自サイドの直前 frame からの score 増加」。
            正値は連鎖発火等。負値は通常起こらない (発火失敗等のエッジ)。
            負値は無視する。

        盤面が None なら落下分の差し引き計算をスキップ。
        """
        # 1P の連鎖 → 2P への予告
        if p1_score_delta > 0:
            ojama_to_p2 = int(p1_score_delta / OJAMA_DIVISOR_GAME)
            # キャンセル: 2P 側に既に予告がある場合 (相殺)
            self._cancel_with_pending("2P", ojama_to_p2)
        # 2P の連鎖 → 1P への予告
        if p2_score_delta > 0:
            ojama_to_p1 = int(p2_score_delta / OJAMA_DIVISOR_GAME)
            self._cancel_with_pending("1P", ojama_to_p1)
        # 落下検出: 盤面のおじゃま数増加 = 落下発生 → pending から減算
        for side, board in (("1P", p1_board), ("2P", p2_board)):
            if board is None:
                continue
            ojama_now = _count_ojama(board)
            state = self._states[side]
            dropped = max(0, ojama_now - state.last_ojama_count)
            if dropped > 0:
                state.pending = max(0, state.pending - dropped)
            state.last_ojama_count = ojama_now

    def _cancel_with_pending(self, target_side: str, ojama_count: int) -> None:
        """target_side への新規予告を、相手の (反対サイドの) pending と相殺する.

        例: 1P が連鎖 → 2P へ予告 ojama_count.
            ただし 1P 側に未消化の pending があれば、まず相殺する.
        """
        opp_side = "2P" if target_side == "1P" else "1P"
        opp_state = self._states[opp_side]
        if opp_state.pending > 0:
            cancel = min(opp_state.pending, ojama_count)
            opp_state.pending -= cancel
            ojama_count -= cancel
        # 残りを target_side の pending に追加
        self._states[target_side].pending += ojama_count

    def pending_for(self, side: str) -> int:
        """指定サイドが受ける予告おじゃま数を返す."""
        if side not in self._states:
            return 0
        return self._states[side].pending


def _count_ojama(board: Board) -> int:
    """盤面のおじゃまセル数を返す (numpy)。"""
    return int((board._grid == COLOR_OJAMA).sum())


__all__ = [
    "OJAMA_DIVISOR_GAME",
    "OJAMA_DROP_DELAY_SEC",
    "OjamaPredictor",
]
