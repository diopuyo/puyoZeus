"""Phase F (B-4) 回し入れ (rotation maneuver) 巧拙トラッカー.

STABLE フレーム間の cell 消失を物理推論 (連鎖シミュレーション) と照合し、
物理的に説明できない消失を「回し入れ候補」としてカウントする。

ぷよぷよでは、上級者は完成形を一度崩して別の形に作り変える「回し入れ」が
頻繁に発生する。連鎖の発火・おじゃま落下では説明できない盤面変化を捉え、
プレーヤーの戦術理解度の代理指標とする。

設計方針:
  - 履歴 (deque, max_history) に直近の STABLE board を保存
  - 新 board 追加時に前 board と比較:
      * 前 board を ChainSimulator.simulate して final_board を取得
      * final_board と新 board が一致 → 連鎖で説明可能 (物理的)
      * 一致しない → 回し入れ候補としてカウント
  - 連鎖直前直後 (前 board に連鎖あり) は判定対象外 (大きな変動を除外)
  - score = clamp(rotation_count / 履歴長, 0, 1)
"""

from __future__ import annotations

from collections import deque
from typing import Deque

from src.board import Board
from src.chain import ChainResult, ChainSimulator
from src.indicators import (
    ROTATION_TRACKER_MAX_HISTORY,
    SCORE_MAX,
    SCORE_MIN,
)


class RotationTracker:
    """STABLE フレーム間の物理整合性を検証して「回し入れ」発生頻度を追跡する.

    Attributes:
        max_history: 履歴に保持する最大 board 数。古いものから自動破棄。
        rotation_count: これまで検出した回し入れ候補数 (履歴中)。
        decisions_count: これまで判定した回数 (連鎖直前直後の除外を引いた数)。
    """

    def __init__(
        self,
        max_history: int = ROTATION_TRACKER_MAX_HISTORY,
        simulator: ChainSimulator | None = None,
    ) -> None:
        """Tracker を初期化する.

        Args:
            max_history: 履歴の最大長 (回し入れ判定の denominator にも使用)。
            simulator: ChainSimulator の差し替え (None なら内部生成)。
        """
        self._max_history: int = max_history
        self._history: Deque[Board] = deque(maxlen=max_history)
        # 判定結果 (True=回し入れ候補) も同じ長さで保持し、score 計算に使用
        self._decisions: Deque[bool] = deque(maxlen=max_history)
        self._simulator: ChainSimulator = simulator or ChainSimulator()

    @property
    def max_history(self) -> int:
        """履歴の最大長."""
        return self._max_history

    @property
    def rotation_count(self) -> int:
        """履歴中の回し入れ候補数."""
        return sum(1 for d in self._decisions if d)

    @property
    def decisions_count(self) -> int:
        """履歴中の判定対象数 (連鎖除外後)."""
        return len(self._decisions)

    @property
    def score(self) -> float:
        """rotation_skill スコア [0, 1].

        rotation_count / max_history で正規化。
        判定が一度も行われていないか max_history が 0 なら 0.0 を返す。
        """
        denom = max(1, self._max_history)
        ratio = self.rotation_count / float(denom)
        return max(SCORE_MIN, min(SCORE_MAX, ratio))

    def update(self, board: Board) -> None:
        """新 STABLE board を追加し、前 board との整合性を判定する.

        前 board が存在し、かつ前 board に連鎖が発生しない場合のみ判定対象。
        連鎖が発生する盤面 (= 大変動が予期される) はノイズが大きいため除外。

        Args:
            board: 評価対象の新 STABLE board。
        """
        if len(self._history) > 0:
            prev = self._history[-1]
            decision = self._is_rotation_candidate(prev, board)
            if decision is not None:
                self._decisions.append(decision)
        # 履歴追加 (deque maxlen で自動的に古いものは破棄)
        self._history.append(board.copy())

    def reset(self) -> None:
        """履歴と判定結果をすべて破棄する (試合切替等で使用)."""
        self._history.clear()
        self._decisions.clear()

    def _is_rotation_candidate(
        self, prev: Board, current: Board,
    ) -> bool | None:
        """prev → current の遷移が物理推論で説明できないかを判定する.

        Returns:
            True: 物理的に説明できない (回し入れ候補)。
            False: 物理的に説明可能 (連鎖消去等)。
            None: 判定対象外 (前 board に連鎖あり等の除外ケース)。
        """
        # 同一盤面なら回し入れ無し (静的状態)
        if self._boards_equal(prev, current):
            return False

        prev_result = self._safe_simulate(prev)
        if prev_result is None:
            # シミュレーション失敗 → 判定対象外
            return None
        # 前 board が連鎖を起こす場合は大変動が予期されるため除外
        if prev_result.chain_count > 0:
            return None

        # 前 board の重力適用後の final_board と現 board が一致すれば物理的
        # (連鎖無しでも apply_gravity による微変動は許容)
        if self._boards_equal(prev_result.final_board, current):
            return False
        return True

    def _safe_simulate(self, board: Board) -> ChainResult | None:
        """ChainSimulator.simulate を例外安全に呼び出す."""
        try:
            return self._simulator.simulate(board)
        except Exception:
            return None

    @staticmethod
    def _boards_equal(a: Board, b: Board) -> bool:
        """2 つの Board の grid が完全一致するかを判定する."""
        return bool((a._grid == b._grid).all())
