"""E25既定OFF: 交換独立台帳と、複数着弾死亡だけの入力整合条件。"""
from __future__ import annotations

from typing import Any
import math
import numpy as np

from src.board import Board, COLOR_UNKNOWN
from src.board_state_machine import BoardState
from src.chain_detector import CHAIN_MECHANISM_FORMULA_READ
from src.exchange_completion_recovery import CompletionRecovery, COLORS
from src.exchange_event_tracker import S3_END_QUIET_SEC
from src.exchange_pending_ledger import PendingLedger
from src.ojama_accounting import CHAIN_TOTAL_MIN_SCORE
from src.scoring import BASE_SCORE_PER_PUYO, score_to_ojama


class ColorScoreGuard:
    """得点で説明できない色消失を拒否し、盤面/NEXTの静穏を確認する。"""

    def __init__(self) -> None:
        self.accepted: Board | None = None
        self.score: float | None = None
        self.key: tuple | None = None
        self.since = 0.
        self.reason = 'missing_stable_board'

    def observe(self, board: Board, queue: tuple, score: float | None, stamp: float) -> None:
        """不整合盤面を次回比較の基準へ昇格させない。"""
        key = (board._grid.tobytes(), queue)
        if key != self.key:
            self.key, self.since = key, stamp
        if score is None or not math.isfinite(score):
            self.reason = 'missing_board_score'
            return
        if np.any(board._grid == COLOR_UNKNOWN):
            self.reason = 'unknown_board'
            return
        if self.accepted is not None:
            lost = sum(max(0, int(np.count_nonzero(self.accepted._grid == c))
                - int(np.count_nonzero(board._grid == c))) for c in COLORS)
            delta = score-self.score
            budget = math.floor(delta/BASE_SCORE_PER_PUYO) if delta >= CHAIN_TOTAL_MIN_SCORE else 0
            if lost > budget or delta < 0:
                self.reason = 'color_score_mismatch'
                return
        self.accepted, self.score, self.reason = board.copy(), score, None

    def blocker(self, stamp: float) -> str | None:
        """既存終了確認と同じ静穏期間を使い、瞬間的なNEXT/盤面揺れを確定しない。"""
        if self.reason is not None:
            return self.reason
        return 'board_unsettled' if stamp-self.since < S3_END_QUIET_SEC else None


class LandingStateSafety:
    """状態は評価ラッパーに閉じ込め、基礎指標と本番設定を変更しない。"""

    def __init__(self, ledger_enabled: bool = True, guard_enabled: bool = True,
                 recovery_enabled: bool = True) -> None:
        self.ledger_enabled = ledger_enabled
        self.guard_enabled = guard_enabled
        self.recovery_enabled = recovery_enabled
        self.reset()

    def reset(self) -> None:
        """試合境界で予告・盤面検証・復元候補をまとめて捨てる。"""
        self.ledger = PendingLedger()
        self.recovery = CompletionRecovery()
        self.guards = [ColorScoreGuard(), ColorScoreGuard()]
        self.chains: dict[tuple[int, int], Any] = {}
        self.elapsed: dict[tuple[int, int], float] = {}

    def observe(self, overlay: Any, result: Any, snapshot: Any, stamp: float) -> None:
        """交換が閉じても連鎖台帳は保持し、各得点の改訂を同じIDへ反映する。"""
        tracker, projection = overlay.tracker, overlay._landing_projection
        for chain in tracker.current.chains if tracker.current else ():
            idx = int(chain.side == '2P')
            key = (idx, chain.chain_id)
            self.chains[key] = chain
            self.elapsed.setdefault(key, tracker._score_elapsed)
            event = (result.p1, result.p2)[idx].chain_event
            if (chain is tracker.latest_chain(chain.side) and event is not None
                    and event.mechanism == CHAIN_MECHANISM_FORMULA_READ and event.trigger_sec >= chain.trigger_sec):
                self.recovery.observe(chain, event.chain_count, stamp, event.total_score)
        totals = {key: (int(score_to_ojama(c.provisional_score,
            elapsed_sec=self.elapsed[key]).ojama_count), self._verified(c)) for key, c in self.chains.items()}
        self.ledger.observe(overlay._game, totals,
            (snapshot.total_dropped_to_p1, snapshot.total_dropped_to_p2))
        for idx, side in enumerate((result.p1, result.p2)):
            if side.state == BoardState.STABLE and side.confirmed_board is not None:
                queue = (*(getattr(side, 'next_pair', ()) or ()), *(getattr(side, 'dnext_pair', ()) or ()))
                self.guards[idx].observe(side.confirmed_board, queue, side.score, stamp)

    @staticmethod
    def _verified(chain: Any) -> bool:
        """現交換に限らず、残っている予告の実測裏付けを調べる。"""
        return ((chain.formula_total is not None or chain.score_ready_reason == 'score_finalize')
                and chain.provisional_score <= max(chain.formula_total or 0, chain.score_delta or 0))

    def blocker(self, projection: Any, tracker: Any, idx: int, stamp: float) -> str | None:
        """この条件の呼出先は追加の複数着弾判定だけに限定する。"""
        if not self.guard_enabled:
            return None
        if projection._chaining(tracker, idx):
            chain = tracker.latest_chain(f'{idx+1}P')
            midchain = getattr(projection, 'midchain', None)
            if midchain is not None and midchain.verified(chain):
                return None
            return None if self.recovery.consistent(chain) else 'completion_prefix_unverified'
        return self.guards[idx].blocker(stamp)

    def signature(self, projection: Any, tracker: Any, stamp: float) -> tuple:
        """整合状態が変わったフレームで、同じ盤面でも再評価を行う。"""
        return tuple(self.blocker(projection, tracker, i, stamp) for i in range(2))
