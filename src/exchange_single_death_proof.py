"""旧単発判定の火力不足を、消去後盤面を含む全応手証明へ接続する。"""
from __future__ import annotations

from typing import Any

from src.board import Board
from src.exchange_event_multilanding import cached_proof


def prove_single_candidate(projection: Any, overlay: Any, latest: Any, board: Board,
                           incoming: int, hands: int, credit: int, idx: int, stamp: float) -> dict:
    """相殺前の高さだけでは断定せず、生存枝と探索打切りは不確定に戻す。"""
    reason = projection.safety.blocker(projection, overlay.tracker, idx, stamp)
    if reason is not None:
        return dict(dead=False, reason=reason)
    return cached_proof(projection, board, tuple(int(v) for v in latest.queue),
                        incoming, hands, overlay.tracker._score_elapsed, credit)
