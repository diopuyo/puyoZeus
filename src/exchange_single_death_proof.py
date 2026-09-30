"""旧単発判定の火力不足を、消去後盤面を含む全応手証明へ接続する。"""
from __future__ import annotations

from typing import Any

from src.board import Board
from src.exchange_event_multilanding import cached_proof

# D5b: これらの理由は「生存枝・相殺可能」の証明。取消の根拠になるのはこの2つだけ。
NEGATIVE_PROOF_REASONS = ('surviving_response', 'optimistic_cancel')


def _keep_legacy(result: dict) -> dict:
    """結論が出ない(blocker・探索打切り等)ので、旧判定(死亡候補)を維持する。"""
    return dict(dead=True, reason='legacy_kept', undecided_reason=result['reason'],
                rounds=result.get('rounds', []), nodes=result.get('nodes', 0))


def resolve_negative_only(result: dict) -> dict:
    """D5b: 生存枝・相殺可能の証明だけ取消、全滅証明は確定、他は旧判定維持。"""
    if result['dead'] or result['reason'] in NEGATIVE_PROOF_REASONS:
        return result
    return _keep_legacy(result)


def prove_single_candidate(projection: Any, overlay: Any, latest: Any, board: Board,
                           incoming: int, hands: int, credit: int, idx: int, stamp: float,
                           negative_only: bool = False) -> dict:
    """相殺前の高さだけでは断定せず、生存枝と探索打切りは不確定に戻す。

    negative_only=True (D5b) では blocker・打切りを取消理由にせず旧判定を維持する。
    """
    reason = projection.safety.blocker(projection, overlay.tracker, idx, stamp)
    if reason is not None:
        result = dict(dead=False, reason=reason)
    else:
        result = cached_proof(projection, board, tuple(int(v) for v in latest.queue),
                              incoming, hands, overlay.tracker._score_elapsed, credit)
    return resolve_negative_only(result) if negative_only else result
