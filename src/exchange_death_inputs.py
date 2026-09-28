"""死亡証明の入力だけを差し替え、モデル用の盤面・受け量を保持する。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any
import numpy as np
from src.board import Board
from src.scoring import score_to_ojama


def projected_ledger(safety: Any, chain: Any, score: float, receiver: int) -> Any:
    """台帳の複製だけで最大打ち返しを反映し、過去の相殺・着地を再生する。"""
    ledger = deepcopy(safety.ledger)
    key = (receiver, chain.chain_id)
    amount = int(score_to_ojama(score, elapsed_sec=safety.elapsed[key]).ojama_count)
    totals = dict(ledger.totals)
    totals[key] = (amount, True)
    ledger.observe(ledger.game, totals, ledger.dropped)
    return ledger


def projected_pending(safety: Any, chain: Any, score: float, receiver: int) -> int:
    """外部監査用に、複製台帳の受け側残量だけを返す。"""
    return projected_ledger(safety, chain, score, receiver).pending[receiver]


def death_inputs(projection: Any, overlay: Any, latest: tuple, incoming: list) -> dict:
    """既存の応手入力を基準に、死亡専用フラグの変更だけを局所適用する。"""
    tracker = overlay.tracker
    net = list(projection.safety.ledger.pending if projection.death_pending_ledger else incoming)
    replies, credit = projection._receivers(tracker, latest, net)
    boards, replies, certain = projection._death_boards(tracker, tuple(s.board for s in latest), replies, credit)
    boards, replies, hidden = list(boards), list(replies), [None, None]
    verified = [projection.safety.ledger.verified(i) for i in range(2)]
    engine = getattr(projection, 'hidden_death', None)
    for idx in range(2):
        chain = tracker.latest_chain(f'{idx+1}P')
        bound = engine.maximum(chain) if engine is not None and projection._chaining(tracker, idx) else None
        if bound is None:
            continue
        key = (idx, chain.chain_id)
        if projection.safety is None or key not in projection.safety.elapsed:
            continue
        score = bound['score'] + chain.drop_bonus_score
        if projection.death_pending_ledger:
            revised = projected_ledger(projection.safety, chain, score, idx)
            net[idx], verified[idx] = revised.pending[idx], revised.verified(idx)
        else:
            elapsed = projection.safety.elapsed[key]
            delta = int(score_to_ojama(score, elapsed_sec=elapsed).ojama_count) - int(
                score_to_ojama(chain.provisional_score, elapsed_sec=elapsed).ojama_count)
            net[idx] = max(0, net[idx]-delta)
        board = Board()
        board._grid = np.asarray(bound['options'][0]['board'], dtype=board._grid.dtype)
        boards[idx] = replies[idx] = board
        credit[idx], certain[idx], hidden[idx] = 0, True, bound
    return dict(incoming=net, boards=tuple(boards), replies=tuple(replies), credit=credit,
                certain=certain, hidden=hidden, verified=verified)


def hidden_proof(projection: Any, bound: dict, queue: tuple, incoming: int,
                 hands: int, elapsed: float, stamp: float) -> dict:
    """最大打ち返しの全同点盤面が死亡する場合だけ証明成立とする。"""
    from src.exchange_event_multilanding import cached_proof
    projection.hidden_death.mark_used(bound, stamp)
    results = []
    for value in bound['options']:
        board = Board()
        board._grid = np.asarray(value['board'], dtype=board._grid.dtype)
        result = cached_proof(projection, board, queue, incoming, hands, elapsed, 0)
        results.append(result)
        if not result['dead']:
            break
    return dict(dead=all(r['dead'] for r in results), reason='hidden_maximum_response',
        maximum_score=bound['score'], tied_boards=len(bound['options']), checked=len(results), proofs=results)
