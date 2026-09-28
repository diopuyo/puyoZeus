"""未観測の発火1手を全列挙し、観測累積点と一致する候補の全員一致だけを使う。"""
from __future__ import annotations

from itertools import accumulate, combinations_with_replacement
from typing import Any

import numpy as np

from src.board import Board, COLOR_UNKNOWN
from src.chain import ChainSimulator
from src.indicators_v2 import _enumerate_placements, SEC_PER_HAND
from src.scoring import calculate_chain_score

COLORS = (1, 2, 3, 4, 5)
PAIR_SIZE = 2


def candidate(board: Board, simulator: ChainSimulator) -> dict | None:
    """UNKNOWNを含まない起点の全段累積点と終端盤面を一組として保持する。"""
    if np.any(board._grid == COLOR_UNKNOWN):
        return None
    result = simulator.simulate(board)
    score = calculate_chain_score(result)
    if not result.chain_count:
        return None
    return dict(prefix=tuple(accumulate(s.score for s in score.steps)),
                board=result.final_board._grid.tolist())


def enumerate_completions(board: Board, simulator: ChainSimulator) -> list[dict]:
    """特定色・特定列へ決め打ちせず、全色組・全合法設置の終端を残す。"""
    if np.any(board._grid == COLOR_UNKNOWN):
        return []
    unique = {}
    for pair in combinations_with_replacement(COLORS, PAIR_SIZE):
        for _, placed in _enumerate_placements(board, pair, simulator):
            value = candidate(placed, simulator)
            if value is not None:
                unique[(value['prefix'], np.asarray(value['board']).tobytes())] = value
    return list(unique.values())


class CompletionRecovery:
    """元予測の整合状態と、0段で欠落した予測の保守的復元を区別する。"""

    def __init__(self) -> None:
        self.entries: dict[int, dict] = {}
        self.audit: list[dict] = []

    def seed(self, chain: Any, event: Any, history: list, simulator: ChainSimulator) -> None:
        """発火時の起点だけを使い、後の盤面や勝敗から候補を作らない。"""
        saved = next((s for s in reversed(history) if s.t_sec < chain.trigger_sec), None)
        board = getattr(event, 'before_board', None)
        board = board if board is not None else (saved.board if saved else None)
        if board is None or np.any(board._grid == COLOR_UNKNOWN):
            return
        original = candidate(board, simulator)
        recover = chain.predicted_final_board is None and chain.predicted_chain_count == 0
        if recover and (saved is None or chain.trigger_sec-saved.t_sec > SEC_PER_HAND):
            return
        options = enumerate_completions(board, simulator) if recover else ([original] if original else [])
        if not options:
            return
        self.entries[chain.chain_id] = dict(options=options, recover=recover,
            original=(chain.predicted_final_score, chain.predicted_chain_count, chain.predicted_final_board),
            reason='awaiting_formula', last=None)

    def observe(self, chain: Any, count: int, stamp: float, observed_score: float | None = None) -> None:
        """全段の実測累積点で候補を絞り、観測との矛盾時は復元を取り消す。"""
        entry = self.entries.get(chain.chain_id)
        if entry is None or chain.formula_total is None or count < 1:
            return
        observed = chain.formula_total if observed_score is None else observed_score
        options = [v for v in entry['options'] if len(v['prefix']) >= count
                   and v['prefix'][count-1] == observed]
        entry['options'] = options
        reason = 'prefix_mismatch' if not options else 'ambiguous_completion'
        unique = len(options) == 1
        terminal_pending = unique and entry['recover'] and len(options[0]['prefix']) == count and not (
            getattr(chain, 'end_confirmed', False))
        if unique:
            reason = 'prefix_consistent'
        if terminal_pending:
            reason = 'awaiting_end_confirmation'
        if entry['recover']:
            self._assign(chain, entry, options if not terminal_pending else [])
        entry['reason'] = reason
        key = (count, observed, len(options), reason)
        if key != entry['last']:
            self.audit.append(dict(t_sec=stamp, side=chain.side, chain_id=chain.chain_id,
                count=count, observed=observed, candidates=len(options), reason=reason,
                recovered=entry['recover'] and unique and not terminal_pending))
            entry['last'] = key

    def _assign(self, chain: Any, entry: dict, options: list) -> None:
        """唯一の整合候補だけ採用し、棄却時は元の予測へ戻して架空火力を残さない。"""
        if len(options) == 1:
            value = options[0]
            chain.predicted_final_score = value['prefix'][-1]
            chain.predicted_chain_count = len(value['prefix'])
            chain.predicted_final_board = value['board']
        else:
            chain.predicted_final_score, chain.predicted_chain_count, chain.predicted_final_board = entry['original']

    def consistent(self, chain: Any) -> bool:
        """既存予測の不整合は新しい複数着弾判定のみを止める。"""
        entry = self.entries.get(chain.chain_id) if chain else None
        return entry is not None and entry['reason'] == 'prefix_consistent'
