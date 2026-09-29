"""複数着弾を全応手で検証する既定OFF実験。打切りは死亡にしない。"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, combinations_with_replacement
import math
from typing import Callable, Any

import numpy as np

from src.board import Board, BOARD_COLS, BOARD_ROWS, COLOR_UNKNOWN
from src.chain import ChainSimulator
from src.indicators_v2 import _enumerate_placements
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.scoring import (OJAMA_MAX_DROP_PER_TURN, calculate_chain_score,
                         score_to_ojama, compute_effective_rate)

COLORS = (1, 2, 3, 4, 5)
PAIR_SIZE = 2
MAX_SEARCH_NODES = 20000
MAX_LANDINGS = BOARD_ROWS
PROOF_CACHE_SIZE = 2048
Optimistic = Callable[[Board, np.ndarray, int, float], float]


class SearchCutoff(Exception):
    """探索の不完了を、全枝死亡と区別する。"""


@dataclass
class Search:
    """一回の証明の上限と診断。盤面は呼び出し元を変更しない。"""
    optimistic: Optimistic
    elapsed: float
    limit: int = MAX_SEARCH_NODES
    nodes: int = 0
    simulator: ChainSimulator = field(default_factory=lambda: ChainSimulator(
        exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED))

    def tick(self) -> None:
        """枝刈りする代わりに、証明全体を不確実として止める。"""
        self.nodes += 1
        if self.nodes > self.limit:
            raise SearchCutoff('node_limit')

    def responses(self, board: Board, queue: tuple, hands: int) -> list[tuple[Board, int]]:
        """手数内の全配置と連鎖後盤面を残し、同盤面は最大相殺量へ楽観統合する。"""
        frontier, all_replies = [(board, 0)], {}
        for hand in range(hands):
            pair = queue[PAIR_SIZE*hand:PAIR_SIZE*(hand+1)]
            pairs = [pair] if len(pair) == PAIR_SIZE and all(v in COLORS for v in pair) else list(
                combinations_with_replacement(COLORS, PAIR_SIZE))
            expanded = {}
            for current, sent in frontier:
                for colors in pairs:
                    for _, placed in _enumerate_placements(current, colors, self.simulator):
                        self.tick()
                        result = self.simulator.simulate(placed)
                        if result.final_board.is_dead():
                            continue
                        score = calculate_chain_score(result).total_score
                        ojama = score_to_ojama(score, elapsed_sec=self.elapsed)
                        amount = ojama.ojama_count + int(ojama.leftover_score > 0)
                        key = result.final_board._grid.tobytes()
                        previous = expanded.get(key, (None, -1))[1]
                        expanded[key] = (result.final_board, max(previous, sent+amount))
            frontier = list(expanded.values())
            for key, (reply, sent) in expanded.items():
                all_replies[key] = (reply, max(sent, all_replies.get(key, (None, -1))[1]))
        return list(all_replies.values())

    def step(self, board: Board, pending: int, queue: tuple, hands: int,
             credit: int) -> tuple[list[tuple[Board, int]], dict]:
        """現盤面の楽観火力を再計算し、最大相殺量を全応手へ無償で与える。"""
        padded = (queue + (0,) * (PAIR_SIZE*PAIR_SIZE))[:PAIR_SIZE*PAIR_SIZE]
        optimistic = self.optimistic(board, np.asarray(padded), hands, self.elapsed)
        if not math.isfinite(optimistic):
            raise SearchCutoff('nonfinite_response')
        if optimistic + credit >= pending:
            raise SearchCutoff('optimistic_cancel')
        replies = self.responses(board, queue, hands)
        maximum = max([optimistic] + [sent for _, sent in replies]) + credit
        remaining = max(0, pending-math.ceil(maximum))
        dropped = min(remaining, OJAMA_MAX_DROP_PER_TURN)
        row = dict(incoming=pending, optimistic_send=optimistic, maximum_send=maximum,
                   cancelled=pending-remaining, dropped=dropped, remaining=remaining-dropped,
                   responses=len(replies))
        survivors = {}
        for reply, _ in replies:
            # 端数の全列配置を列挙し、ランダムな一配置の死亡を根拠にしない。
            for columns in combinations(range(BOARD_COLS), dropped % BOARD_COLS):
                self.tick()
                landed = self.simulator.drop_ojama_with_remainder_columns(reply, dropped, columns)
                if not landed.is_dead():
                    survivors[landed._grid.tobytes()] = (landed, remaining-dropped)
        return list(survivors.values()), row


def prove_multilanding(board: Board, queue: tuple, incoming: int, hands: int,
                       elapsed: float, optimistic: Optimistic, credit: int = 0,
                       node_limit: int = MAX_SEARCH_NODES) -> dict:
    """全応手が窒息した場合だけ死亡。生存枝・UNKNOWN・打切りは拒否する。"""
    if np.any(board._grid == COLOR_UNKNOWN):
        return dict(dead=False, reason='unknown_board', rounds=[], nodes=0)
    if hands < 1 or board.is_dead():
        return dict(dead=False, reason='invalid_start', rounds=[], nodes=0)
    search, rounds = Search(optimistic, elapsed, node_limit), []
    frontier, consumed = [(board.copy(), incoming)], 0
    try:
        for turn in range(MAX_LANDINGS):
            budget = hands if turn == 0 else 1
            next_states, detail = {}, []
            for current, pending in frontier:
                remaining_queue = () if turn > 0 and hands > 1 else queue[PAIR_SIZE*consumed:]
                survivors, row = search.step(current, pending, remaining_queue,
                    budget, credit if turn == 0 else 0)
                detail.append(row)
                for landed, left in survivors:
                    if left <= 0:
                        return dict(dead=False, reason='surviving_response', rounds=rounds+[detail], nodes=search.nodes)
                    key = landed._grid.tobytes()
                    previous = next_states.get(key, (None, left))[1]
                    next_states[key] = (landed, min(previous, left))
            rounds.append(detail)
            if not next_states:
                return dict(dead=True, reason='all_responses_dead', rounds=rounds, nodes=search.nodes)
            frontier, consumed = list(next_states.values()), consumed+budget
    except SearchCutoff as exc:
        return dict(dead=False, reason=str(exc), rounds=rounds, nodes=search.nodes)
    return dict(dead=False, reason='landing_limit', rounds=rounds, nodes=search.nodes)


def cached_proof(projection: Any, board: Board, queue: tuple, incoming: int,
                 hands: int, elapsed: float, credit: int) -> dict:
    """時間は得点換算レートだけに依存するため、同一の証明入力を再利用する。"""
    grid = board._grid
    key = (grid.tobytes(), grid.shape, grid.dtype.str, queue, incoming, hands,
           compute_effective_rate(elapsed), credit)
    cache = projection.multi_landing_cache
    if key not in cache:
        if len(cache) >= PROOF_CACHE_SIZE:
            cache.pop(next(iter(cache)))
        cache[key] = prove_multilanding(board, queue, incoming, hands, elapsed,
                                       projection._optimistic_response, credit)
    return cache[key]


def evaluate_multilanding(projection: Any, overlay: Any, latest: tuple, incoming: list,
                          hands: tuple, t_sec: float, value: dict, context: dict | None = None) -> dict:
    """E22の死亡を維持し、確実な入力でのみ複数着弾の証明を追加する。"""
    replies, credit = projection._receivers(overlay.tracker, latest, incoming)
    boards, replies, certain = projection._death_boards(
        overlay.tracker, tuple(s.board for s in latest), replies, credit)
    hidden = [None, None]
    if context is not None:
        incoming, boards, replies, certain, credit, hidden = (context[k] for k in
            ('incoming', 'boards', 'replies', 'certain', 'credit', 'hidden'))
    dead, audits = list(value['dead_sides']), []
    for i, board in enumerate(replies):
        side = f'{i+1}P'
        blockers = [('single_landing', incoming[i] <= (0 if hidden[i] else OJAMA_MAX_DROP_PER_TURN)),
            ('already_dead', side in dead), ('uncertain_completion', not certain[i]),
            ('unknown_budget', not projection._known_budget(overlay.tracker, 1-i, t_sec)),
            ('unverified_attack', not (context['verified'][i] if context is not None
                                      else projection._verified_attack(overlay.tracker, 1-i))),
            ('unknown_board', bool(np.any(boards[i]._grid == COLOR_UNKNOWN)))]
        reason = next((name for name, blocked in blockers if blocked), None)
        if reason is None and hidden[i] is None and getattr(projection, 'safety', None) is not None:
            reason = projection.safety.blocker(projection, overlay.tracker, i, t_sec)
        result = dict(dead=False, reason=reason) if reason else _proof(projection,
            board, latest[i].queue, incoming[i], hands[i],
            overlay.tracker._score_elapsed, credit[i], hidden[i], t_sec)
        audits.append(result)
        if result['dead']:
            dead.append(side)
    value['multi_landing'] = audits
    value['dead_sides'] = dead
    if len(dead) == 1:
        from src.exchange_event_landing import UNAVOIDABLE_DEATH_PROBABILITY
        value.update(source='unavoidable_death', p1=UNAVOIDABLE_DEATH_PROBABILITY
                     if dead[0] == '1P' else 1-UNAVOIDABLE_DEATH_PROBABILITY)
    elif len(dead) > 1:
        from src.exchange_event_landing import logit_mean
        value.update(source='S3_landing', p1=logit_mean(value['base_p1'], value['gfe_p1']))
    return value


def _proof(projection: Any, board: Board, queue: np.ndarray, incoming: int, hands: int,
           elapsed: float, credit: int, hidden: dict | None, stamp: float) -> dict:
    """隠し段の最大応手は死亡専用の全同点盤面証明へ渡す。"""
    queue = tuple(int(v) for v in queue)
    if hidden is not None:
        from src.exchange_death_inputs import hidden_proof
        return hidden_proof(projection, hidden, queue, incoming, hands, elapsed, stamp)
    return cached_proof(projection, board, queue, incoming, hands, elapsed, credit)
