"""既存複数着弾証明を先回り盤へ接続する。結果は撃ち合い内の同一入力で共有。"""
from __future__ import annotations
from src.phase_j._live_ex_frozen.src import prefire_work_budget as work_budget

from itertools import combinations
from time import perf_counter
from typing import Any
import numpy as np
from src.phase_j._live_ex_frozen.src.board import Board, BOARD_COLS, BOARD_ROWS, COLOR_UNKNOWN
from src.phase_j._live_ex_frozen.src.exchange_event_multilanding import prove_multilanding, MAX_SEARCH_NODES
from src.phase_j._live_ex_frozen.src.exchange_event_landing import ExchangeLandingProjection, remaining_hands
from src.phase_j._live_ex_frozen.src.exchange_fast_expand import drop_ojama_fast
from src.phase_j._live_ex_frozen.src.scoring import (compute_effective_rate, OJAMA_MAX_DROP_PER_TURN,
    MARGIN_TIME_START_SEC, MARGIN_TIME_START_FROM_FIRST_MOVE_SEC)
from src.phase_j._live_ex_frozen.src.post_counter_geometry import all_landings_dead
from src.phase_j._live_ex_frozen.src.prefire_worker import LOW_SPEC_RATIO

PAIR_SIZE = 2
SIDES = (0, 1)
PROOF_LOW_SPEC_MS = 350.
MS = 1000.


def optimistic(board: Board, queue: np.ndarray, hands: int, elapsed: float) -> float:
    return ExchangeLandingProjection._optimistic_response(None, board, queue, hands, elapsed)


def prove_and_project(board: Board, queue: tuple, pending: int, hands: int,
                      elapsed: float, immediate: bool, limit: int) -> dict:
    """既に置き終わった手の着弾を先に処理し、その後は既存全応手証明へ渡す。"""
    if np.any(board._grid == COLOR_UNKNOWN) or board.is_dead():
        return dict(dead=False, reason='invalid_start', nodes=0, valid=False)
    first = min(pending, OJAMA_MAX_DROP_PER_TURN) if immediate else 0
    if immediate and all_landings_dead(board, first):
        return dict(dead=True, reason='capacity_immediate', nodes=0, valid=True,
                    drop=first, remaining=pending-first, free_cells=int(np.sum(board._grid == 0)))
    starts = [drop_ojama_fast(board, first, columns)
              for columns in combinations(range(BOARD_COLS), first % BOARD_COLS)] if immediate else [board]
    nodes, results, began = 0, [], perf_counter()
    for current in starts:
        if current.is_dead():
            continue
        left = pending-first
        if left <= 0:
            return dict(dead=False, reason='landing_complete', valid=True, nodes=nodes,
                        board=current._grid.ravel().tolist(), pending=0, drop=first)
        projected: dict = {}
        proof = prove_multilanding(current, queue, left, max(1, hands), elapsed, optimistic,
            node_limit=max(0, limit-nodes), count_first=True, projection=projected,
            max_seconds=None if work_budget.enabled() else max(0., PROOF_LOW_SPEC_MS/MS/LOW_SPEC_RATIO-(perf_counter()-began)))
        nodes += min(proof['nodes'], max(0, limit-nodes))
        results.append(proof)
        if not proof['dead']:
            return dict(proof, nodes=nodes, valid=bool(projected), projection=projected,
                        **({k: projected[k] for k in ('board', 'pending')} if projected else {}))
    fast = all(r['reason'].startswith('capacity') for r in results)
    return dict(dead=True, reason='capacity_all_branches' if fast else 'all_responses_dead',
                nodes=nodes, valid=True, proofs=results, drop=first)


class LandingV49C:
    """キャッシュは観測入力が同一の場合だけ利用し、勝敗ラベルを受け取らない。"""

    def __init__(self) -> None:
        self.fight: tuple | None = None
        self.cache: dict = {}
        self.certificates: dict = {}

    def reset(self, game: int, fight: float | None) -> None:
        identity = (game, fight)
        if self.fight is None or game != self.fight[0]:
            self.certificates = {}
        if identity != self.fight:
            self.fight, self.cache = identity, {}

    def side(self, value: dict, engine: Any, side: int, limit: int) -> dict:
        state, pending = engine.state, int(value['pending'][side])
        if pending <= 0:
            return dict(dead=False, reason='no_pending', nodes=0, valid=True, cache_hit=False)
        board = Board.from_list(np.asarray(value['boards'][side]).reshape(BOARD_ROWS, BOARD_COLS).tolist())
        schedule, other = value['landing_schedule'][side], engine.chains[1-side]
        active = value['board_prediction_active']
        if any(s.state in ('CHAIN', 'GRAVITY_SETTLE') and not active[i]
               for i, s in enumerate(state.sides)):
            return dict(dead=False, reason='chain_forecast_unavailable', nodes=0,
                        valid=False, cache_hit=False)
        immediate = bool(active[side] and schedule['certain'])
        hands = (remaining_hands(len(other['best']['scores']), other['started'], state.t)
                 if other and active[1-side] else 1)
        # 次に置く組が未観測なら、全色の現在組を無償で認めて死亡を厳しくする。
        queue = tuple(state.sides[side].queue)
        if not immediate:
            queue = (0, 0)+queue
        elapsed = state.t-value['margin_origin_sec'] if value['margin_origin_sec'] is not None else 0.
        elapsed += MARGIN_TIME_START_SEC-MARGIN_TIME_START_FROM_FIRST_MOVE_SEC
        key = (side, board._grid.tobytes(), queue, pending, hands, immediate,
               compute_effective_rate(elapsed), limit)
        hit = key in self.cache
        if not hit:
            self.cache[key] = prove_and_project(board, queue, pending, hands, elapsed, immediate, limit)
        result = dict(self.cache[key], cache_hit=hit, input_sec=state.t, hands=hands,
                      incoming=pending, confirmed=value['confirmed_observed'][side])
        # 推測した相手送り・未完了盤だけを根拠に死亡を断定しない。
        result['certified'] = bool(result['dead'] and not any(active) and
            value['confirmed_observed'][side] >= pending and
            state.sides[side].state == 'STABLE' and state.sides[side].board_t == state.t)
        if result['dead'] and not result['certified']:
            result['valid'] = False
        return result

    def apply(self, engine: Any, value: dict) -> dict:
        self.reset(engine.state.game_idx, engine.fight)
        if self.certificates:
            held = [dict(self.certificates[i], cache_hit=True, reason='certified_continuation',
                         nodes=0) if i in self.certificates else
                    dict(dead=False, valid=True, nodes=0, cache_hit=True, reason='opponent_certified')
                    for i in SIDES]
            value.update(landing_proofs=held, death_sides=list(self.certificates),
                         proof_nodes=0, physics_valid=False, prediction_kind='既存の全応手死亡証明を保持')
            return value
        audits, nodes = [], 0
        for side in SIDES:
            proof = self.side(value, engine, side, max(0, MAX_SEARCH_NODES-nodes))
            nodes += 0 if proof['cache_hit'] else proof['nodes']
            audits.append(proof)
            if proof.get('board') is not None:
                value['boards'][side] = proof['board']
                value['pending'][side] = proof['pending']
        dead = [i for i, proof in enumerate(audits) if proof.get('certified')]
        if len(dead) == 1:
            self.certificates[dead[0]] = dict(audits[dead[0]])
        value.update(landing_proofs=audits, death_sides=dead, proof_nodes=nodes,
                     physics_valid=all(p['valid'] for p in audits) and
                         not any(COLOR_UNKNOWN in b for b in value['boards']),
                     prediction_kind='全応手探索の楽観生存盤（確定未来ではない）')
        return value
