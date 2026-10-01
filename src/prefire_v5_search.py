"""Phase 5: 状態を失わない既知ツモの有限探索 (既定OFF層専用)。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from src import prefire_exchange_sim as sim
from src import puyo_core_bridge as native
from src.board import COLOR_UNKNOWN
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.ojama_accounting import cancel_own_pending_then_send_surplus
from src.exchange_virtual_board import land_pending_ojama_onto_board

TOP_K = 8  # 採点前固定。得点順で候補化し、残すK本を局面価値で比較する。
SEARCH_DEPTH = 2
PAIR_SIZE = 2
CACHE_SIZE = 512
OK, MISSING, FIRING, INVALID = range(4)


@dataclass(frozen=True)
class Position:
    """火力・残し盤面・消費ツモ・未処理おじゃまを一組で扱う。"""

    board: bytes
    queue: tuple[int, ...]
    pending: int = 0
    score: int = 0
    consumed: int = 0
    chains: int = 0
    path: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Exchange:
    """両側の着弾後状態と監査用会計。"""

    sides: tuple[Position, Position]
    cancelled: tuple[int, int]
    dropped: tuple[int, int]
    dead: tuple[bool, bool]


@lru_cache(maxsize=CACHE_SIZE)
def status(raw: bytes, known: tuple[int, ...], depth: int = SEARCH_DEPTH) -> int:
    """発火遷移を未読・窒息より先に識別し、静止探索に混ぜない。"""
    board = sim._board(raw)
    if np.any(board._grid == COLOR_UNKNOWN):
        return INVALID
    if native.simulate_chain(board, exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED).chain_count:
        return FIRING
    if sim._dead(board._grid):
        return INVALID
    required = depth * PAIR_SIZE
    if len(known) < required or any(c not in sim.PLAYABLE_COLORS for c in known[:required]):
        return MISSING
    return OK


def extend(state: Position) -> tuple[Position, ...]:
    """1組を消費して全配置を解く。死亡する手も終端候補として残す。"""
    if len(state.queue) < PAIR_SIZE:
        return ()
    pair = state.queue[:PAIR_SIZE]
    if any(c not in sim.PLAYABLE_COLORS for c in pair):
        return ()
    result = []
    for p in sim._placements(sim._board(state.board), pair):
        chain = p.chain_result
        result.append(Position(chain.final_board._grid.astype(np.int8).tobytes(),
            state.queue[PAIR_SIZE:], state.pending, int(chain.exact_score), state.consumed + 1,
            int(chain.chain_count), (*state.path, (p.col, p.rotation))))
    return tuple(result)


@lru_cache(maxsize=CACHE_SIZE)
def candidates(state: Position, depth: int = SEARCH_DEPTH, k: int | None = TOP_K) -> tuple[Position, ...]:
    """最初の発火と末端の待機を全列挙後、各手数・発火/待機別に固定Kへ絞る。"""
    if status(state.board, state.queue, depth) != OK:
        return ()
    frontier, terminals = [state], []
    for hand in range(depth):
        following = []
        for previous in frontier:
            children = extend(previous)
            if not children:
                terminals.append(previous)
            for child in children:
                if child.chains or sim._dead(sim._board(child.board)._grid) or hand == depth - 1:
                    terminals.append(child)
                else:
                    following.append(child)
        frontier = following
    groups: dict[tuple[int, bool], list[Position]] = {}
    for terminal in terminals:
        groups.setdefault((terminal.consumed, bool(terminal.chains)), []).append(terminal)
    return tuple(p for group in groups.values() for p in
                 sorted(group, key=lambda x: (-x.score, x.path))[:k])


def resolve(first: Position, second: Position, attacker: int, elapsed: float) -> Exchange:
    """攻撃→応手の順で自己予告を相殺し、両者へ1ターン着弾する。残量は保持。"""
    sides = (first, second)
    pending = [p.pending for p in sides]
    cancelled = [0, 0]
    for side in (attacker, 1 - attacker):
        generated = int(sim.send_ojama(sides[side].score, elapsed))
        cancelled[side] = min(generated, pending[side])
        pending[side], pending[1-side] = cancel_own_pending_then_send_surplus(
            generated, pending[side], pending[1-side])
    boards = tuple(sim._board(p.board) for p in sides)
    landed = tuple(land_pending_ojama_onto_board(b, boards[1-i], pending[i]) for i, b in enumerate(boards))
    result = tuple(replace(p, board=landed[i][0]._grid.astype(np.int8).tobytes(),
                           pending=pending[i] - landed[i][1]) for i, p in enumerate(sides))
    return Exchange(result, tuple(cancelled), tuple(v[1] for v in landed),
                    tuple(sim._dead(sim._board(p.board)._grid) for p in result))
