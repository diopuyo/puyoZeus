"""既知範囲をnativeで全列挙する5B探索。枝刈りは厳密なminimax境界のみ。"""
from __future__ import annotations

from functools import lru_cache
from dataclasses import replace
import hashlib
from typing import Callable

import numpy as np

from src import prefire_v5_search as base
from src.prefire_v5_value import ValueFunction
from src.indicators_v2 import SEC_PER_HAND

MAX_KNOWN_HANDS = 3
SIDE_CACHE_SIZE = 128
UNKNOWN_ROLLOUTS = 8
GAME_COLOR_COUNT = 4


def known_depth(queue: tuple[int, ...]) -> int:
    """未読より先の組は探索しない。既知3組なら必ず3組とも列挙する。"""
    depth = 0
    for offset in range(0, min(len(queue), MAX_KNOWN_HANDS * base.PAIR_SIZE), base.PAIR_SIZE):
        pair = queue[offset:offset+base.PAIR_SIZE]
        if len(pair) != base.PAIR_SIZE or any(c not in base.sim.PLAYABLE_COLORS for c in pair):
            break
        depth += 1
    return depth


@lru_cache(maxsize=SIDE_CACHE_SIZE)
def candidates(state: base.Position, depth: int | None = None, k: int | None = None,
               side: int = 0) -> tuple[base.Position, ...]:
    """側・盤面・既知組を含む完全キーで再用する。K引数は互換用で制限を許さない。"""
    if k is not None:
        raise ValueError('5Bの既知探索ではK制限を使用できない')
    depth = known_depth(state.queue) if depth is None else depth
    if depth <= 0 or base.status(state.board, state.queue, depth) != base.OK:
        return ()
    engine = base.native._native
    if engine is None or not hasattr(engine, 'prefire_terminals_py'):
        raise RuntimeError('5Bにはprefire_terminals_pyを含むpuyo_coreのビルドが必要')
    pairs = [state.queue[i:i+base.PAIR_SIZE] for i in range(0, depth*base.PAIR_SIZE, base.PAIR_SIZE)]
    terminals = engine.prefire_terminals_py(list(state.board), pairs, base.GHOST_CHAIN_RULE_ENABLED)
    result = tuple(base.Position(bytes(board), state.queue[len(path)*base.PAIR_SIZE:],
                   state.pending, score, state.consumed+len(path), chains, (*state.path, *map(tuple, path)))
                   for board, score, chains, path in terminals)
    return tuple(sorted(result, key=lambda p: (p.consumed, bool(p.chains), -p.score, p.path)))


def unique(options: tuple[base.Position, ...]) -> tuple[base.Position, ...]:
    """将来評価の全入力が同じ状態だけ統合し、列挙順で最初の手順を残す。"""
    seen, result = set(), []
    for p in options:
        key = (p.board, p.queue, p.pending, p.score, p.consumed, p.chains)
        if key not in seen:
            seen.add(key)
            result.append(p)
    return tuple(result)


def choose(states: tuple[base.Position, base.Position], attacker: int, elapsed: float,
           value_fn: ValueFunction, depth: int | None = None,
           k: int | None = None, unknown_rollouts: int = 0,
           resolver: Callable | None = None, options_fn: Callable | None = None) -> tuple | None:
    """全候補集合上のminimax。内側の境界が既存最善を超えられない枝だけ打ち切る。"""
    resolve = resolver or base.resolve
    if k is not None:
        raise ValueError('5Bの既知探索ではK制限を使用できない')
    options = tuple(options_fn(p, depth, side) if options_fn else unique(candidates(p, depth, k, side))
                    for side, p in enumerate(states))
    if not all(options):
        return None
    if depth is None and unknown_rollouts:
        sampled = response_states(states, 1-attacker, unknown_rollouts)
        if len(sampled) > 1:
            groups = tuple(options_fn(p, None, 1-attacker) if options_fn else
                           unique(candidates(p, side=1-attacker)) for p in sampled)
            return sampled_choose(options[attacker], groups, attacker, elapsed, value_fn, resolve)
    sign, best, best_signed = (1 if attacker == 0 else -1), None, -float('inf')
    responses = list(options[1-attacker])
    for attack in options[attacker]:
        worst, worst_signed = None, float('inf')
        time = elapsed + max(0, attack.consumed-1) * SEC_PER_HAND
        for response in responses:
            pair = (attack, response) if attacker == 0 else (response, attack)
            score = value_fn(resolve(*pair, attacker, time), time)
            if sign * score < worst_signed:
                worst, worst_signed = (score, attack, response), sign * score
            if worst_signed <= best_signed:
                break
        if worst_signed > best_signed:
            best, best_signed = worst, worst_signed
            # 直前最善の最悪応手を先に調べる。集合も最善値も変えない。
            responses.remove(worst[2])
            responses.insert(0, worst[2])
    return best


def response_states(states: tuple[base.Position, base.Position], side: int,
                    count: int = UNKNOWN_ROLLOUTS) -> tuple[base.Position, ...]:
    """既知2組の後の未知3組目だけ8標本化する。観測queue自体は書き換えない。"""
    state = states[side]
    known = known_depth(state.queue)
    colors = sorted({c for p in states for c in (*p.board, *p.queue) if c in base.sim.PLAYABLE_COLORS})
    if known != MAX_KNOWN_HANDS-1 or len(colors) != GAME_COLOR_COUNT or count <= 1:
        return (state,)
    # 入力・側からのみseedを決める。未来ツモや対戦結果は使わない。
    digest = hashlib.sha256(state.board + bytes(state.queue) + bytes([side])).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    draws = rng.choice(colors, size=(count, base.PAIR_SIZE))
    prefix = state.queue[:known*base.PAIR_SIZE]
    return tuple(replace(state, queue=(*prefix, *(int(c) for c in pair))) for pair in draws)


def sampled_choose(attacks: tuple, response_groups: tuple, attacker: int, elapsed: float,
                   value_fn: ValueFunction, resolver: Callable | None = None) -> tuple | None:
    """未知標本ごとの最善応手値を平均し、その期待値で攻撃を選ぶ。既知配置は全列挙。"""
    sign, best, best_signed = (1 if attacker == 0 else -1), None, -float('inf')
    resolve = resolver or base.resolve
    for attack in attacks:
        values = []
        time = elapsed + max(0, attack.consumed-1) * SEC_PER_HAND
        for responses in response_groups:
            worst = None
            for response in responses:
                pair = (attack, response) if attacker == 0 else (response, attack)
                score = value_fn(resolve(*pair, attacker, time), time)
                if worst is None or sign*score < sign*worst[0]:
                    worst = (score, attack, response)
            if worst is None:
                return None
            values.append(worst)
        score = sum(item[0] for item in values) / len(values)
        if sign*score > best_signed:
            # 監査用の応手には最悪標本を記録。表示値は全標本の期待値。
            response = min(values, key=lambda item: sign*item[0])[2]
            best, best_signed = (score, attack, response), sign*score
    return best
