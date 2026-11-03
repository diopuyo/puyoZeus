"""可視3組で進む非発火経路から、途中の発火候補をすべて分岐保存する。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any
from src.phase_j._live_ex_frozen.src.indicators_v2 import SEC_PER_HAND
from src.phase_j._live_ex_frozen.src.prefire_search_v410 import Budget, MAX_HANDS, VISIBLE_PAIRS, SearchBudgetExpired
from src.phase_j._live_ex_frozen.src.prefire_search_v411 import RecordedPlan, choices, action_of, add_fire, add_quiet
from src.phase_j._live_ex_frozen.src.prefire_context_v410 import Context
from src.phase_j._live_ex_frozen.src.prefire_world_v410 import World

NO_FIRE = 'no_fire'


def retain(native: Any, ctx: Context, world: World, quiet: RecordedPlan, row: Any,
           visible: list, index: int, stamp: float, label: str) -> RecordedPlan:
    candidate = deepcopy(quiet)
    candidate.policy = f'hand_{index+1:03d}'
    candidate.moves += 1
    candidate.opportunities.append(dict(t=stamp, score=int(row.exact_score), action=action_of(row)))
    add_fire(native, ctx, world, candidate, row, visible[0], stamp, visible, index, label)
    return candidate


def candidates(native: Any, ctx: Context, initial: World, side: int, start: float, deadline: float,
               pairs: list, budget: Budget, label: str) -> tuple[dict, bool]:
    world = initial.copy()
    quiet = RecordedPlan(side, NO_FIRE, start, deadline)
    found = {}
    hands = min(MAX_HANDS, max(0, int((deadline-start+1.e-9)/SEC_PER_HAND)))
    try:
        for index in range(hands):
            budget.check()
            stamp = start+(index+1)*SEC_PER_HAND
            world.advance(stamp)
            if world.invalid or any(world.dead):
                quiet.reason = world.invalid or 'projected_terminal_before_fire'
                break
            visible = pairs[index:index+VISIBLE_PAIRS]
            fire, move = choices(native, world.boards[side], visible, budget.width, ctx.colors)
            if fire is not None:
                branch = retain(native, ctx, world, quiet, fire, visible, index, stamp, label)
                found[branch.policy] = branch
            if move is None:
                quiet.reason = 'no_legal_quiet_move'
                break
            quiet.moves += 1
            add_quiet(world, quiet, move, stamp, visible, index, label)
        else:
            quiet.reason = 'no_fire_policy' if hands else 'no_placement_time'
        # 合法手がなく途中停止した非発火案を、締切まで生き残る案にしない。
        if quiet.reason in ('no_fire_policy', 'no_placement_time'):
            found[NO_FIRE] = quiet
    except SearchBudgetExpired:
        return found, False
    return found, True
