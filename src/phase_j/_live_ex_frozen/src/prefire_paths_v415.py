"""2テープのYを同じ手番で進め、発見した発火を先に評価できる形で渡す。"""
from __future__ import annotations

from typing import Any,Iterator
from src.phase_j._live_ex_frozen.src.indicators_v2 import SEC_PER_HAND
from src.phase_j._live_ex_frozen.src.prefire_search_v410 import Budget,MAX_HANDS,VISIBLE_PAIRS
from src.phase_j._live_ex_frozen.src.prefire_search_v411 import RecordedPlan,choices,add_quiet
from src.phase_j._live_ex_frozen.src.prefire_search_v412 import retain,NO_FIRE


def steps(native: Any, ctx: Any, pairs: list, budget: Budget) -> Iterator[dict]:
    side=ctx.receiver; world=ctx.world.copy()
    quiet=RecordedPlan(side,NO_FIRE,ctx.input_sec,ctx.ends[ctx.attacker]+SEC_PER_HAND)
    hands=min(MAX_HANDS,max(0,int((quiet.deadline-quiet.start+1.e-9)/SEC_PER_HAND)))
    found={}
    for index in range(hands):
        budget.check(); stamp=quiet.start+(index+1)*SEC_PER_HAND; world.advance(stamp)
        if world.invalid or any(world.dead):
            return
        visible=pairs[index:index+VISIBLE_PAIRS]
        fire,move=choices(native,world.boards[side],visible,budget.width,ctx.colors)
        if fire is not None:
            p=retain(native,ctx,world,quiet,fire,visible,index,stamp,'Y'); found[p.policy]=p
        if move is not None:
            quiet.moves+=1; add_quiet(world,quiet,move,stamp,visible,index,'Y')
        yield dict(found)
        if move is None:
            return
    quiet.reason='no_fire_policy' if hands else 'no_placement_time'
    found[NO_FIRE]=quiet
    yield found
