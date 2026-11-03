"""即発火と締切までの伸長を独立に進める。各手は可視3組だけを参照する。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from src.phase_j._live_ex_frozen.src.indicators_v2 import SEC_PER_HAND
from src.phase_j._live_ex_frozen.src.prefire_search_v410 import Plan, Budget, DEATH_CELL, VISIBLE_PAIRS, MAX_HANDS
from src.phase_j._live_ex_frozen.src.prefire_context_v410 import Context, chain_events
from src.phase_j._live_ex_frozen.src.prefire_world_v410 import World, Event, LOCK, START

MODES = ('immediate', 'extend')
BOARD_COLUMNS = 6


class BoardMismatch(ValueError):
    """予測の物理整合を保証できない場合はGへ戻す。"""


class ColorMismatch(BoardMismatch):
    """予測盤が試合の4色と整合しない場合は探索前に棄権する。"""


def guard(raw: tuple | list, colors: tuple) -> None:
    if any(c not in (0, 9, *colors) for c in raw):
        raise ColorMismatch('projected_board_color_mismatch')
    # 空洞盤は既存の列挙と段別トレースで落下規則が異なる。
    for col in range(BOARD_COLUMNS):
        column = raw[col::BOARD_COLUMNS]
        if any(upper and not lower for upper, lower in zip(column, column[1:])):
            raise BoardMismatch('projected_board_gravity_mismatch')


def action_of(row: Any) -> tuple[int, int]:
    return ((row.col, 2) if row.rotation == 0 else (row.col, 0) if row.rotation == 2
            else (row.col+1, 3) if row.rotation == 3 else (row.col, 1))


@dataclass
class RecordedPlan(Plan):
    placements: list[dict] = field(default_factory=list)
    opportunities: list[dict] = field(default_factory=list)

    def audit(self) -> dict:
        return dict(super().audit(), placements=self.placements, opportunities=self.opportunities)


def choices(native: Any, raw: tuple, visible: list, width: int, colors: tuple,
            need_quiet: bool = True) -> tuple:
    """その手での最大発火と、可視NEXT評価が最大の非発火を返す。"""
    guard(raw, colors)
    if any(c not in colors for pair in visible[:VISIBLE_PAIRS] for c in pair):
        raise ColorMismatch('projected_tape_color_mismatch')
    rows = native.enumerate_and_simulate_placements_py(list(raw), *visible[0], False, True)
    legal = [r for r in rows if not r.final_grid[DEATH_CELL]]
    fires = [r for r in legal if r.exact_score > 0]
    fire = max(fires, key=lambda r: r.exact_score) if fires else None
    if fire is not None and not need_quiet:
        return fire, None
    quiet = {action_of(r): r for r in legal if not r.exact_score}
    if not quiet:
        return fire, None
    ranked = native.ama_search_py(list(raw), visible[:VISIBLE_PAIRS], width,
                                  min(len(visible), VISIBLE_PAIRS), list(colors))
    action = next((tuple(r[:2]) for r in ranked if tuple(r[:2]) in quiet), next(iter(quiet)))
    return fire, quiet[action]


def add_fire(native: Any, ctx: Context, world: World, plan: RecordedPlan, row: Any,
             pair: tuple, stamp: float, visible: list, index: int, label: str) -> None:
    action = action_of(row)
    raw = world.boards[plan.side]
    # 穴を含む認識盤でも、列挙で評価した設置盤をそのまま完走させる。
    placed = list(row.placed_grid)
    trace = native.simulate_chain_trace_py(placed, True)
    if (not trace or sum(int(step[4]) for step in trace) != int(row.exact_score)
            or list(trace[-1][1]) != list(row.final_grid)):
        raise BoardMismatch('projected_placement_trace_mismatch')
    plan.fire, plan.scores = stamp, [int(step[4]) for step in trace]
    final = tuple(trace[-1][1])
    plan.events.append(Event(stamp, START, plan.side, label, before=raw))
    events, _, plan.end = chain_events(plan.side, label, stamp, plan.scores, final,
        ctx.carry[plan.side], ctx.origin, stamp, credit=ctx.credit[plan.side])
    plan.events.extend(events)
    plan.placements.append(dict(t=stamp, index=index, visible=visible, action=action,
                                 before=list(raw), placed=placed, after=list(final), scores=plan.scores))


def add_quiet(world: World, plan: RecordedPlan, row: Any, stamp: float,
              visible: list, index: int, label: str) -> None:
    raw, final = world.boards[plan.side], tuple(row.final_grid)
    event = Event(stamp, LOCK, plan.side, label, board=final, before=raw)
    plan.events.append(event)
    plan.placements.append(dict(t=stamp, index=index, visible=visible, action=action_of(row),
                                 before=list(raw), placed=list(row.placed_grid), after=list(final), scores=[]))
    world.add([event])
    world.advance(stamp)


def plan(native: Any, ctx: Context, initial: World, side: int, start: float, deadline: float,
         pairs: list, mode: str, budget: Budget, label: str) -> RecordedPlan:
    world = initial.copy()
    result = RecordedPlan(side, mode, start, deadline)
    hands = min(MAX_HANDS, max(0, int((deadline-start+1e-9)/SEC_PER_HAND)))
    for index in range(hands):
        budget.check()
        stamp = start+(index+1)*SEC_PER_HAND
        world.advance(stamp)
        if world.invalid or any(world.dead):
            result.reason = world.invalid or 'projected_terminal_before_fire'
            break
        visible = pairs[index:index+VISIBLE_PAIRS]
        fire, quiet = choices(native, world.boards[side], visible, budget.width, ctx.colors,
                              mode == 'extend' and index < hands-1)
        if fire is not None:
            result.opportunities.append(dict(t=stamp, score=int(fire.exact_score), action=action_of(fire)))
        take_fire = fire is not None and (mode == 'immediate' or index == hands-1 or quiet is None)
        if take_fire:
            result.moves += 1
            add_fire(native, ctx, world, result, fire, visible[0], stamp, visible, index, label)
            break
        if quiet is None:
            result.reason = 'no_legal_move'
            break
        result.moves += 1
        add_quiet(world, result, quiet, stamp, visible, index, label)
    if result.fire is None and result.reason is None:
        result.reason = 'no_fire_in_time' if hands else 'no_placement_time'
    return result
