"""可視NEXTを先に読む方式F。未知テープの先を手の選択へ渡さない。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from time import perf_counter
import numpy as np
from src.phase_j._live_ex_frozen.src.board import BOARD_ROWS, BOARD_COLS, DEATH_ROW, DEATH_COL
from src.phase_j._live_ex_frozen.src.indicators_v2 import SEC_PER_HAND
from src.phase_j._live_ex_frozen.src.scoring import compute_effective_rate
from src.phase_j._live_ex_frozen.src.prefire_context_v410 import Context, chain_events
from src.phase_j._live_ex_frozen.src.prefire_world_v410 import World, Event, LOCK, START

VISIBLE_PAIRS = 3
MAX_HANDS = 128
SEED = 41020261021
SIDE_SEED_STRIDE = 100003
DEATH_CELL = DEATH_ROW*BOARD_COLS+DEATH_COL
BUDGETS = {100: (2, 1), 250: (4, 2), 500: (8, 3)}


class SearchBudgetExpired(RuntimeError):
    """計算途中のテープを勝率へ混ぜないための中断。"""


@dataclass(frozen=True)
class Budget:
    """秒予算ごとの固定作業配分。実測超過を隠さず別途採点する。"""
    milliseconds: int = 100
    deadline: float = float('inf')

    def __post_init__(self) -> None:
        if self.milliseconds not in BUDGETS:
            raise ValueError('予算は100/250/500ms')

    @property
    def samples(self) -> int:
        return BUDGETS[self.milliseconds][0]

    @property
    def width(self) -> int:
        return BUDGETS[self.milliseconds][1]

    def check(self) -> None:
        if perf_counter() >= self.deadline:
            raise SearchBudgetExpired('探索の時間上限')


@dataclass
class Plan:
    side: int
    policy: str
    start: float
    deadline: float
    events: list[Event] = field(default_factory=list)
    fire: float | None = None
    end: float | None = None
    scores: list[int] = field(default_factory=list)
    moves: int = 0
    reason: str | None = None
    visible: list[list[tuple[int, int]]] = field(default_factory=list)

    def audit(self) -> dict:
        return dict(side=self.side, policy=self.policy, start_sec=self.start, deadline_sec=self.deadline,
                    fire_sec=self.fire, send_sec=[e.t for e in self.events if e.kind == 0],
                    end_sec=self.end, scores=self.scores, moves=self.moves, reason=self.reason)


def tape(queue: tuple[int, ...], colors: tuple[int, ...], sample: int, side: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(SEED+sample+SIDE_SEED_STRIDE*side)
    values = rng.choice(colors, size=(MAX_HANDS+VISIBLE_PAIRS, 2)).astype(int)
    for i, color in enumerate(queue):
        if color in colors:
            values[i//2, i % 2] = color
    return [tuple(map(int, p)) for p in values]


def place(raw: tuple[int, ...], pair: tuple[int, int], x: int, rotation: int) -> tuple[int, ...] | None:
    """amaの軸ぷよ・回転番号で設置する。通常のtop/bottom表現とは区別する。"""
    a, b, y = (pair[1], pair[0], x) if rotation == 2 else (
        pair[0], pair[1], x+1 if rotation == 1 else x-1 if rotation == 3 else x)
    grid = list(raw)
    for col, color in ((x, a), (y, b)):
        if not 0 <= col < BOARD_COLS:
            return None
        empty = [r for r in range(BOARD_ROWS) if grid[r*BOARD_COLS+col] == 0]
        if not empty:
            return None
        grid[max(empty)*BOARD_COLS+col] = color
    return tuple(grid)


def choose(native: Any, raw: tuple, visible: list, remaining: int, colors: tuple,
           width: int, target: int) -> tuple[int, int] | None:
    if remaining == 1:
        found = native.enumerate_and_simulate_placements_py(list(raw), *visible[0], False, True)
        legal = [r for r in found if not r.final_grid[DEATH_CELL]]
        if not legal:
            return None
        best = max(legal, key=lambda r: r.exact_score)
        return ((best.col, 2) if best.rotation == 0 else (best.col, 0) if best.rotation == 2
                else (best.col+1, 3) if best.rotation == 3 else (best.col, 1))
    visible = visible[:min(remaining, VISIBLE_PAIRS)]
    rows = native.ama_search_py(list(raw), visible, width, len(visible),
                                list(colors), cap=target)
    return tuple(rows[0][:2]) if rows else None


def one_move(native: Any, world: World, side: int, visible: list, remaining: int,
             ctx: Context, budget: Budget, target: int) -> tuple:
    raw = world.boards[side]
    action = choose(native, raw, visible, remaining, ctx.colors, budget.width, target)
    placed = None if action is None else place(raw, visible[0], *action)
    if placed is None:
        return None, (), []
    trace = native.simulate_chain_trace_py(list(placed), True)
    final = tuple(trace[-1][1]) if trace else placed
    return final, action, [int(step[4]) for step in trace]


def plan(native: Any, ctx: Context, initial: World, side: int, start: float, deadline: float,
         pairs: list, policy: str, budget: Budget, label: str) -> Plan:
    world, result = initial.copy(), Plan(side, policy, start, deadline)
    hands = min(MAX_HANDS, max(0, int((deadline-start+1e-9)/SEC_PER_HAND)))
    for index in range(hands):
        budget.check()
        stamp = start+(index+1)*SEC_PER_HAND
        world.advance(stamp)
        if world.invalid or any(world.dead):
            result.reason = world.invalid or 'projected_terminal_before_fire'
            break
        visible = pairs[index:index+VISIBLE_PAIRS]
        target = (world.ledger.pending(side)*compute_effective_rate(stamp-ctx.origin, from_first_move=True)
                  if policy == 'counter' else 0)
        final, action, scores = one_move(native, world, side, visible, hands-index, ctx, budget, target)
        result.visible.append(visible)
        if final is None:
            result.reason = 'no_legal_move'
            break
        result.moves += 1
        if scores:
            result.fire, result.scores = stamp, scores
            result.events.append(Event(stamp, START, side, label, before=world.boards[side]))
            events, _, result.end = chain_events(side, label, stamp, scores, final,
                ctx.carry[side], ctx.origin, stamp, credit=ctx.credit[side])
            result.events.extend(events)
            break
        event = Event(stamp, LOCK, side, label, board=final, before=world.boards[side])
        result.events.append(event)
        world.add([event])
        world.advance(stamp)
    if result.fire is None and result.reason is None:
        result.reason = 'no_fire_in_time' if hands else 'no_placement_time'
    return result
