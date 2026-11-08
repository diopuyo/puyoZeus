"""現在までの証跡だけから、既発火X/Yの仮説時計を組み立てる。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from src.phase_j._live_ex_frozen.src.exchange_count_ledger import gross, FORMULA_SCORE_DELAY_SEC, STEP_SEC
from src.phase_j._live_ex_frozen.src.indicators_v2 import estimate_chain_anim_duration_sec, SEC_PER_HAND
from src.phase_j._live_ex_frozen.src.prefire_physics_v49c import chain_records
from src.phase_j._live_ex_frozen.src.b3_all_clear_credit import BONUS_OJAMA
from src.phase_j._live_ex_frozen.src.prefire_ledger_v410 import Batch, Ledger
from src.phase_j._live_ex_frozen.src.prefire_world_v410 import World, Event, GENERATE, CONFIRM, FINISH

SIDES = (0, 1)
PAIR_SIZE = 2
CALIBRATION = 'empirical_table_2026_08_14'


@dataclass
class Context:
    world: World
    input_sec: float
    evidence_max: float
    origin: float
    attacker: int
    receiver: int
    ends: tuple[float, float]
    triggers: tuple[float | None, float | None]
    queues: tuple[tuple[int, ...], tuple[int, ...]]
    colors: tuple[int, ...]
    carry: tuple[int, int]
    credit: tuple[int, int]


def duration(count: int) -> float:
    if not count:
        return 0.
    return max(estimate_chain_anim_duration_sec(count, CALIBRATION),
               FORMULA_SCORE_DELAY_SEC+(count-1)*STEP_SEC)


def chain_events(side: int, source: str, trigger: float, scores: list[int], final: tuple,
                 carry: int, origin: float, now: float, reflected: int = 0,
                 credit: int = 0, times: list[float] | None = None) -> tuple[list[Event], int, float]:
    """既反映分を引き、未反映の各段だけ生成する。予測時刻は証跡時刻に数えない。"""
    stamps = times or [trigger+FORMULA_SCORE_DELAY_SEC+i*STEP_SEC for i in range(len(scores))]
    total, events, rest = credit, [], reflected
    if credit:
        take, rest = max(0, credit-rest), max(0, rest-credit)
        if take:
            # 得点の列がまだ空 (scores も times も空) の発火でも、全消しの分は 1 段目と同じ時刻に生成する (11/6 IndexError)
            first = stamps[0] if stamps else trigger+FORMULA_SCORE_DELAY_SEC
            events.append(Event(max(now, first), GENERATE, side, source, take))
    for points, stamp in zip(scores, stamps):
        amount, carry = gross([points], [stamp], carry, origin)
        total += amount
        take, rest = max(0, amount-rest), max(0, rest-amount)
        if take:
            events.append(Event(max(now, stamp), GENERATE, side, source, take))
    end = max(now, trigger+duration(len(scores)), max(stamps, default=now))
    events += [Event(end, CONFIRM, side, source),
               Event(end, FINISH, side, source, board=final, trigger=trigger)]
    return events, carry, end


def observed_batches(engine: Any) -> list[Batch]:
    result = []
    for side, batches in enumerate(engine.state.sl.core.batches):
        for batch in batches:
            result.append(Batch(side, int(batch.n), f'C{batch.src_side}', float(batch.gen_t),
                                batch.confirmed_t if batch.confirmed else None))
    return result


def committed(engine: Any, value: dict, world: World) -> tuple:
    core, now = engine.state.sl.core, engine.state.t
    ends, triggers, carry, credit = [now, now], [None, None], [], []
    for side in SIDES:
        carry.append(int(engine.state.sl_snap[f'carry_p{side+1}']))
        credit.append(BONUS_OJAMA*len(core.all_clear[side].pending))
        chain, rec = engine.chains[side], value['reconciliation'][side]
        if chain is not None:
            triggers[side] = chain['started']
        if not value['board_prediction_active'][side] or chain is None or rec is None:
            continue
        trigger, scores = chain['started'], chain['best']['scores']
        events, carry[side], ends[side] = chain_events(side, f'C{side}', trigger, scores,
            tuple(value['boards'][side]), int(chain['carry_at_fire']), core.origin_sec, now,
            int(rec['reflected']), int(core.all_clear[side].forecast(trigger)), rec['times'])
        triggers[side] = trigger
        credit[side] = BONUS_OJAMA if not any(value['boards'][side]) else 0
        world.add(events)
    return tuple(ends), tuple(triggers), tuple(carry), tuple(credit)


def initiative(engine: Any, active: list[int]) -> int:
    """終わった古い発火をYと誤認しない。重なりか直後1手の返しだけ同じ往復とする。"""
    lead = min(active, key=lambda i: (engine.chains[i]['started'], i))
    other = 1-lead
    previous = engine.chains[other]
    if other in active or previous is None or previous['started'] >= engine.chains[lead]['started']:
        return lead
    ended = [r.t_end for r in chain_records(engine.state.sl.core, other, previous['started'])
             if r.t_end is not None and r.t_end <= engine.state.t]
    end = max(ended, default=previous.get('physical_end_sec'))
    return other if end is not None and end+SEC_PER_HAND >= engine.chains[lead]['started'] else lead


def context(engine: Any, value: dict) -> tuple[Context | None, str | None]:
    now, core = engine.state.t, engine.state.sl.core
    colors = tuple(engine.state.colors.known()[1])
    if core.origin_sec is None or len(colors) != 4:
        return None, 'origin_or_colors_missing'
    evidence = max(value['evidence_times'])
    if evidence > now:
        raise ValueError('未来の証跡')
    if any(c not in (0, 9, *colors) for b in value['current_boards'] for c in b):
        return None, 'unknown_board'
    active = [i for i in SIDES if value['board_prediction_active'][i] and engine.chains[i]]
    if not active:
        return None, 'no_committed_attack'
    if any(engine.state.sides[i].state in ('CHAIN', 'GRAVITY_SETTLE') and i not in active for i in SIDES):
        return None, 'unresolved_chain_board'
    if any(engine.state.sides[i].state == 'OJAMA_FALL' for i in SIDES):
        return None, 'garbage_landing_in_progress'
    attacker = initiative(engine, active)
    batches = observed_batches(engine)
    if any(b.confirmed_at is None and int(b.source[1:]) not in active for b in batches):
        return None, 'unconfirmed_source_without_chain'
    world = World(now, [tuple(b) for b in value['current_boards']], Ledger.from_batches(batches))
    ends, triggers, carries, credits = committed(engine, value, world)
    receiver = 1-attacker
    if triggers[receiver] is not None and triggers[receiver] < triggers[attacker]:
        triggers = tuple(None if i == receiver else triggers[i] for i in SIDES)
    queues = tuple(tuple(engine.state.sides[i].queue) if i in active else
                   (0, 0)+tuple(engine.state.sides[i].queue) for i in SIDES)
    return Context(world, now, evidence, core.origin_sec, attacker, 1-attacker,
                   ends, triggers, queues, colors, carries, credits), None
