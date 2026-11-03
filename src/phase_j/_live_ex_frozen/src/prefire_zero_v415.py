"""置き終わりの証跡が対応する確定盤だけ、設置なしで発火させる。"""
from __future__ import annotations

from typing import Any
from src.phase_j._live_ex_frozen.src.prefire_context_v410 import Context,chain_events
from src.phase_j._live_ex_frozen.src.prefire_search_v411 import RecordedPlan,guard
from src.phase_j._live_ex_frozen.src.prefire_world_v410 import Event,START
from src.phase_j._live_ex_frozen.src.indicators_v2 import SEC_PER_HAND

QUIET=frozenset(('STABLE','TSUMO_FALL'))
BUSY=frozenset(('CHAIN','GRAVITY_SETTLE'))
ZERO='hand_000_connected'


class PlacementEvidence:
    """合図と同じ時点の確定盤を結び、盤面変更後へ許可を持ち越さない。"""
    def __init__(self) -> None:
        self.game: int | None=None
        self.authorized: list[dict | None]=[None,None]

    def observe(self, state: Any, raw: list, stamp: float) -> None:
        if self.game!=state.game_idx:
            self.game=state.game_idx; self.authorized=[None,None]
        for i,s in enumerate(state.sides):
            board=tuple(map(int,s.board.ravel())) if s.board is not None else None
            event=(raw[i].get('chain_event') or {}).get('namespace',{})
            signals=[name for name,flag in (('next_slide',s.slide),('ojama_fall',s.state=='OJAMA_FALL'),
                ('formula',bool(event) and event.get('mechanism')!='baseline')) if flag]
            old=self.authorized[i]
            if old and old['board']!=board:
                self.authorized[i]=None
            if signals and board is not None:
                self.authorized[i]=dict(board=board,t=stamp,signals=signals)

    def snapshot(self, ctx: Context, state: Any) -> dict:
        side=ctx.receiver; record=self.authorized[side]
        allowed=bool(record and record['board']==ctx.world.boards[side] and record['t']<=ctx.input_sec
                     and state.sides[side].state in QUIET and ctx.triggers[side] is None)
        return dict(allowed=allowed,evidence_sec=record['t'] if allowed else ctx.input_sec,
                    signals=record['signals'] if allowed else [],state=state.sides[side].state)


def connected(native: Any, ctx: Context, evidence: dict) -> RecordedPlan | None:
    """操作中の組を置かず、13段目を接着対象から除外した物理だけで解く。"""
    side=ctx.receiver
    if not evidence.get('allowed') or ctx.triggers[side] is not None or evidence.get('state') not in QUIET:
        return None
    if evidence['evidence_sec']>ctx.input_sec:
        raise ValueError('未来の置き終わり合図')
    raw=ctx.world.boards[side]; guard(raw,ctx.colors)
    trace=native.simulate_chain_trace_py(list(raw),True)
    if not trace:
        return None
    scores=[int(s[4]) for s in trace]
    p=RecordedPlan(side,ZERO,ctx.input_sec,ctx.ends[ctx.attacker]+SEC_PER_HAND)
    p.fire,p.scores,p.reason=ctx.input_sec,scores,'confirmed_connected_with_lock_signal'
    p.events.append(Event(p.fire,START,side,'Y',before=raw))
    events,_,p.end=chain_events(side,'Y',p.fire,scores,tuple(trace[-1][1]),
        ctx.carry[side],ctx.origin,ctx.input_sec,credit=ctx.credit[side])
    p.events.extend(events)
    return p
