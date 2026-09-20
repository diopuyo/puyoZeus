"""原消費票の条件付きprefix。単独では観測資格も公開権限も持たない。"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import prefix_phase_math_v2 as P

ASSUMPTION = 'ちぎり補完後world内の未較正一様prior。手数間の混合priorは定義しない。'
VERSION = 'consumed-prefix/v1'


@dataclass(frozen=True)
class ConsumedArrival:
    scope: tuple
    token: str
    pair: tuple[int, int]
    frame: int
    call_token: str


@dataclass(frozen=True)
class Prepared:
    expected: Any
    following: Any
    before_prefix: int
    after_prefix: int
    arrivals: tuple[ConsumedArrival, ...]
    call_token: str


class Lane:
    def __init__(self, engine: Any, initial: Any) -> None:
        engine.B.validate(initial)
        self.engine, self.initial, self.current = engine, initial, initial
        self.arrivals: tuple[ConsumedArrival, ...] = ()
        self.families = (P.Family(0, P.SETTLED, initial),)
        self.committed = 0
        self.last_frame = initial.frame
        self.fired: frozenset[str] = frozenset()
        self.prior = engine.H.uncalibrated_uniform(ASSUMPTION)

    def append(self, scope: tuple, event: Any) -> None:
        check = self.engine.B.require
        check(scope == self.initial.scope, 'consumed_prefix_scope')
        frame = event.consumed_frame
        previous = self.arrivals[-1].frame if self.arrivals else self.initial.frame
        check(type(frame) is int and previous < frame <= self.initial.deadline
              and self.last_frame < frame, 'consumed_prefix_arrival_clock')
        check(type(event.occurrence_token) is str and bool(event.occurrence_token)
              and event.occurrence_token not in self.initial.tokens
              and all(a.token != event.occurrence_token for a in self.arrivals), 'consumed_prefix_token')
        check(type(event.source_call_token) is str and bool(event.source_call_token), 'consumed_prefix_call')
        check(type(event.pair) is tuple and len(event.pair) == 2
              and all(type(c) is int and c in self.engine.B.PIECE_COLORS for c in event.pair), 'consumed_prefix_pair')
        self.arrivals += (ConsumedArrival(scope, event.occurrence_token, event.pair,
                                         frame, event.source_call_token),)

    def condition(self, frame: int, observed: Any, fired: frozenset[str]) -> tuple:
        check = self.engine.B.require
        check(type(fired) is frozenset, 'consumed_prefix_fire_type')
        check(self.fired <= fired, 'consumed_prefix_fire_revoked')
        check(type(frame) is int and self.last_frame < frame <= self.initial.deadline,
              'consumed_prefix_observation_clock')
        following = tuple(child for family in self.families for child in P.condition(
            self.engine, family, self.arrivals, frame, observed, fired, self.prior))
        check(bool(following), 'consumed_prefix_zero_support')
        check(sum(len(f.value.worlds) for f in following) <= self.engine.B.MAX_WORLDS,
              'consumed_prefix_support_limit')
        check(all(f.prefix >= self.committed for f in following), 'consumed_prefix_regression')
        self.families, self.last_frame, self.fired = following, frame, fired
        return following

    def prepare(self, call: str) -> Prepared | None:
        check = self.engine.B.require
        check(type(call) is str and bool(call), 'consumed_prefix_prepare_call')
        if len(self.families) != 1 or self.families[0].phase != P.SETTLED:
            return None
        family = self.families[0]
        if family.prefix == self.committed:
            return None
        check(self.committed < family.prefix <= len(self.arrivals), 'consumed_prefix_prepare_range')
        following = family.value
        tokens = self.initial.tokens + tuple(a.token for a in self.arrivals[:family.prefix])
        check(following.tokens == tokens and following.tokens[:len(self.current.tokens)] == self.current.tokens,
              'consumed_prefix_ancestor')
        check(following.scope == self.current.scope and following.deadline == self.current.deadline
              and self.current.frame < following.frame == self.last_frame, 'consumed_prefix_prepare_clock')
        return Prepared(self.current, following, self.committed, family.prefix,
                        self.arrivals[self.committed:family.prefix], call)

    def accept(self, prepared: Prepared, following: Any) -> None:
        check = self.engine.B.require
        check(type(prepared) is Prepared and prepared.expected is self.current
              and prepared.before_prefix == self.committed, 'consumed_prefix_accept_once')
        check(following is prepared.following and len(self.families) == 1
              and following is self.families[0].value, 'consumed_prefix_accept_value')
        check(prepared.after_prefix == self.families[0].prefix
              and prepared.arrivals == self.arrivals[self.committed:prepared.after_prefix]
              and bool(prepared.call_token), 'consumed_prefix_accept_receipt')
        self.current, self.committed = following, prepared.after_prefix

    def snapshot(self, serializer: Any) -> dict:
        return dict(version=VERSION, initial=serializer.encode(self.initial), current=serializer.encode(self.current),
            arrivals=[asdict(a) for a in self.arrivals], committed=self.committed, last_frame=self.last_frame,
            fired_tokens=sorted(self.fired), families=[dict(prefix=f.prefix, phase=f.phase,
                state=serializer.encode(f.value), mixture_weight=None) for f in self.families],
            observation_qualification_connected=False, original_fifo_changed=False,
            physical_certified=False, quality_gate_clear=False)
