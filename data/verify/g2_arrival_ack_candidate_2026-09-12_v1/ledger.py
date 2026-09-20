"""到来・暫定盤面反映・原FIFO消費を分離する私有台帳。原FIFOには触れない。"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

Scope = tuple[Any, ...]
COLORS = frozenset((1, 2, 3, 4, 5))


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('arrival_ack:' + reason)


@dataclass(frozen=True)
class Arrival:
    scope: Scope
    token: str
    pair: tuple[int, int]
    frame: int
    call_token: str


@dataclass(frozen=True)
class Ack:
    token: str
    frame: int
    call_token: str


@dataclass(frozen=True)
class Ledger:
    scope: Scope
    start: int
    deadline: int
    clock: int
    arrivals: tuple[Arrival, ...] = ()
    applied: tuple[str, ...] = ()
    acknowledgements: tuple[Ack, ...] = ()
    source_connected: bool = field(default=False, init=False)
    physical_certified: bool = field(default=False, init=False)
    quality_gate_clear: bool = field(default=False, init=False)


def check(value: Ledger) -> None:
    require(type(value) is Ledger and type(value.scope) is tuple and len(value.scope) == 7, 'scope')
    require(all(type(s) is str and bool(s) for s in value.scope[:2])
            and all(type(n) is int and n >= 0 for n in value.scope[2:6])
            and value.scope[-1] in ('1P', '2P'), 'scope_fields')
    require(all(type(x) is int for x in (value.start, value.deadline, value.clock))
            and 0 <= value.start <= value.clock <= value.deadline, 'clock')
    require(type(value.arrivals) is tuple and type(value.applied) is tuple
            and type(value.acknowledgements) is tuple, 'immutable_collections')
    tokens = tuple(a.token for a in value.arrivals)
    require(len(tokens) == len(set(tokens)), 'duplicate_arrival')
    previous = value.start
    for arrival in value.arrivals:
        require(type(arrival) is Arrival and arrival.scope == value.scope, 'arrival_scope')
        require(type(arrival.frame) is int and previous < arrival.frame <= value.clock, 'arrival_clock')
        require(type(arrival.token) is str and bool(arrival.token)
                and type(arrival.call_token) is str and bool(arrival.call_token), 'arrival_token')
        require(type(arrival.pair) is tuple and len(arrival.pair) == 2
                and all(type(c) is int and c in COLORS for c in arrival.pair), 'pair')
        previous = arrival.frame
    require(value.applied == tokens[:len(value.applied)], 'applied_prefix')
    require(tuple(a.token for a in value.acknowledgements) == tokens[:len(value.acknowledgements)], 'ack_prefix')
    require(len({a.call_token for a in value.acknowledgements}) == len(value.acknowledgements), 'duplicate_ack_call')
    previous = value.start
    for arrival, ack in zip(value.arrivals, value.acknowledgements):
        require(type(ack) is Ack and type(ack.frame) is int
                and max(previous, arrival.frame) <= ack.frame <= value.clock, 'ack_clock')
        require(type(ack.call_token) is str and bool(ack.call_token), 'ack_call')
        previous = ack.frame


def advance_clock(value: Ledger, frame: int) -> Ledger:
    check(value)
    require(type(frame) is int and value.clock <= frame <= value.deadline, 'event_clock')
    return replace(value, clock=frame)


def arrive(value: Ledger, arrival: Arrival) -> Ledger:
    result = replace(advance_clock(value, arrival.frame), arrivals=value.arrivals + (arrival,))
    check(result)
    return result


def applied(value: Ledger, tokens: tuple[str, ...], frame: int) -> Ledger:
    require(type(tokens) is tuple and bool(tokens), 'empty_application')
    result = replace(advance_clock(value, frame), applied=value.applied + tokens)
    check(result)
    return result


def acknowledge(value: Ledger, token: str, pair: tuple[int, int], frame: int, call: str) -> Ledger:
    result = advance_clock(value, frame)
    index = len(value.acknowledgements)
    require(index < len(value.arrivals), 'unknown_ack')
    arrival = value.arrivals[index]
    require(token == arrival.token and type(pair) is tuple and pair == arrival.pair
            and all(type(c) is int for c in pair), 'ack_identity')
    result = replace(result, acknowledgements=value.acknowledgements + (Ack(token, frame, call),))
    check(result)
    return result


def drained(value: Ledger) -> bool:
    check(value)
    return len(value.arrivals) == len(value.applied) == len(value.acknowledgements)
