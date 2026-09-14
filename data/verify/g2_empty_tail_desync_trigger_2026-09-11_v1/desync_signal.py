"""連続する原NEXT不整合の観測信号。reset/FIFO/currentの認可とは分離する。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

STRIDE = 2
COLORS = frozenset(range(1,6))


@dataclass(frozen=True)
class Fact:
    frame: int
    scope: tuple[Any, ...]
    baseline_frame: int
    baseline_pair: tuple[int, int]
    pair: tuple[int, int]
    dnext: tuple[int, int]
    candidate: tuple[int, int, int, str]
    quiet_frames: tuple[int, int]
    native_quiet: bool
    reason: str = 'await_dnext_successor'
    native_appended: bool = False


def valid(fact: Fact) -> bool:
    if type(fact) is not Fact: return False
    pairs = (fact.baseline_pair,fact.pair,fact.dnext)
    return (type(fact) is Fact and type(fact.frame) is int and type(fact.baseline_frame) is int
        and type(fact.scope) is tuple and len(fact.scope)==6 and fact.scope[3] in ('1P','2P')
        and type(fact.scope[2]) is int and type(fact.scope[4]) is int
        and all(type(fact.scope[i]) is str and bool(fact.scope[i]) for i in (0,1,5))
        and all(type(p) is tuple and len(p)==2 and all(type(c) is int and c in COLORS for c in p) for p in pairs)
        and fact.native_quiet is True and fact.native_appended is False
        and fact.reason=='await_dnext_successor' and fact.pair!=fact.baseline_pair
        and type(fact.quiet_frames) is tuple and all(type(f) is int for f in fact.quiet_frames)
        and fact.quiet_frames==(fact.frame-STRIDE,fact.frame)
        and type(fact.candidate) is tuple and len(fact.candidate)==4 and all(type(fact.candidate[i]) is int for i in (0,1,2))
        and fact.candidate[0]>0 and fact.baseline_frame<=fact.candidate[1]<=fact.candidate[2]<=fact.frame
        and type(fact.candidate[3]) is str and bool(fact.candidate[3]))


def joined(previous: Fact | None, current: Fact | None) -> bool:
    if previous is None or current is None or not valid(previous) or not valid(current): return False
    return (current.frame==previous.frame+STRIDE and current.scope==previous.scope
        and current.candidate==previous.candidate and current.baseline_frame==previous.baseline_frame
        and current.baseline_pair==previous.baseline_pair and current.pair==previous.pair
        and current.dnext==previous.dnext)


class Matcher:
    def __init__(self) -> None:
        self.previous: Fact | None = None
        self.issued: set[Any] = set()

    def push(self, fact: Fact | None) -> tuple[Fact, Fact] | None:
        previous, self.previous = self.previous, fact
        if not joined(previous,fact): return None
        key = (fact.scope,fact.candidate)
        if key in self.issued: return None
        self.issued.add(key)
        return previous,fact
