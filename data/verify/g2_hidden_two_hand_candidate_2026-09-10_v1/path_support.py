"""原配置列挙による二手の支持集合。確率・観測・公開権を創作しない。"""
from __future__ import annotations
from dataclasses import dataclass, field
import itertools
from typing import Any

ROWS, COLS, HIDDEN_ROWS, UNKNOWN, PAIR_SIZE = 13, 6, 1, 10, 2
COLORS = frozenset(range(1, 6))
VALUES = COLORS | {0, 9}
Grid = tuple[tuple[int, ...], ...]


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('hidden_path_' + reason)


def grid(value: Any, *, hidden_unknown: bool = False) -> Grid:
    require(isinstance(value, (list,tuple)) and len(value) == ROWS, 'rows')
    require(all(isinstance(row,(list,tuple)) and len(row)==COLS for row in value), 'cols')
    require(all(type(c) is int and (c in VALUES or (hidden_unknown and r < HIDDEN_ROWS and c == UNKNOWN))
                for r,row in enumerate(value) for c in row), 'colors')
    return tuple(tuple(row) for row in value)


def pair(value: Any) -> tuple[int, int]:
    require(isinstance(value,(list,tuple)) and len(value)==PAIR_SIZE
            and all(type(c) is int and c in COLORS for c in value), 'pair')
    return tuple(value)


def board(core: Any, value: Grid) -> Any:
    result = core.Board()
    for r,row in enumerate(value):
        for c,color in enumerate(row):
            result.set(r,c,color)
    return result


def options(core: Any, before: Grid, colors: tuple[int,int]) -> tuple[Grid,...]:
    source, found = board(core,before), set()
    for ordered in sorted(set(itertools.permutations(colors))):
        for _,_,placed in core.enumerate_placements(source,ordered,filter_dead=False):
            found.add(grid(placed.to_dict()['grid']))
    require(grid(source.to_dict()['grid'])==before, 'enumerator_mutated_input')
    return tuple(sorted(found))


def compatible(raw: Grid, inferred: Grid) -> bool:
    return all(c == UNKNOWN or c == inferred[r][col] for r,row in enumerate(raw) for col,c in enumerate(row))


@dataclass(frozen=True)
class Support:
    prefix: Grid
    final: Grid
    raw: Grid
    pairs: tuple[tuple[int,int],tuple[int,int]]
    physical_certified: bool = field(default=False, init=False)
    probability_assigned: bool = field(default=False, init=False)
    accounting_permission: bool = field(default=False, init=False)
    current_permission: bool = field(default=False, init=False)


def infer(core: Any, before: Any, first: Any, second: Any, raw: Any) -> tuple[Support,...]:
    initial, observed = grid(before), grid(raw,hidden_unknown=True)
    pairs = pair(first),pair(second)
    result = []
    for prefix in options(core,initial,pairs[0]):
        for final in options(core,prefix,pairs[1]):
            if compatible(observed,final):
                result.append(Support(prefix,final,observed,pairs))
    return tuple(result)
