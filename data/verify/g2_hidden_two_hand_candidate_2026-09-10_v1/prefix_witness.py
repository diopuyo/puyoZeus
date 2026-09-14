"""同head/同原J保持票に束縛した可視二手のfresh二観測。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import path_support as P

MIN_VOTES, FPS = 2, 60


@dataclass
class Votes:
    binding: Any
    held: Any
    support: P.Support
    first: tuple[int, float]
    last: tuple[int, float]
    count: int


def observed(control: Any, binding: Any, signals: Any, view: Any) -> Any:
    # 実pipe参照はscopeのid値から再生成せず、Linkの同call原参照を使う。
    live = control.provider.link.current(view)
    raw, proof = control.provider.raw(live['pipe'],view.scope[-1],view)
    P.require(proof['captured_frame']==view.frame and proof['raw_grid']==raw, 'raw_clock')
    if raw != control._parts.T.board_key(signals.cnn_board): return None
    if any(c==P.UNKNOWN for row in raw[P.HIDDEN_ROWS:] for c in row): return None
    if not any(c==P.UNKNOWN for row in raw[:P.HIDDEN_ROWS] for c in row): return None
    if not control.provider.no_origin(live['pipe'],view.scope[-1],view): return None
    return raw,proof


def nonfiring(core: Any, support: P.Support) -> bool:
    return all(core.simulate_chain(P.board(core,g)).chain_count==0 for g in (support.prefix,support.final))


def vote(control: Any, binding: Any, held: Any, support: P.Support, view: Any) -> Votes:
    previous = getattr(binding,'hidden_prefix_votes',None)
    now = view.frame,view.clock
    same = (previous is not None and previous.binding is binding and previous.held is held
        and previous.support==support and control.provider.adjacent(previous.last,view))
    value = Votes(binding,held,support,previous.first if same else now,now,
                  previous.count+1 if same else 1)
    binding.hidden_prefix_votes = value
    return value


def clear(binding: Any) -> None:
    binding.hidden_prefix_votes = None
