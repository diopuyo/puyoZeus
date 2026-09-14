"""私有laneの一括反映準備。資格・保存・所有接続前には呼出認可を与えない。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import prefix_phase_math_v2 as P

VERSION = 'identified_settled_prefix/v1'


@dataclass(frozen=True)
class Prepared:
    expected: Any
    following: Any
    ledger_before: Any
    ledger_after: Any
    receipt: dict


def select(parts: Any, current: Any, ledger: Any, base: tuple, families: tuple,
           frame: int) -> tuple[Any, tuple]:
    """曖昧な履歴の確率混合をしない。唯一settledだけを候補にする。"""
    check, engine = parts.mode.B.require, parts.mode.C.T
    parts.mode.L.check(ledger)
    engine.B.validate(current)
    check(type(families) is tuple and len(families) == 1, 'prefix_commit_ambiguous')
    family = families[0]
    check(type(family) is P.Family and family.phase == P.SETTLED, 'prefix_commit_not_settled')
    following = family.value
    engine.B.validate(following)
    check(type(base) is tuple and base == ledger.applied[:len(base)]
        and len(base) <= len(ledger.applied), 'prefix_commit_base')
    check(current.scope == following.scope == ledger.scope
        and current.deadline == following.deadline == ledger.deadline, 'prefix_commit_scope')
    check(type(frame) is int and current.frame < frame == following.frame
        and ledger.clock <= frame <= ledger.deadline, 'prefix_commit_clock')
    check(type(family.prefix) is int and family.prefix >= 0, 'prefix_commit_prefix_type')
    total = len(base) + family.prefix
    check(len(ledger.applied) < total <= len(ledger.arrivals), 'prefix_commit_prefix')
    tokens = tuple(a.token for a in ledger.arrivals[:total])
    check(len(following.tokens) >= total and following.tokens[-total:] == tokens
        and following.tokens[:len(current.tokens)] == current.tokens, 'prefix_commit_tokens')
    count = len(ledger.applied)
    check(not count or current.tokens[-count:] == ledger.applied, 'prefix_commit_current_tokens')
    simulator = engine.B.ChainSimulator(exclude_hidden_row_from_pop=True)
    check(all(not simulator.find_erasable_groups(engine.B.Board.from_dict({'grid': world.grid}))
        for world in following.worlds), 'prefix_commit_unsettled_physics')
    return following, tokens[count:]


def prepare(parts: Any, current: Any, ledger: Any, base: tuple, families: tuple,
            frame: int, call: str, evidence_key: str) -> Prepared:
    """原台帳を先に検査し、まだRegistry/FIFO/Modeには書き込まない。"""
    check = parts.mode.B.require
    check(type(call) is str and bool(call) and type(evidence_key) is str and bool(evidence_key),
        'prefix_commit_source_key')
    following, tokens = select(parts, current, ledger, base, families, frame)
    after = parts.mode.L.applied(ledger, tokens, frame)
    receipt = dict(kind=VERSION, source_call_token=call, applied_frame=frame,
        base_applied_tokens=base, applied_tokens=tokens, evidence_key=evidence_key,
        state=parts.mode.BASE.V1.S.encode(following), ledger_after=asdict(after),
        physical_certified=False, original_fifo_changed=False,
        source_qualification_connected=False, runtime_connected=False)
    return Prepared(current, following, ledger, after, receipt)
