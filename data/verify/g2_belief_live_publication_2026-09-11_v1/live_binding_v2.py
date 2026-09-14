"""旧caller票を受け取らず、原発行票と確率反映証拠を束縛する修復入口。"""
from __future__ import annotations

from typing import Any
import live_binding as OLD
import journal_witness as W
import reflection as R


def current(rec: Any, factory: Any, pipe: Any, journal: Any, registry: Any,
            bindings: tuple[Any, Any], contract: Any, witness: W.Witness,
            modes: tuple[Any, Any]) -> OLD.Bound:
    OLD.C.require(type(witness) is W.Witness and witness.journal is journal, 'witness_owner')
    OLD.C.require(len(modes) == len(bindings) == 2, 'reflection_both_sides')
    OLD.C.require(rec.rows, 'context_rows_missing')
    steps = witness.pair(rec.rows[-1]['frame_idx'])
    bound = OLD.current(rec, factory, pipe, journal, registry, bindings, contract, steps)
    for mode, binding, value, token in zip(modes, bindings, bound.values, bound.tokens, strict=True):
        OLD.C.require(mode.connection.registry is registry and mode.connection.binding is binding, 'reflection_binding')
        R.verify(mode, value, token, bound.frame)
    return bound
