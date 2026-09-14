"""原J/Registry照合のv2本体を保持し、初回2Pの厳格な反映検査へ差し替える。"""
from __future__ import annotations

from types import FunctionType
from typing import Any
import live_binding_v2 as V2
import reflection_v3 as R

_CURRENT=FunctionType(V2.current.__code__,dict(vars(V2),R=R))


def current(rec: Any,factory: Any,pipe: Any,journal: Any,registry: Any,
            bindings: tuple[Any,Any],contract: Any,witness: Any,modes: tuple[Any,Any]) -> Any:
    return _CURRENT(rec,factory,pipe,journal,registry,bindings,contract,witness,modes)

