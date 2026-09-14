"""独立検収の比較callee差替と旧公開証拠の境界を原最終処理へ追加する。"""
from __future__ import annotations
from dataclasses import asdict
from types import FunctionType, SimpleNamespace as N
import sys
from typing import Any
import live_finish as OLD
import private_code as CODE
import live_boundary_v2 as BOUNDARY


def private_samecall(factory: Any, lease: Any, journal: Any, rows: Any, evidence: Any) -> Any:
    module = sys.modules[type(factory.controller.private_suffix_completion_rows[0]).__module__]
    CODE.verify(module)
    return OLD.private_samecall(factory,lease,journal,rows,evidence)


def prepared(factory: Any, state: Any, lease: Any, evidence: Any) -> Any:
    def check(*args: Any) -> Any:
        return BOUNDARY.check(*args,old_owner_scope=asdict(lease.archive.binding.owner.state.scope))
    function = FunctionType(OLD.prepared.__code__,dict(vars(OLD),private_samecall=private_samecall,
                                                       PUBLICATION=N(check=check)))
    return function(factory,state,lease,evidence)


stage,read = OLD.stage,OLD.read
