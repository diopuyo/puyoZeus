"""旧最終構成を保持し、reset帰還直後だけ新隔離hookを接続する私有候補。"""
from __future__ import annotations
from pathlib import Path
import sys
from types import FunctionType,SimpleNamespace as N
from typing import Any
from inflight_loader import CONNECTION as C

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'g2_empty_tail_reset_live_adapter_2026-09-11_v1'
sys.path.insert(0,str(PREVIOUS))
import adapter_v4 as V4
import adapter_v5 as V5


def dependencies(stack: Any,old: Any) -> Any:
    modules=V4.dependencies(stack,old)
    original=modules.proof.Context
    class Context(original):
        def perform(self,advisory: Any,frame: int,clock: float) -> None:
            super().perform(advisory,frame,clock)
            try:
                C.install(self.stack,self.recovery,self.state)
            except BaseException as error:
                self.error=repr(error)
                raise
    modules.proof=N(**(vars(modules.proof)|dict(Context=Context)))
    return modules


def configured(stack: Any) -> Any:
    invoke=FunctionType(V4.configured.__code__,dict(vars(V4),dependencies=dependencies,
        finalize=V5.namespace['finalize']))
    return invoke(stack)
