"""原NEXTの不整合信号で次入口を予約し、原更新完了後だけreset資格へ渡す。"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import sys
from types import MethodType
from typing import Any, Callable

VERIFY = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(VERIFY/'g2_empty_tail_desync_trigger_2026-09-11_v1'))
import entry_boundary_v3 as E


class Arming:
    def __init__(self, stack: Any, pipe: Any, state: Any, observer: Any,
                 callback: Callable[[Any,int,float],None]) -> None:
        self.stack,self.pipe,self.state,self.observer = stack,pipe,state,observer
        self.callback,self.guard = callback,state['repeat_scope_guard']
        self.armed,self.entry,self.error = None,None,None
        self.rows: list[Any] = []

    def observe(self) -> None:
        if self.error is not None: raise self.error
        advisory = self.observer.pending
        if advisory is None: return
        try:
            E.V.E.require(self.armed is None and self.entry is None,'automatic_second_reset_refused')
            self.armed = advisory
            target = advisory.facts[-1].frame+E.V.E.STRIDE
            self.entry = E.install(self.stack,self.pipe,self.state,self,None,target)
            self.rows.append(dict(stage='armed_during_original_update',frame=target,
                                  reset_executed=False,quality_gate_clear=False))
        except BaseException as exc:
            self.error = exc
            raise

    def perform(self, unused: Any, frame: int, clock: float) -> None:
        # Entry.beforeが前frame両側metadataを確認した後だけ、実完了した信号を消費。
        try:
            advisory = self.observer.take(self.pipe,frame-E.V.E.STRIDE)
            E.V.E.require(advisory is self.armed and advisory is not None,'automatic_signal_identity')
            self.callback(advisory,frame,clock)
            self.rows.append(dict(stage='qualified_callback_returned',frame=frame,quality_gate_clear=False))
        except BaseException as exc:
            self.error = exc
            self.rows.append(dict(stage='callback_failed',frame=frame,error=repr(exc)))
            raise


def install(stack: Any, pipe: Any, state: Any, observer: Any,
            callback: Callable[[Any,int,float],None]) -> Arming:
    value = Arming(stack,pipe,state,observer,callback)
    original = observer.observe
    E.V.E.require('observe' not in vars(observer),'automatic_observer_already_wrapped')
    def observe(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(*args,**kwargs)
        value.observe()
        return result
    stack.callback(vars(observer).pop,'observe',None)
    observer.observe = MethodType(observe,observer)
    return value
