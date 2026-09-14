"""1P同scope診断のみ。旧証拠を残して未達停止し、新scopeへ継続しない。"""
from __future__ import annotations
import sys
from types import MethodType
from typing import Any
import scope_stop_fixed as F
import scope_stop_evidence as E


class ScopeStop(RuntimeError):
    """計算/品質の完了を許可しない、一次理由を持つ停止。"""


class Guard:
    def __init__(self, factory: Any, pipe: Any, rows: list[Any]) -> None:
        self.factory, self.pipe, self.rows = factory, pipe, rows
        self.controller = factory.provider.journal.controller
        self.actual = F.bind(pipe,self.controller)
        self.binding: Any = None
        self.expected_scope: tuple[Any, ...] | None = None
        self.expected_owner_scope: Any = None
        self.before: dict[str, Any] | None = None
        self.error: ScopeStop | None = None
        self.record: dict[str, Any] | None = None
        self.frame: int | None = None
        self.clock: float | None = None

    def owned(self) -> Any:
        value = self.factory.controller.history.get(E.SIDE)
        if self.binding is not None and value is not self.binding:
            self.stop('binding_changed')
        if value is not None and self.binding is None:
            self.binding = value
            self.expected_scope = value.scope
            self.expected_owner_scope = value.owner.state.scope
        return value

    def caller(self, frame: Any, pipe: Any) -> None:
        if self.error is not None: raise self.error
        assert pipe is self.pipe and frame.f_code is self.actual.__code__, 'scope_stop_caller'
        assert frame.f_globals is self.actual.__globals__, 'scope_stop_globals'
        assert self.factory.provider.journal.controller is self.controller, 'scope_stop_instance'
        assert self.actual.__globals__['__next_live'] is self.controller, 'scope_stop_live_instance'

    def check(self, capture: bool = False) -> None:
        if self.error is not None: raise self.error
        binding = self.owned()
        if binding is None: return
        actual = E.scope(self.factory,self.pipe)
        if not E.same(binding.scope,self.expected_scope) or not E.same(self.expected_scope,actual):
            self.stop('scope_changed')
        scope = binding.owner.state.scope
        if (scope != self.expected_owner_scope or type(scope) is not type(self.expected_owner_scope)
                or scope.source_sha256 != actual[0].removeprefix('sha256:') or scope.run_id != actual[1]
                or scope.side != E.SIDE or type(scope.reset_epoch) is not int or scope.reset_epoch != actual[2]):
            self.stop('owner_scope_changed')
        if capture: self.before = E.snapshot(binding,self.factory,self.pipe)

    def stop(self, reason: str) -> None:
        if self.error is None:
            self.error = ScopeStop('same_scope_stop:'+reason)
            current = self.factory.controller.history.get(E.SIDE)
            self.record = dict(stage='same_scope_stop', reason=reason, frame=self.frame, time_sec=self.clock,
                clock_role='latest_update_not_reset_event_time', baseline_scope=self.expected_scope,
                old=self.before, actual_scope=E.scope(self.factory,self.pipe), native_now=E.native(self.pipe),
                owner_retained=current is self.binding, owner_now=None if current is None else E.asdict(current.owner.state),
                quality_met=False, computation_complete=False, new_scope_continuation=False, native_rollback=False)
            self.rows.append(self.record)
        raise self.error

    def reset(self, original: Any, *args: Any, **kwargs: Any) -> Any:
        self.check(capture=True)
        if self.binding is not None: self.stop('reset_requested')
        return original(*args,**kwargs)


def references(factory: Any, pipe: Any) -> tuple[Any, ...]:
    control = factory.provider.journal.controller
    return (pipe.reset, type(pipe).update, control.begin, control.enqueue, control.end, control.reset)


def install(stack: Any, factory: Any, pipe: Any, patch: Any, rows: list[Any]) -> Guard:
    assert not factory.controller.history, 'scope_stop_install_after_baseline'
    guard = Guard(factory,pipe,rows)
    control = guard.controller
    begin, enqueue, end, reset, pipe_reset = control.begin, control.enqueue, control.end, control.reset, pipe.reset
    def on_begin(value: Any, frame: int, clock: float) -> Any:
        guard.caller(sys._getframe().f_back,value)
        guard.frame, guard.clock = frame, clock
        guard.check(capture=True)
        return begin(value,frame,clock)
    def on_enqueue(value: Any, side: str, frame: int, clock: float, active: bool, pair: Any) -> Any:
        guard.caller(sys._getframe().f_back,value)
        guard.check()
        if guard.binding is not None and active is False: guard.stop('inactive_after_baseline')
        return enqueue(value,side,frame,clock,active,pair)
    def on_end(success: bool) -> Any:
        try:
            if success: guard.check(capture=True)
        except BaseException:
            end(False)
            raise
        return end(success)
    def on_reset(value: Any) -> Any:
        assert value is pipe, 'scope_stop_reset_pipe'
        return guard.reset(reset,value)
    def on_pipe_reset(self: Any, *args: Any, **kwargs: Any) -> Any:
        assert self is pipe, 'scope_stop_reset_self'
        return guard.reset(pipe_reset,*args,**kwargs)
    for name,value in dict(begin=on_begin,enqueue=on_enqueue,end=on_end,reset=on_reset).items():
        patch(stack,control,name,value)
    patch(stack,pipe,'reset',MethodType(on_pipe_reset,pipe))
    return guard
