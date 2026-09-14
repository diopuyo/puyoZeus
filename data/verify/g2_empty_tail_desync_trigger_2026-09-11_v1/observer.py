"""原enqueueの終了した実更新にだけ束縛する、権限を持たない不整合信号。"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MethodType
from typing import Any

from desync_signal import Fact, Matcher, valid

ORIGINAL_SHA = 'b71a992a94d5231278242ac14b73dab725786e713f543956a33a9f40179b86dc'
FPS = 60


def require(condition: bool, name: str) -> None:
    if not condition:
        raise ValueError('desync_observer:' + name)


@dataclass(frozen=True)
class Advisory:
    facts: tuple[Fact, Fact]
    invocation: Any
    state: Any
    missing_count: str = 'UNCERTIFIED'
    reset_permission: bool = False
    current_permission: bool = False


class Observer:
    def __init__(self, adapter: Any, side: str) -> None:
        method = type(adapter).emit
        require(sha256(Path(method.__code__.co_filename).read_bytes()).hexdigest()
                == ORIGINAL_SHA, 'original_adapter_source')
        require(method.__globals__['Adapter'] is type(adapter), 'original_adapter_type')
        require(side in ('1P', '2P') and adapter.enabled is True, 'active_side')
        self.adapter, self.side = adapter, side
        self.matcher, self.pending = Matcher(), None
        self.original = adapter.emit
        self.candidate_type = method.__globals__['MotionCandidate']

    def observe(self, pipe: Any, side: str, frame: int, reason: str, committed: bool) -> None:
        if side != self.side:
            return
        self.pending = None
        if reason != 'await_dnext_successor' or committed:
            self.matcher.push(None)
            return
        fact, invocation, state = self.fact(pipe, frame)
        facts = self.matcher.push(fact)
        if facts is not None:
            self.pending = Advisory(facts, invocation, state)

    def fact(self, pipe: Any, frame: int) -> tuple[Fact | None, Any, Any]:
        adapter, side = self.adapter, self.side
        active = adapter.controller.active
        require(active is not None, 'actual_invocation')
        invocation = adapter.controller._invocation(pipe, frame, active.time_sec)
        require(invocation.error is None and invocation.accounting_started, 'actual_accounting')
        runtime = invocation.runtime
        state, history = adapter.states[(id(pipe), side)], runtime.histories[side]
        require(history.clock == frame and state['blocked'] is None, 'actual_history')
        active, pair, detail, epoch, geometry = history.payload
        require(active is True and epoch == state['epoch'], 'actual_epoch')
        require(geometry is not None and geometry[0] == state['segment'], 'actual_segment')
        candidate = state['pending']
        require(type(candidate) is self.candidate_type and candidate == state['last_candidate'],
                'actual_pending_candidate')
        quiet, observed_detail = adapter.native.NextEnqueueController._quiet(
            adapter.controller, invocation, side, pair)
        require(detail == observed_detail, 'actual_quiet_detail')
        dto = getattr(invocation.main, 'p1' if side == '1P' else 'p2')
        fields = (candidate.sequence_number, candidate.first_support_frame,
                  candidate.available_frame, candidate.reference)
        scope = (adapter.source_id, adapter.run_id, id(pipe), side, epoch, state['segment'])
        fact = Fact(frame, scope, state['baseline_frame'], state['baseline_dnext'],
                    pair, dto.dnext_pair, fields, geometry[1], quiet)
        empty = not getattr(pipe, '_pending_tsumo_' + side.lower())
        return (fact if valid(fact) and empty else None), invocation, state

    def take(self, pipe: Any, frame: int) -> Advisory | None:
        value = self.pending
        if value is None:
            return None
        adapter, control = self.adapter, self.adapter.controller
        invocation, last = value.invocation, value.facts[-1]
        runtime = invocation.runtime
        require(control.active is None and not runtime.busy, 'update_not_completed')
        require(runtime.pipe is pipe and control.instances.get(id(pipe)) is runtime, 'actual_pipe')
        require(runtime.completed == frame == invocation.frame == last.frame, 'completed_frame')
        require(invocation.error is None, 'failed_update')
        require(adapter.states[(id(pipe), self.side)] is value.state, 'actual_state_reference')
        require(runtime.histories[self.side].epoch == last.scope[4], 'completed_epoch')
        require(not getattr(pipe, '_pending_tsumo_' + self.side.lower()), 'nonempty_FIFO')
        self.pending = None
        return value


def install(stack: ExitStack, adapter: Any, side: str = '1P') -> Observer:
    observer = Observer(adapter, side)
    existed, previous = 'emit' in vars(adapter), vars(adapter).get('emit')
    def emit(instance: Any, pipe: Any, actual_side: str, frame: int,
             reason: str, committed: bool) -> None:
        observer.original(pipe, actual_side, frame, reason, committed)
        observer.observe(pipe, actual_side, frame, reason, committed)
    def restore() -> None:
        if existed:
            adapter.emit = previous
        else:
            vars(adapter).pop('emit', None)
    stack.callback(restore)
    adapter.emit = MethodType(emit, adapter)
    return observer
