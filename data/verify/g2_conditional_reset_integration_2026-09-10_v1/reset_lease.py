"""原Recoveryによる一回resetと新基点待機を、旧scope保存へ束縛する。"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
import archive_state as A

SIDE, FPS, STRIDE = '1P', 60, 2


class Lease:
    """有限CPU用の資格。直接N resetや待機中の無資格resetは通さない。"""
    def __init__(self, guard: Any, evidence: Any) -> None:
        self.guard, self.evidence = guard, evidence
        self.raw_pipe, self.raw_native = guard.pipe.reset, guard.controller.reset
        self.active = self.used = self.waiting = False
        self.depth = self.outer_calls = self.native_calls = 0
        self.recovery: Any = None
        self.archive: Any = None
        self.new_scope: Any = None
        self.new_generation: int | None = None
        self.events: list[Any] = []

    def require(self, value: bool, reason: str) -> None:
        if not value:
            self.guard.stop('conditional_reset:'+reason)

    def qualify(self, recovery: Any, frame: int, clock: float) -> None:
        g = self.guard
        self.require(not self.active and not self.used, 'lease_reused')
        self.require(recovery.factory is g.factory and recovery.pipe is g.pipe, 'recovery_identity')
        self.require(type(frame) is int and type(clock) is float and clock == frame/FPS, 'clock')
        self.require(g.frame is not None and frame == g.frame+STRIDE, 'next_frame')
        g.check(capture=True)
        old = g.binding
        self.require(old is not None and recovery.pending is None and recovery.reset_count == 0, 'old_scope')
        self.require(not recovery.control.calls and not recovery.control.tickets
            and recovery.journal.active is None and g.controller.active is None, 'busy')
        state = old.owner.state
        registered = getattr(old,'conditional_firing_registered',None)
        self.require(type(registered) is dict and len(state.origins) == len(state.debts) == 1, 'conditional_origin')
        # 原Sはimmutable証拠を正規化するため、登録元と所有値の参照同一は要求しない。
        self.require(type(registered['origin']) is type(state.origins[0])
            and registered['origin'] == state.origins[0] == state.debts[0].origin
            and not state.consumed_ids, 'unsettled_origin')
        self.require(state.current is not None and getattr(old,'firing_ticket',None) is None, 'old_current_or_ticket')
        self.archive, self.recovery = A.Archive(recovery.control,old), recovery

    def perform(self, recovery: Any, frame: int, clock: float) -> Any:
        self.qualify(recovery,frame,clock)
        self.active = True
        self.events.append(dict(kind='reset_qualified',frame=frame,clock=clock,old=self.archive.receipt()))
        try:
            result = recovery.pipe.reset(match_start_sec=clock)
            self.require(self.outer_calls == self.native_calls == 1 and self.depth == 0, 'delegate_count')
            self.require(recovery.reset_count == 1 and recovery.pending is not None, 'recovery_not_waiting')
            self.require(recovery.control.history.get(SIDE) is None and len(recovery.archive) == 1, 'old_active_not_detached')
            self.require(recovery.archive[0][0] is self.archive.binding, 'archive_owner_mismatch')
            self.archive.verify()
            self.new_scope = self.evidence.scope(self.guard.factory,self.guard.pipe)
            old_scope = self.archive.binding.scope
            self.require(self.new_scope[:2] == old_scope[:2] and self.new_scope[2] == old_scope[2]+1
                and self.new_scope[3:5] == old_scope[3:5] and self.new_scope[-1] == old_scope[-1], 'reset_scope')
            self.used = self.waiting = True
            self.guard.binding = self.guard.expected_scope = self.guard.expected_owner_scope = None
            self.events.append(dict(kind='reset_waiting',scope=self.new_scope,outer_calls=1,native_calls=1))
            return result
        except BaseException as error:
            self.events.append(dict(kind='reset_failed',error=repr(error)))
            if self.guard.error is None:
                try:
                    self.guard.stop('conditional_reset:original_or_archive_failed')
                except RuntimeError:
                    pass
            raise
        finally:
            self.active = False

    def reset(self, original: Any, *args: Any, **kwargs: Any) -> Any:
        self.require(self.active and not self.used, 'reset_without_lease')
        if original == self.raw_pipe:
            self.require(self.outer_calls == self.depth == self.native_calls == 0, 'duplicate_outer')
            self.outer_calls, self.depth = 1, 1
            try:
                return original(*args,**kwargs)
            finally:
                self.depth = 0
        self.require(original == self.raw_native and self.depth == 1 and self.native_calls == 0, 'direct_or_duplicate_native')
        self.require(len(args) == 1 and args[0] is self.guard.pipe and not kwargs, 'native_pipe')
        self.native_calls = 1
        return original(*args,**kwargs)

    def observe_wait(self) -> None:
        self.require(not self.active, 'update_during_reset')
        if not self.waiting:
            return
        self.archive.verify()
        actual = self.evidence.scope(self.guard.factory,self.guard.pipe)
        self.require(actual[:5] == self.new_scope[:5] and actual[-1] == self.new_scope[-1], 'waiting_scope')
        old_generation = self.archive.binding.scope[5]
        self.require(type(actual[5]) is int and actual[5] >= old_generation, 'waiting_generation')
        if actual[5] > old_generation:
            if self.new_generation is None:
                self.new_generation = actual[5]
            self.require(actual[5] == self.new_generation, 'waiting_generation_changed')
        elif self.new_generation is not None:
            self.require(False,'waiting_generation_regressed')
        current = self.recovery.control.history.get(SIDE)
        if current is None:
            self.require(self.recovery.pending is not None and self.recovery.baseline_count == 0, 'waiting_ticket')
            return
        self.require(self.recovery.pending is None and self.recovery.baseline_count == 1, 'baseline_not_original')
        self.require(self.new_generation is not None and current.owner is not self.archive.owner
            and current.scope == actual, 'new_owner')
        self.require(current.owner.state.current is None and not current.owner.state.origins
            and not current.owner.state.debts, 'old_evidence_leaked')
        self.require(not any(n.startswith('conditional_') for n in vars(current)), 'old_conditional_attributes')
        self.waiting = False
        self.new_scope = actual
        self.events.append(dict(kind='new_scope_owned',state=asdict(current.owner.state),
            current_permission=False,physical_certified=False))
