"""2P初回基準に続く原Jを採録する。消費を着地・確率更新とは扱わない。"""
from __future__ import annotations

from types import SimpleNamespace as N
from typing import Any
import native_consumption as V
import second_basis as S
import journal_context as C
import second_retirement as RETIRE


class Mode:
    def __init__(self, evidence: Any, witness: Any, pipe: Any, registry: Any,
                 factory: Any, deadline: int, policy: Any, side: str = '2P') -> None:
        C.require(side in ('1P', '2P'), 'normal_tracking_side')
        self.side = side
        binding, receipt = S.initialize(evidence, witness, pipe, registry, factory, deadline, policy, side=side)
        self.evidence, self.witness = evidence, witness
        recovery = N(pipe=pipe, factory=factory, journal=witness.journal, error=None,
                     evidence=N(scope=self.scope))
        self.connection = N(binding=binding, registry=registry, recovery=recovery)
        self.native = V.Recorder(self.connection)
        self.initial_receipt = receipt
        self.activation = dict(frame=registry.current(binding).frame,
                               source_call_token=binding.initial_call_token)
        self.applied: list[Any] = []
        self.error: BaseException | None = None
        self.closed = False
        self.retired_receipt: Any = None
        evidence.registered_scope=binding.scope

    def scope(self, factory: Any, pipe: Any) -> tuple[Any, ...]:
        c = self.connection
        C.require(factory is c.registry.factory and pipe is c.recovery.pipe, 'second_tracking_owner')
        sm, journal = getattr(pipe, '_sm_' + self.side.lower()), c.recovery.journal
        frame = sm.context.frame_idx
        row = journal.scope(pipe, self.side, frame, frame/60)
        return (row['source_id'], row['run_id'], journal.epoch(pipe, self.side),
                id(pipe), id(sm), row['generation']['reset_epoch'], self.side)

    def observe(self, item: Any, error: Any, result: Any = None) -> Any:
        C.require(not self.closed and self.error is None and self.evidence.error is None
                  and not self.evidence.closed, 'second_tracking_lifetime')
        return self.native.observe(item, error)


def install(stack: Any, mode: Mode) -> None:
    journal = mode.connection.recovery.journal
    existed, previous = 'complete_step' in vars(journal), vars(journal).get('complete_step')
    original = journal.complete_step
    def completed(item: Any, result: Any, error: Any, profile: Any) -> Any:
        failure,retirement = None,None
        if item['scope']['side'] == mode.side and mode.retired_receipt is None:
            try:
                retirement=RETIRE.prepare(mode,item) if error is None and mode.error is None else None
                if retirement is None: mode.observe(item, error, result)
            except BaseException as caught: failure = caught
        try:
            returned = original(item, result, error, profile)
        except BaseException as caught:
            mode.error = caught
            raise
        if failure is not None:
            mode.error = failure
            raise failure
        if retirement is not None:
            try: RETIRE.commit(mode,retirement)
            except BaseException as caught:
                mode.error=caught
                raise
        return returned
    def close(kind: Any, body: Any, trace: Any) -> bool:
        try:
            C.require(vars(journal).get('complete_step') is completed, 'second_tracking_foreign_hook')
            setattr(journal, 'complete_step', previous) if existed else delattr(journal, 'complete_step')
        except BaseException as caught:
            mode.error = caught
            if body is None: raise
        finally: mode.closed = True
        return False
    stack.push(close)
    journal.complete_step = completed
