"""正常左右の同J初期化/更新を、原資格scheduleと親M1保存へ接続する。"""
from __future__ import annotations
from dataclasses import asdict
import json
from typing import Any
import journal_context as C
import journal_witness as W
import second_observation as O
import second_basis as B
import second_physical as P
import second_tracking as T
import live_binding_v3 as V
import trained_sidecar as SAVE
import serialization as S
import evaluation_flags as F
import capture_schedule as PLAN
import normal_settings as K
import normal_owner as OWN
import normal_eligibility as E


def write(path: Any, packet: Any) -> None:
    raw = json.dumps(packet, ensure_ascii=False, allow_nan=False, sort_keys=True).encode()
    with path.open('xb') as stream:
        stream.write(raw)
    C.require(path.read_bytes() == raw, 'normal_saved_bytes')


class Session:
    def __init__(self, stack: Any, context: Any, policy: Any, physical: Any,
                 contract: Any, members: Any, frames: tuple[int, ...]) -> None:
        C.require(frames == K.EARLIEST == PLAN.EARLIEST, 'normal_schedule_targets')
        self.stack, self.context, self.policy = stack, context, policy
        self.physical, self.contract, self.members, self.frames = physical, contract, members, frames
        self.state, self.factory, self.pipe = (context[k] for k in ('state', 'factory', 'pipe'))
        self.journal = self.factory.provider.journal
        self.current_modes: list[Any] = [None, None]
        self.modes: list[Any] = []
        self.saved: list[Any] = []
        self.holds: list[Any] = []
        self.error, self.restored, self.ready = None, False, False
        self.schedule, self.schedule_rows = PLAN.Schedule(), 0
        self.owner = OWN.install(stack, self.factory, self.pipe, self.state, K.INITIALIZATION_START, K.LAST)
        stack.push(self.close)
        self.witness = W.install(stack, self.journal)
        self.evidence = tuple(O.install(stack, self.journal, self.state, side=side) for side in K.SIDES)
        self.evaluation_flags = F.install(stack, self.journal)
        self.stream = stack.enter_context((self.state['output'] / 'M1_CAPTURE_SCHEDULE.jsonl').open('x', encoding='utf-8'))
        self.ready = True

    def basis(self, frame: int) -> None:
        if not self.owner.eligible_frame(frame):
            return
        for index, side in enumerate(K.SIDES):
            current = self.current_modes[index]
            if current is not None and not current.closed:
                continue
            try:
                mode = self.owner.create_mode(P, self.evidence[index], self.witness, self.policy, self.physical, side)
            except B.BasisHold as hold:
                self.holds.append(dict(frame=frame, side=side, reason=str(hold)))
                continue
            T.install(self.stack, mode)
            self.current_modes[index] = mode
            self.modes.append(mode)

    def capture(self) -> Any:
        self.owner.require_live()
        C.require(all(m is not None for m in self.current_modes), 'normal_capture_basis_missing')
        bindings = tuple(m.connection.binding for m in self.current_modes)
        return V.current(self.state['provisional_context_observer'], self.factory, self.pipe,
                         self.journal, self.owner.registry, bindings, self.contract,
                         self.witness, tuple(self.current_modes))

    def completed(self, frame: int) -> None:
        rec = self.state['provisional_context_observer']
        C.require(self.ready and self.error is None and rec.active is None and not rec.errors
                  and rec.rows and rec.rows[-1]['frame_idx'] == frame, 'normal_after_original_update')
        self.basis(frame)
        reasons = ('before_window',) if frame < self.frames[0] else E.reasons(self, frame)
        self.schedule, decision = PLAN.observe(self.schedule, frame, reasons)
        packet = decision | dict(flags=self.evaluation_flags.latest, saved=False)
        try:
            if decision['action'] == 'REQUEST':
                path = self.state['output'] / f'BELIEF_M1_{frame}.json'
                saved = SAVE.run(self.capture, self.members, path, seed=frame)
                C.require(saved['frame'] == frame, 'normal_saved_frame')
                self.saved.append(saved)
                self.schedule = PLAN.accept(self.schedule, frame)
                packet['saved'] = True
        except BaseException as error:
            self.error, packet['error'] = error, repr(error)
            raise
        finally:
            self.stream.write(json.dumps(packet, ensure_ascii=False, allow_nan=False) + '\n')
            self.stream.flush()
            self.schedule_rows += 1

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        if not self.ready:
            return False
        try:
            modes = [dict(side=m.side, initial=m.initial_receipt, retired=m.retired_receipt, applied=m.applied,
                current=None if m.retired_receipt is not None else S.encode(self.owner.registry.current(m.connection.binding)),
                closed=m.closed, error=None if m.error is None else repr(m.error)) for m in self.modes]
            packet = dict(saved=self.saved, holds=self.holds, modes=modes, restored=self.restored,
                observer_closed=all(e.closed for e in self.evidence), witness_closed=self.witness.closed,
                observer_errors=[None if e.error is None else repr(e.error) for e in self.evidence],
                witness_error=None if self.witness.error is None else repr(self.witness.error),
                error=None if body is None else repr(body), session_error=None if self.error is None else repr(self.error),
                schedule=asdict(self.schedule), schedule_rows=self.schedule_rows, original_targets=self.frames,
                evaluation_flags_closed=self.evaluation_flags.closed,
                evaluation_flags_error=None if self.evaluation_flags.error is None else repr(self.evaluation_flags.error),
                normal_initialization_start=K.INITIALIZATION_START, quality_gate_clear=False)
            write(self.state['output'] / 'BELIEF_M1_SESSION.json', packet)
            if body is None:
                C.require(self.error is None and all(e.error is None for e in self.evidence)
                    and self.witness.error is None and all(m.error is None for m in self.modes), 'normal_prior_error')
                C.require(self.restored and packet['observer_closed'] and self.witness.closed
                    and self.evaluation_flags.closed and self.evaluation_flags.error is None
                    and self.stream.closed and all(m.closed for m in self.modes), 'normal_cleanup')
                C.require(tuple(r['frame'] for r in self.saved) == PLAN.finish(self.schedule), 'normal_coverage')
        except BaseException as error:
            self.error = error
            try:
                write(self.state['output'] / 'M1_SESSION_CLOSE_FAILURE.json',
                      dict(error=repr(error), original_body=None if body is None else repr(body),
                           schedule=asdict(self.schedule), saved=self.saved, quality_gate_clear=False))
            except BaseException as save_error:
                self.close_save_error = repr(save_error)
            if body is None:
                raise
        return False
