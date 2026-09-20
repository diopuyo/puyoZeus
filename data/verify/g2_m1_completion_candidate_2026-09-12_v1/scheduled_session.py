"""元基準/原capture/原保存を維持し、有資格採録二件と実終端を要求する。"""
from __future__ import annotations

from dataclasses import asdict
import json
import sys
from typing import Any

import old_session as BASE
import capture_schedule as S
import evaluation_flags as F
import capture_eligibility as E

ORIGINAL = BASE.Session.completed.__globals__
C, SAVE = ORIGINAL['C'], ORIGINAL['SAVE']
SERIAL, write = BASE.Session.close.__globals__['S'], BASE.Session.close.__globals__['write']


def save_close_failure(session: Any, error: BaseException, body: Any) -> None:
    value = dict(error=repr(error), original_body=None if body is None else repr(body),
                 schedule=asdict(session.schedule), flags=session.evaluation_flags.latest,
                 saved=session.saved, quality_gate_clear=False)
    try:
        write(session.state['output'] / 'M1_SESSION_CLOSE_FAILURE.json', value)
    except BaseException as caught:
        session.close_save_error = repr(caught)
        print('M1_CLOSE_FAILURE_SAVE_ERROR=' + repr(caught), file=sys.stderr, flush=True)


class Session(BASE.Session):
    def __init__(self, stack: Any, context: Any, policy: Any, physical: Any,
                 contract: Any, members: Any, frames: tuple[int, ...]) -> None:
        self.schedule, self.schedule_rows, self.schedule_stream = S.Schedule(), 0, None
        self.schedule_ready = False
        C.require(frames == S.EARLIEST, 'schedule_original_targets')
        super().__init__(stack, context, policy, physical, contract, members, frames)
        self.evaluation_flags = F.install(stack, self.journal)
        self.schedule_stream = (self.state['output'] / 'M1_CAPTURE_SCHEDULE.jsonl').open('x', encoding='utf-8')
        stack.callback(self.schedule_stream.close)
        self.schedule_ready = True

    def completed(self, frame: int) -> None:
        rec = self.state['provisional_context_observer']
        C.require(self.error is None, 'session_prior_error')
        C.require(rec.active is None and not rec.errors and rec.rows
                  and rec.rows[-1]['frame_idx'] == frame, 'session_after_original_update')
        self.basis()
        reasons = ('before_window',) if frame < S.EARLIEST[0] else E.reasons(self, frame)
        self.schedule, decision = S.observe(self.schedule, frame, reasons)
        packet = decision | dict(flags=self.evaluation_flags.latest, saved=False)
        try:
            if decision['action'] == 'REQUEST':
                C.require(not any(r['frame'] == frame for r in self.saved), 'session_duplicate_evaluation')
                path = self.state['output'] / f'BELIEF_M1_{frame}.json'
                saved = SAVE.run(self.capture, self.members, path, seed=frame)
                C.require(saved['frame'] == frame, 'schedule_saved_frame')
                self.saved.append(saved)
                self.schedule = S.accept(self.schedule, saved['frame'])
                packet['saved'] = True
        except BaseException as error:
            self.error = error
            packet['error'] = repr(error)
            raise
        finally:
            self.schedule_stream.write(json.dumps(packet, ensure_ascii=False, allow_nan=False) + '\n')
            self.schedule_stream.flush()
            self.schedule_rows += 1

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        if not self.schedule_ready:
            return super().close(kind, body, trace)
        try:
            modes = [dict(initial=m.initial_receipt, retired=m.retired_receipt, applied=m.applied,
                current=None if m.retired_receipt is not None else SERIAL.encode(m.connection.registry.current(m.connection.binding)),
                closed=m.closed, error=None if m.error is None else repr(m.error)) for m in self.modes]
            packet = dict(saved=self.saved, holds=self.holds, modes=modes, restored=self.restored,
                observer_closed=self.evidence.closed, witness_closed=self.witness.closed,
                error=None if body is None else repr(body), session_error=None if self.error is None else repr(self.error),
                observer_error=None if self.evidence.error is None else repr(self.evidence.error),
                witness_error=None if self.witness.error is None else repr(self.witness.error), quality_gate_clear=False,
                schedule=asdict(self.schedule), schedule_rows=self.schedule_rows, original_targets=self.frames,
                evaluation_flags_closed=self.evaluation_flags.closed,
                evaluation_flags_latest=self.evaluation_flags.latest,
                evaluation_flags_error=None if self.evaluation_flags.error is None else repr(self.evaluation_flags.error))
            write(self.state['output'] / 'BELIEF_M1_SESSION.json', packet)
            if body is None:
                C.require(self.error is None and self.evidence.error is None and self.witness.error is None
                          and all(m.error is None for m in self.modes), 'session_prior_error')
                C.require(self.restored and self.evidence.closed and self.witness.closed, 'session_cleanup')
                C.require(self.evaluation_flags.closed and self.evaluation_flags.error is None
                          and self.schedule_stream.closed, 'schedule_cleanup')
                C.require(tuple(r['frame'] for r in self.saved) == S.finish(self.schedule), 'session_evaluation_coverage')
        except BaseException as caught:
            self.error = caught
            save_close_failure(self, caught, body)
            if body is None: raise
        return False
