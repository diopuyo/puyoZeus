"""実J/historyの同frame資格だけから原Recovery/Leaseへ委譲する。"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any

KEY = 'live_empty_reset_context'
WAIT_FRAMES = 14


def require(ok: bool, reason: str) -> None:
    if not ok: raise ValueError('live_empty_reset:'+reason)


def history(state: Any, journal: Any, previous: int) -> tuple[Any,Any]:
    sink,meta = state['live_history_sink'],state['collector_metadata_sink']
    require(not sink.closed and not sink.errors and not meta.closed and not meta.errors and not meta.busy,
            'actual_stream_state')
    expected = ((sink.stream,'directional_history.jsonl'),(journal.stream,'atomic_journal.jsonl'),
                (meta.rows.stream,'collector_metadata.jsonl'))
    for stream,name in expected:
        require(Path(stream.name).resolve()==(state['output']/name).resolve() and not stream.closed,'actual_stream_path')
        stream.flush()
    read = lambda name: [json.loads(line) for line in (state['output']/name).read_text().splitlines()]
    history_rows,journal_rows = read('directional_history.jsonl'),read('atomic_journal.jsonl')
    require(history_rows and history_rows[-1]['scope']['frame_idx']==previous,'actual_history_tail')
    return history_rows,journal_rows


class Context:
    def __init__(self, stack: Any, state: Any, factory: Any, pipe: Any, modules: Any) -> None:
        self.stack,self.state,self.factory,self.pipe,self.modules = stack,state,factory,pipe,modules
        self.proof,self.recovery,self.error = None,None,None
        self.rows: list[Any] = []

    def perform(self, advisory: Any, frame: int, clock: float) -> None:
        try:
            Q,factory,state = self.modules.Q,self.factory,self.state
            guard = state['repeat_scope_guard']
            require(self.proof is None and guard.pipe is self.pipe and guard.factory is factory,'actual_guard')
            require(guard.frame==advisory.facts[-1].frame==frame-Q.E.STRIDE and clock==frame/Q.E.FPS,'actual_clock')
            rows,journal = history(state,factory.provider.journal,guard.frame)
            archive_module = Q.E.fixed('g2_empty_tail_archive_candidate_2026-09-10_v1/empty_archive.py')
            archive = archive_module.Archive(factory.controller,factory.controller.history['1P'])
            self.proof = Q.E.PrefixEvidence(factory,self.modules.parts,archive,journal,rows,self.modules.step,guard.frame)
            old = factory.controller.history['1P'].owner.state
            self.rows.append(dict(stage='qualified',frame=frame,old_integer_present=old.current is not None,
                                  old_owner=asdict(old),samecall=self.proof.result))
            prototype = Q.recovery(factory,self.pipe,state,guard.reset_lease.evidence)
            recovery_module = sys.modules[type(prototype).__module__]
            recovery_module.I = N(**(vars(recovery_module.I)|dict(DEADLINE=frame+WAIT_FRAMES)))
            self.recovery = recovery_module.install(self.stack,factory,self.pipe,state,guard.reset_lease.evidence)
            recovery_module.recording(self.stack,state['live_history_sink'],self.recovery)
            guard.reset_lease.empty_evidence = self.proof
            guard.reset_lease.perform(self.recovery,frame,clock)
            self.rows.append(dict(stage='reset_returned',frame=frame,missing_count='UNCERTIFIED'))
        except BaseException as exc:
            self.error = repr(exc)
            self.rows.append(dict(stage='failed',frame=frame,error=self.error))
            raise

    def close(self) -> None:
        arming = self.state.get('live_empty_arming')
        armed = None if arming is None else dict(events=arming.rows,
            entry=None if arming.entry is None else arming.entry.rows,
            error=None if arming.error is None else repr(arming.error))
        value = dict(events=self.rows,error=self.error,quality_gate_clear=False,
                     arming=armed,recovery=None if self.recovery is None else self.recovery.rows)
        with (self.state['output']/'LIVE_EMPTY_RESET.json').open('x',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
