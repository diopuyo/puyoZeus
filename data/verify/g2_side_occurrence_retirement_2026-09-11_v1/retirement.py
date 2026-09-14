"""原片側resetに続けて1Pのdirectional状態だけを退役する。新手権限は発行しない。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

SOURCE = Path(__file__).resolve().parent.parent / 'g2_directional_next_enqueue_2026-09-09_v2/occurrence.py'
SHA = 'b71a992a94d5231278242ac14b73dab725786e713f543956a33a9f40179b86dc'
SIDE, OTHER = '1P', '2P'


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('side_occurrence:' + reason)


class Retirement:
    def __init__(self, context: Any) -> None:
        self.context, self.pipe = context, context.pipe
        self.adapter = context.state['directional_next_runtime']['adapter']
        self.module = sys.modules[type(self.adapter).__module__]
        require(Path(self.module.__file__).resolve() == SOURCE and type(self.adapter) is self.module.Adapter
                and hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA, 'source')
        self.controller = context.factory.provider.journal.controller
        require(self.adapter.controller is self.controller and self.controller.active is None, 'controller')
        self.runtime = self.controller.instances[id(self.pipe)]
        require(self.runtime.pipe is self.pipe, 'runtime')
        self.key = (id(self.pipe), SIDE)
        self.old = self.adapter.states.get(self.key)
        self.old_value = deepcopy(self.old)
        self.history = self.runtime.histories[SIDE]
        require(type(self.history) is self.adapter.native.History, 'history_type')
        if self.old is not None:
            require(self.old['blocked'] is None and self.old['epoch'] == self.history.epoch, 'old_state')
        self.others = {key: (value, deepcopy(value)) for key, value in self.adapter.states.items() if key != self.key}
        self.other_history = self.runtime.histories[OTHER]
        self.other_history_value = asdict(self.other_history)

    def unchanged(self) -> None:
        require(self.context.state['directional_next_runtime']['adapter'] is self.adapter, 'adapter_changed')
        require(self.adapter.controller is self.controller and self.controller.active is None, 'active_invocation')
        require(self.controller.instances[id(self.pipe)] is self.runtime, 'runtime_changed')
        other = {key: value for key, value in self.adapter.states.items() if key != self.key}
        require(set(other) == set(self.others) and all(other[key] is value and other[key] == saved
                for key, (value, saved) in self.others.items()), 'other_state_changed')
        require(self.runtime.histories[OTHER] is self.other_history
                and asdict(self.other_history) == self.other_history_value, 'other_history_changed')

    def finish(self, frame: int) -> dict[str, Any]:
        self.unchanged()
        current = self.runtime.histories[SIDE]
        require(type(current) is self.adapter.native.History and current is not self.history
                and current.epoch == self.history.epoch + 1, 'new_history')
        require((current.accepted, current.clock, current.payload) == (None, None, None), 'enqueue_already_started')
        require(self.adapter.states.get(self.key) is self.old and self.old == self.old_value, 'old_state_changed')
        if self.old is not None:
            fresh = self.module.empty_state(current.epoch, None)
            fresh['retired_segment'] = self.old.get('segment') or self.old.get('retired_segment')
            self.adapter.states[self.key] = fresh
        self.unchanged()
        require(self.runtime.histories[SIDE] is current and current.clock is None, 'history_replaced')
        return dict(stage='side_occurrence_retired', frame=frame, old=self.old_value,
            new=deepcopy(self.adapter.states.get(self.key)), old_epoch=self.history.epoch, new_epoch=current.epoch,
            absent_noop=self.old is None, other_sides_unchanged=True, native_history_replaced=False,
            source_sha=SHA, pipe_id=id(self.pipe), emitted=False,
            journal_retirement_row=False, quality_gate_clear=False)


def save(context: Any, report: Any, original_error: Any) -> None:
    def encode(value: Any) -> Any:
        if is_dataclass(value): return asdict(value)
        raise TypeError('unsupported_retirement_evidence:' + type(value).__name__)
    try:
        with (context.state['output'] / 'SIDE_OCCURRENCE_RETIREMENT.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, default=encode, allow_nan=False)
    except BaseException as error:
        if original_error is None: raise
        context.rows.append(dict(stage='retirement_save_failed', error=repr(error)))


def derived(parent: Any) -> Any:
    class Context(parent):
        def perform(self, advisory: Any, frame: int, clock: float) -> None:
            report, error, proof = dict(stage='occurrence_preflight', frame=frame), None, None
            try:
                proof = Retirement(self)
                report.update(stage='original_side_reset', old=proof.old_value,
                              old_epoch=proof.history.epoch, source_sha=SHA)
                super().perform(advisory, frame, clock)
                report['stage'] = 'after_reset_retirement'
                report = proof.finish(frame)
            except BaseException as caught:
                error = caught
                report.update(error=repr(caught), quality_gate_clear=False)
                if proof is not None:
                    try:
                        report.update(new=deepcopy(proof.adapter.states.get(proof.key)),
                            state_changed=proof.adapter.states.get(proof.key) is not proof.old,
                            runtime_epoch=proof.runtime.histories[SIDE].epoch)
                    except BaseException as diagnostic:
                        report['failure_snapshot_error'] = repr(diagnostic)
                self.error = repr(caught)
                raise
            finally:
                save(self, report, error)
    return Context
