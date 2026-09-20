"""原samecallを検査した非発火emptyだけに、保存付き原resetを許す。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace as N
from typing import Any
import completion_code as CODE

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
PINS = {
    'g2_conditional_reset_integration_2026-09-10_v1/reset_lease.py': '5d613358814adb9640d9cb7d36849f936bc6c22ac63a7fb043f1154397dfc134',
    'g2_conditional_reset_integration_2026-09-10_v1/reset_guard.py': '8562ba6976880f08aafbe18b141362534eb7a679bbb07072109a1b8f619a6c0f',
    'g2_empty_tail_archive_candidate_2026-09-10_v1/empty_archive.py': '2260627ec660d329c28785cb02ba1e67873dda6497dda5ac61296014921217a4',
    'g2_empty_tail_finalizer_2026-09-10_v1/empty_completion.py': '953469a284cab8376258cd07abebfb07fe3db45e76e37c7a936db49118dd6a85',
}
FPS, STRIDE, SIDE = 60, 2, '1P'


def fixed(relative: str) -> Any:
    path = VERIFY / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PINS[relative], 'empty_reset_source'
    alias = '_empty_reset_' + path.stem
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class PrefixEvidence:
    """リセット前の原検査成功と実参照を保持。保存値だけからは発行しない。"""
    def __init__(self, factory: Any, parts: Any, archive: Any,
                 journal: Any, history: Any, step: Any, frame: int) -> None:
        self.factory, self.parts, self.archive = factory, parts, archive
        self.journal, self.history, self.step, self.frame = journal, history, step, frame
        self.binding = factory.controller.history[SIDE]
        self.used = False
        self.result = self.check()
        self.result_sha = digest(self.result)
        self.receipt = deepcopy(archive.receipt())
        self.receipt_sha = digest(self.receipt)

    def check(self) -> Any:
        f, b, p = self.factory, self.binding, self.parts
        assert f.controller.history.get(SIDE) is b and self.archive.binding is b
        self.archive.verify()
        p.basis.validate(f.controller, b, b.empty_tail_basis)
        assert not b.owner.state.origins and not b.owner.state.debts and not b.owner.state.consumed_ids
        assert not b.empty_tail_basis.queue and b.next_token is b.next_started is b.candidate is None
        assert b.empty_tail_basis.token in b.consumed_tokens
        assert getattr(b, 'firing_ticket', None) is None
        assert getattr(b, 'conditional_firing_registered', None) is None
        assert max(r['scope']['frame_idx'] for r in self.history) == self.frame
        path = Path(p.completion.__file__).resolve()
        expected = VERIFY / 'g2_empty_tail_finalizer_2026-09-10_v1/empty_completion.py'
        assert path == expected and hashlib.sha256(path.read_bytes()).hexdigest() == PINS[str(path.relative_to(VERIFY))]
        CODE.verify(p.completion)
        result = p.completion.verify(f, p.C, self.journal, self.history, self.step)
        assert result['empty_commit_samecall_verified'] and result['commits'] > 0
        return result


def original_lease() -> Any:
    path = VERIFY / 'g2_conditional_reset_integration_2026-09-10_v1'
    previous, prior = list(sys.path), sys.modules.get('archive_state')
    try:
        sys.path.insert(0, str(path))
        if prior is not None:
            assert Path(prior.__file__).resolve() == path / 'archive_state.py'
        return fixed('g2_conditional_reset_integration_2026-09-10_v1/reset_lease.py')
    finally:
        sys.path[:] = previous
        if prior is None:
            sys.modules.pop('archive_state', None)


_ORIGINAL = original_lease()


class Lease(_ORIGINAL.Lease):
    def __init__(self, guard: Any, evidence: Any) -> None:
        super().__init__(guard, evidence)
        self.empty_evidence: PrefixEvidence | None = None
        self.retirement: Any = None
        self.empty_new_binding: Any = None
        self.empty_new_owner: Any = None

    def qualify(self, recovery: Any, frame: int, clock: float) -> None:
        proof = self.empty_evidence
        if proof is None:
            return super().qualify(recovery, frame, clock)
        g = self.guard
        self.require(type(proof) is PrefixEvidence and not proof.used, 'empty_evidence')
        self.require(not self.active and not self.used, 'lease_reused')
        self.require(recovery.factory is g.factory is proof.factory and recovery.pipe is g.pipe, 'recovery_identity')
        self.require(recovery.control is g.factory.controller
            and recovery.journal is g.factory.provider.journal, 'recovery_control_identity')
        self.require(type(frame) is int and type(clock) is float and clock == frame/FPS, 'clock')
        self.require(g.frame == proof.frame and frame == g.frame+STRIDE, 'next_frame')
        g.check(capture=True)
        self.require(g.binding is proof.binding and recovery.pending is None and recovery.reset_count == 0, 'old_scope')
        self.require(not recovery.control.calls and not recovery.control.tickets
            and recovery.journal.active is None and g.controller.active is None, 'busy')
        self.require(proof.check() == proof.result and digest(proof.result) == proof.result_sha,
            'empty_samecall_changed')
        self.require(digest(proof.archive.receipt()) == proof.receipt_sha, 'empty_archive_changed')
        self.archive, self.recovery = proof.archive, recovery

    def perform(self, recovery: Any, frame: int, clock: float) -> Any:
        result = super().perform(recovery, frame, clock)
        proof = self.empty_evidence
        if proof is not None:
            self.require(digest(proof.result) == proof.result_sha, 'empty_samecall_changed_after_reset')
            self.retirement = dict(kind='empty_scope_retirement/v1', old_scope=list(proof.binding.scope),
                archive_sha256=proof.receipt_sha, samecall=deepcopy(proof.result), missing_count='UNCERTIFIED')
            recovery.pending['empty_retirement'] = deepcopy(self.retirement)
            proof.used = True
        return result

    def observe_wait(self) -> None:
        previously_waiting = self.waiting
        if self.retirement is not None and self.recovery.pending is not None:
            self.require(self.recovery.pending.get('empty_retirement') == self.retirement, 'retirement_changed')
        super().observe_wait()
        if self.used and not self.waiting and self.empty_evidence is not None:
            current = self.recovery.control.history[SIDE]
            if previously_waiting:
                self.require(self.empty_new_binding is self.empty_new_owner is None, 'new_binding_already_captured')
                self.empty_new_binding, self.empty_new_owner = current, current.owner
            self.require(current is self.empty_new_binding and current.owner is self.empty_new_owner, 'reset_new_binding')
            self.require(current.scope == self.new_scope, 'reset_new_scope')
            self.require(not any(n.startswith('empty_') for n in vars(current)), 'old_empty_attributes')
            rows = [r for r in self.recovery.rows if r['kind'] == 'new_baseline']
            self.require(len(rows) == 1 and rows[0]['proof']['reset_source'].get('empty_retirement')
                == self.retirement, 'baseline_retirement_link')

def adapted_guard(original: Any, guard: Any) -> Any:
    adapted = FunctionType(guard.adapted.__code__, dict(guard.adapted.__globals__, L=N(Lease=Lease)))
    return adapted(original)
