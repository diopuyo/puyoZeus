"""原samecall検査の現役所有条件だけを、実resetで保存した旧所有権へ拡張する。"""
from __future__ import annotations
import ast
from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any
import empty_reset as E
import completion_code as CODE

SOURCE = E.VERIFY/'g2_empty_tail_finalizer_2026-09-10_v1/empty_completion.py'
OWNER_ASSERT = 'control.history[key[-1]] is binding and current.scope == state.scope'


def retired_owner(lease: Any, factory: Any, binding: Any, current: Any,
                  state: Any, key: Any) -> bool:
    assert type(lease) is E.Lease and type(lease.empty_evidence) is E.PrefixEvidence
    proof, recovery = lease.empty_evidence, lease.recovery
    assert lease.used and not lease.active and not lease.waiting and proof.used
    assert recovery.reset_count == recovery.baseline_count == 1 and recovery.pending is None
    assert lease.outer_calls == lease.native_calls == 1
    assert recovery.factory is proof.factory is factory and recovery.control is factory.controller
    assert recovery.journal is factory.provider.journal
    assert lease.archive is proof.archive and lease.archive.binding is binding is proof.binding
    lease.archive.verify()
    assert E.digest(lease.archive.receipt()) == proof.receipt_sha
    assert E.digest(proof.result) == proof.result_sha
    assert len(recovery.archive) == 1
    old, saved, record = recovery.archive[0]
    assert old is binding and binding.owner.state is saved is current
    assert asdict(saved) == record['owner'] and current.scope == state.scope
    active = factory.controller.history[key[-1]]
    assert active is lease.empty_new_binding and active.owner is lease.empty_new_owner, 'reset_new_binding'
    assert active.scope == lease.new_scope == recovery.evidence.scope(factory,recovery.pipe), 'reset_new_scope'
    assert active is not binding and active.owner is not binding.owner
    assert active.scope[2] == binding.scope[2]+1 and key[-1] == binding.scope[-1] == E.SIDE
    assert key[0] <= proof.frame and lease.retirement['old_scope'] == list(binding.scope)
    assert lease.retirement['archive_sha256'] == proof.receipt_sha
    assert lease.retirement['samecall'] == proof.result
    assert lease.retirement['missing_count'] == 'UNCERTIFIED'
    lease.observe_wait()
    return True


def verify(lease: Any, factory: Any) -> Any:
    proof = lease.empty_evidence
    assert type(proof) is E.PrefixEvidence
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == E.PINS[str(SOURCE.relative_to(E.VERIFY))]
    original = proof.parts.completion
    CODE.verify(original)
    assert Path(original.__file__).resolve() == SOURCE
    tree = ast.parse(raw)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify')
    targets = [n for n in ast.walk(function) if isinstance(n, ast.Assert) and ast.unparse(n.test) == OWNER_ASSERT]
    assert len(targets) == 1
    targets[0].test = ast.parse('retired_owner(lease, factory, binding, current, state, key)', mode='eval').body
    namespace = dict(vars(original), retired_owner=retired_owner, lease=lease)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(SOURCE), 'exec'), namespace)
    result = namespace['verify'](factory, proof.parts.C, proof.journal, proof.history, proof.step)
    assert result == proof.result
    return dict(original_samecall=result, retired_owner_verified=True,
        current_history_replaced=False, multi_scope_finalizer_verified=False, quality_gate_clear=False)
