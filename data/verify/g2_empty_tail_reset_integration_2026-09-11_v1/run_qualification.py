"""原collector76の同call成功後、実Recoveryへのempty資格を接続する。"""
from __future__ import annotations
import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace as N
from typing import Any
import empty_reset as E

ROOT = Path(__file__).resolve().parent
FINAL = ROOT.parent / 'g2_empty_tail_finalizer_2026-09-10_v1'
RECOVERY = ROOT.parent / 'g2_reset_recovery_candidate_2026-09-10_v1'
KEPT: dict[str, Any] = {}


def module(alias: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(alias, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[alias] = value
    spec.loader.exec_module(value)
    return value


def guard_module() -> Any:
    prior = sys.modules.get('reset_lease')
    try:
        sys.modules['reset_lease'] = E._ORIGINAL
        return E.fixed('g2_conditional_reset_integration_2026-09-10_v1/reset_guard.py')
    finally:
        if prior is None: sys.modules.pop('reset_lease', None)
        else: sys.modules['reset_lease'] = prior


def recovery(factory: Any, pipe: Any, state: Any, evidence: Any) -> Any:
    prior = sys.modules.get('reset_inputs')
    try:
        sys.modules['reset_inputs'] = module('_empty_qualification_reset_input', RECOVERY/'reset_inputs.py')
        value = module('_empty_qualification_recovery', RECOVERY/'recovery_connection.py')
        return value.Recovery(factory, pipe, state, evidence)
    finally:
        if prior is None: sys.modules.pop('reset_inputs', None)
        else: sys.modules['reset_inputs'] = prior


def qualified(state: Any, factory: Any, pipe: Any) -> None:
    output, kept = state['output'], KEPT['runtime']
    read = lambda name: [json.loads(line) for line in (output/name).read_text().splitlines()]
    history, journal = read('directional_history.jsonl'), read('atomic_journal.jsonl')
    guard = state['repeat_scope_guard']
    archive_module = E.fixed('g2_empty_tail_archive_candidate_2026-09-10_v1/empty_archive.py')
    archive = archive_module.Archive(factory.controller, factory.controller.history['1P'])
    step = module('_empty_qualification_step', ROOT.parent/'g2_private_suffix_fusion_2026-09-10_v1/samecall.py').journal_step
    proof = E.PrefixEvidence(factory, kept['empty_parts'], archive, journal, history, step, guard.frame)
    KEPT['prefix_evidence'] = proof
    receiver = recovery(factory, pipe, state, guard.reset_lease.evidence)
    lease = guard.reset_lease
    previous = lease.empty_evidence, lease.archive, lease.recovery
    try:
        lease.empty_evidence = proof
        lease.qualify(receiver, guard.frame+E.STRIDE, (guard.frame+E.STRIDE)/E.FPS)
    finally:
        lease.empty_evidence, lease.archive, lease.recovery = previous
    assert not guard.reset_lease.used and receiver.reset_count == 0
    row = dict(qualified=True, reset_executed=False, actual_Recovery=True, actual_factory=True,
        samecall=proof.result, archive=proof.receipt, prefix_frame=guard.frame,
        old_integer_coexistence=False, artificial_inputs=True, quality_gate_clear=False)
    with (output/'EMPTY_RESET_QUALIFICATION.json').open('x', encoding='utf-8') as stream:
        json.dump(row, stream, ensure_ascii=False, indent=2)


def wrap_loader(original: Any) -> Any:
    def load(alias: str, path: Path, stack: Any) -> Any:
        value = original(alias, path, stack)
        if alias == '_empty_fused_checks':
            before = value.verify
            def verify(state: Any, factory: Any, pipe: Any, rows: Any, trace: Any) -> Any:
                result = before(state, factory, pipe, rows, trace)
                qualified(state, factory, pipe)
                return result
            value.verify = verify
        return value
    return load


def checked_loader(selected: Any, base: Any) -> Any:
    original = F.checked_loader(selected, base)
    def load(alias: str, path: Path, stack: Any) -> Any:
        value = original(alias, path, stack)
        if alias == '_tail_suffix_shared_entry':
            previous = value.load
            def shared(inner: Any) -> Any:
                facade = previous(inner)
                common = value.C.R.COMMON.B.K
                def install(scope: Any, *args: Any) -> None:
                    old = common.load
                    def guarded(name: str, source: Path, sub: Any) -> Any:
                        loaded = old(name, source, sub)
                        return E.adapted_guard(loaded, guard_module()) if name == '_combined_scope_stop' else loaded
                    try:
                        common.load = guarded
                        facade.install(scope, *args)
                    finally:
                        common.load = old
                return N(**(vars(facade) | dict(install=install)))
            value.load = shared
        return value
    return load


def connector(base: Any, stack: Any, kept: Any) -> Any:
    KEPT['runtime'] = kept
    return F.connector(base, stack, kept)


def execute() -> int:
    tree = ast.parse((FINAL/'run_fused.py').read_bytes())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
    hits = 0
    for node in ast.walk(function):
        if isinstance(node, ast.With):
            for index, item in enumerate(list(node.body)):
                if ast.unparse(item) == 'spec.loader.exec_module(base)':
                    node.body.insert(index+1, ast.parse('base.load = wrap_loader(base.load)').body[0])
                    hits += 1
    assert hits == 1
    namespace = dict(vars(F), ROOT=ROOT, connector=connector, checked_loader=checked_loader, wrap_loader=wrap_loader)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), __file__, 'exec'), namespace)
    return namespace['execute']()


F = module('_original_empty_fused_runner', FINAL/'run_fused.py')

if __name__ == '__main__':
    raise SystemExit(FunctionType(F.main.__code__, dict(vars(F), ROOT=ROOT, execute=execute))())
