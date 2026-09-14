"""旧sharedの実tailへ検収済みsuffix/私有配置と成功観測を一回追加する。"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace as N
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent/'g2_conditional_shared_runtime_candidate_2026-09-10_v1'
SUFFIX = ROOT.parent/'g2_hidden_tail_suffix_candidate_2026-09-10_v1'
PRIVATE = ROOT.parent/'g2_private_suffix_continuation_2026-09-10_v1'
EVIDENCE = ROOT.parent/'g2_private_suffix_evidence_2026-09-10_v1'
FACTORY = 'private_suffix_factory'
MODULES = 'private_suffix_modules'


def load_modules(stack: Any, load: Any, references: Any) -> Any:
    shared = sys.modules['_attach_shared_entry']
    assert Path(shared.__file__).resolve() == SHARED/'run_shared.py', 'private_shared_origin'
    tail = shared.C.R.H.TAIL
    history = tail.C.H
    assert Path(history.__file__).resolve() == ROOT.parent/'g2_hidden_tail_candidate_2026-09-10_v1/tail_history.py'
    previous = list(sys.path)
    stack.callback(sys.path.__setitem__, slice(None), previous)
    sys.path.insert(0, str(SUFFIX))
    addon = load('_private_live_suffix_install', SUFFIX/'install.py', stack)
    for module, name in ((addon.S, 'source.py'), (addon.V, 'votes.py'), (addon.C, 'consume.py')):
        assert Path(module.__file__).resolve() == SUFFIX/name, 'private_suffix_import_origin'
    names = ('basis', 'placement', 'exit', 'connection', 'origin')
    parts = {name: load('_private_live_'+name, PRIVATE/(name+'.py'), stack) for name in names}
    evidence = load('_private_live_evidence', EVIDENCE/'observer.py', stack)
    completion = load('_private_live_completion', ROOT/'completion.py', stack)
    return N(history=history, patch=tail.patch, addon=addon, basis=parts['basis'],
        placement=parts['placement'], exits=parts['exit'], connection=parts['connection'],
        origin=parts['origin'], evidence=evidence, completion=completion,
        references=references, verify=verify)


def install(stack: Any, factory: Any, pipe: Any, state: Any, modules: Any, write: Any) -> None:
    assert FACTORY not in state and MODULES not in state, 'private_suffix_duplicate_install'
    assert state['conditional_runtime_factory'] is factory, 'private_suffix_foreign_factory'
    control = factory.controller
    assert control.provider is factory.provider and not control.history, 'private_suffix_after_baseline'
    state[FACTORY], state[MODULES] = factory, modules
    modules.references.watch(stack, factory, state, modules, write)
    try:
        modules.origin.verify(modules.history)
        modules.addon.install(stack, modules.history, modules.patch)
        modules.connection.install(stack, factory, modules.patch, modules.history,
            modules.basis, modules.placement, modules.exits, control.hidden_history_rows)
        modules.evidence.install(stack, factory, modules.basis, modules.patch)
        modules.completion.install(stack, factory, modules.evidence, modules.basis,
            modules.placement, modules.patch)
        state[modules.references.KEY]['installed'] = True
    except BaseException as exc:
        state[modules.references.KEY]['error'] = type(exc).__name__+':'+str(exc)
        raise


def verify(state: Any) -> dict[str, Any]:
    factory, modules = state[FACTORY], state[MODULES]
    assert factory is state['conditional_runtime_factory'], 'private_suffix_foreign_factory'
    assert factory.controller.provider is factory.provider, 'private_suffix_provider_changed'
    return modules.references.verify(state)
