"""旧終了条件を保持し、私有履歴だけを構造/当時の実証拠/worldへ三段結合する。"""
from __future__ import annotations
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT.parent/'g2_private_suffix_finalizer_stage1_2026-09-10_v1'
WORLD = ROOT.parent/'g2_private_suffix_world_2026-09-10_v1'
STEP = ROOT.parent/'g2_private_suffix_fusion_2026-09-10_v1/samecall.py'


def load(stack: Any, alias: str, path: Path) -> Any:
    assert alias not in sys.modules
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    stack.callback(sys.modules.pop, alias, None)
    spec.loader.exec_module(module)
    return module


def read(path: Path) -> Any:
    text = path.read_text()
    return [json.loads(line) for line in text.splitlines()] if path.suffix == '.jsonl' else json.loads(text)


def context(state: Any) -> Any:
    original = state['repeated_firing_constructor']
    assert original['installed'] and original['closed'] and original['references_restored']
    factory = state['conditional_runtime_factory']
    assert factory is state['private_suffix_factory']
    control = factory.controller
    assert control.provider is factory.provider and not control.calls and not control.tickets
    state['combined_restored']()
    assert state['conditional_full_installs'] == state['conditional_revision_connection']['installs'] == 1
    assert state['combined_install']['configure_calls'] == 1
    state['private_suffix_modules'].verify(state)
    return factory


def audit(goals: Any, rows: Any, legal: Any, output: Any, *, factory: Any, parts: Any,
          firing_rows: Any, conditional_rows: Any, original: Any = None) -> dict[str, Any]:
    control = factory.controller
    with ExitStack() as stack:
        before = list(sys.path)
        stack.callback(sys.path.__setitem__, slice(None), before)
        sys.path[:0] = [str(STAGE1), str(WORLD)]
        stage1 = load(stack, '_private_live_finalizer_stage1', STAGE1/'compat.py')
        if not stage1.selected(rows):
            assert not parts.evidence.verify(control) and not control.private_suffix_completion_rows
            assert original is not None
            return original()
        world = load(stack, '_private_live_finalizer_world', WORLD/'world.py')
        step = load(stack, '_private_live_finalizer_step', STEP).journal_step
        with world.libraries().G.libraries():
            pass
        journal = read(output/'atomic_journal.jsonl')
        result = stage1.audit_stage1(goals, rows, legal, output,
            firing_rows=firing_rows, journal_rows=journal,
            conditional_rows=conditional_rows, controller=control, factory=factory)
        checked = world.verify_world(history_rows=rows, journal_rows=journal,
            conditional_rows=conditional_rows, hidden_events=control.hidden_current_events,
            hidden_history=control.hidden_history_rows, hidden_lifetime=control.hidden_lifetime_rows,
            outer_rows=read(output/'POSTCOMMIT_CONSUMER_ROWS.json'), controller=control, factory=factory)
        observed = parts.completion.verify(factory, parts.evidence, journal, rows, step)
        allowed = (result['private_suffix_structure_verified'] and result['same_live_controller_verified']
            and checked['world_PB_verified'] and checked['actual_live_scope_verified']
            and observed['private_commit_samecall_verified'])
        assert allowed
        return result | dict(conditional_world=checked, private_samecall=observed,
            runtime_finalization_allowed=allowed, world_PB_verified=checked['world_PB_verified'],
            conditional_current_counted_as_integer=False, physical_certified=False,
            production_permission=False, quality_gate_clear=False)


def evaluate(goals: Any, rows: Any, legal: Any, output: Any, state: Any) -> dict[str, Any]:
    factory = context(state)
    return audit(goals, rows, legal, output, factory=factory, parts=state['private_suffix_modules'],
        firing_rows=state['repeated_firing_constructor']['rows'], conditional_rows=state['conditional_full_rows'],
        original=lambda: state['private_suffix_original_finalizer'](goals, rows, legal, output, state))
