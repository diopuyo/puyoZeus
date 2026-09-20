"""原86更新の実factoryを保持し、閉じた保存票と二段集計へ接続する。"""
from __future__ import annotations
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from types import FunctionType, ModuleType, SimpleNamespace as N
from typing import Any
import fusion as F

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent / 'g2_conditional_shared_runtime_candidate_2026-09-10_v1'
OBS = ROOT.parent / 'g2_conditional_current_call_evidence_2026-09-10_v2'
JOIN = ROOT.parent / 'g2_conditional_current_call_join_2026-09-10_v1'
LIVE = ROOT.parent / 'g2_conditional_live_adapter_2026-09-10_v3'


def load(alias: str, path: Path, stack: Any) -> Any:
    spec = F.importlib.util.spec_from_file_location(alias, path)
    module = F.importlib.util.module_from_spec(spec)
    assert alias not in sys.modules
    sys.modules[alias] = module
    stack.callback(sys.modules.pop, alias, None)
    spec.loader.exec_module(module)
    return module


def loop(old: Any, facade: Any, observer: Any, connection: Any, args: Any, inner: Any, data: Any) -> None:
    def patch(scope: Any, obj: Any, name: str, value: Any) -> None:
        before = getattr(obj, name)
        scope.callback(setattr, obj, name, before)
        setattr(obj, name, value)
    def install(scope: Any, factory: Any, pipe: Any, state: Any, rows: Any) -> None:
        facade.install(scope, factory, pipe, state, rows)
        connection.install(scope, observer, factory, state, patch, old.E.K.write)
        data['admission'] = state['conditional_full_rows']
    def installed(*values: Any) -> None:
        pass
    shared = N(**(vars(old.E.H.C) | dict(install=install)))
    history = N(**(vars(old.E.H) | dict(C=shared)))
    execution = N(**(vars(old.E) | dict(H=history)))
    FunctionType(old.loop.__code__, dict(vars(old), E=execution, C=N(install=installed)))(args, inner, data)


def finish(old: Any, kept: Any, stack: Any) -> Any:
    state, factory = kept['state'], kept['factory']
    output = state['output']
    read = lambda name: json.loads((output / name).read_text())
    rows = [json.loads(line) for line in (output / 'directional_history.jsonl').read_text().splitlines()]
    journal = [json.loads(line) for line in (output / 'atomic_journal.jsonl').read_text().splitlines()]
    helper = ModuleType('common')
    helper.FIRST, helper.END, helper.HISTORY_FIRST = old.I.FRAMES[0], old.I.FRAMES[-1]+old.I.STRIDE, rows[0]['scope']['frame_idx']
    helper.FPS, helper.STRIDE, helper.FRAMES = old.I.INPUT.FPS, old.I.STRIDE, old.I.FRAMES
    helper.require, helper.read = old.E.K.require, old.E.K.read
    previous = sys.modules['common']
    try:
        sys.modules['common'] = helper
        goals = load('_fusion_runtime_goals', old.E.K.VERIFY / 'g2_history_publication_probe_runtime_2026-09-10_v13/goals.py', stack)
    finally:
        sys.modules['common'] = previous
    control = factory.controller
    result = F.evaluate(goals, rows, control.legal, output, firing_rows=read('ADMISSION.json')['firing'],
        journal_rows=journal, conditional_rows=state['conditional_full_rows'],
        hidden_events=control.hidden_current_events, hidden_history=control.hidden_history_rows,
        hidden_lifetime=control.hidden_lifetime_rows, outer_rows=read('POSTCOMMIT_CONSUMER_ROWS.json'),
        controller=control, factory=factory)
    assert result['runtime_finalization_allowed'] and result['world_PB_verified']
    old.E.K.write(output / 'FUSED_RUNTIME.json', result)
    return result


def main() -> int:
    with ExitStack() as stack:
        paths = list(sys.path)
        stack.callback(sys.path.__setitem__, slice(None), paths)
        sys.path.insert(0, str(SHARED))
        sys.path.insert(0, str(OBS))
        observer = load('_fusion_capture_observer', OBS / 'observer.py', stack)
        connection = load('_fusion_capture_connection', LIVE / 'call_connection.py', stack)
        shared = load('_fusion_shared_entry', SHARED / 'run_shared.py', stack)
        facade = shared.load(stack)
        old = load('_fusion_original_86', shared.C.R.NEXT / 'run_next.py', stack)
        kept: dict[str, Any] = {}
        def selected(args: Any, inner: Any, data: Any) -> None:
            loop(old, facade, observer, connection, args, inner, data)
        def drive(*args: Any) -> Any:
            state, pipe, factory = args[2], args[3][0], args[6]
            before = facade.references(factory, pipe, state)
            kept.update(state=state, factory=factory)
            try:
                return FunctionType(old.drive.__code__, dict(vars(old), loop=selected))(*args)
            finally:
                assert facade.references(factory, pipe, state) == before
        def guards() -> Any:
            return old.guards() | facade.guards() | {str(p): old.E.K.sha(p)
                for directory in (ROOT, F.STAGE1, F.WORLD, OBS, JOIN, LIVE) for p in directory.glob('*.py')}
        result = FunctionType(old.main.__code__, dict(vars(old), ROOT=ROOT, drive=drive, guards=guards))()
        if result == 0:
            connection.verify(kept['state'])
            finish(old, kept, stack)
        return result


if __name__ == '__main__':
    raise SystemExit(main())
