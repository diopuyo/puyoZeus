"""原v9の構成を保持し、継続prefixと保存完全連結を追加する候補。"""
from __future__ import annotations
from pathlib import Path
from types import FunctionType, SimpleNamespace as N
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'g2_private_suffix_live_adapter_2026-09-10_v1'
ROLLING = ROOT.parent/'g2_rolling_two_hand_prefix_2026-09-10_v1'
LINK = ROOT.parent/'g2_rolling_prefix_saved_link_2026-09-10_v1'
BRIDGE = ROOT.parent/'g2_rolling_stage1_bridge_2026-09-10_v1'
KEY = 'rolling_prefix_references'
IMPORTS = ('rolling_prepare', 'rolling_commit', 'rolling_link_fixed', 'rolling_link_journal')


def imports(stack: Any) -> None:
    """追加部品の通常importも元のmodule登録へ復元する。"""
    missing = object()
    before = {name:sys.modules.get(name, missing) for name in IMPORTS}
    def restore() -> None:
        for name, module in before.items():
            if module is missing: sys.modules.pop(name, None)
            else: sys.modules[name] = module
    stack.callback(restore)


def targets(factory: Any) -> dict[str, Any]:
    cls = type(factory.controller)
    return dict(prepared=cls.prepared.__globals__['V1'].prepared,
        call=cls.call, consumed=cls.consumed_history)


def watch(stack: Any, factory: Any, state: Any, write: Any) -> None:
    assert KEY not in state
    before = targets(factory)
    state[KEY] = dict(installed=False, closed=False, restored=False)
    def closed() -> None:
        after = targets(factory)
        equal = {name:after[name] is function for name,function in before.items()}
        state[KEY].update(closed=True, restored=all(equal.values()), equal=equal)
        write(state['output']/'ROLLING_PREFIX_REFERENCES.json', state[KEY])
        write(state['output']/'ROLLING_PREPOP_CHECKS.json',
            getattr(factory.controller, 'hidden_rolling_prepop_checks', []))
        assert state[KEY]['restored'], 'rolling_reference_restore'
    stack.callback(closed)


def connect(stack: Any, previous: Any, connection: Any) -> None:
    extra = sys.modules['_private_live_extra']
    original = extra.install
    def install(inner: Any, factory: Any, pipe: Any, state: Any, modules: Any, write: Any) -> None:
        original(inner, factory, pipe, state, modules, write)
        shared = sys.modules['_attach_shared_entry']
        watch(inner, factory, state, write)
        connection.install(inner, factory, modules.patch, shared.C.R.H.PREFIX.C.H,
            modules.history.T.V, modules.addon.S.owner)
        state[KEY]['installed'] = True
    previous.patch(stack, extra, 'install', install)


def finalize(stack: Any, previous: Any, main: Any, checker: Any, joining: Any) -> None:
    closure = main.__globals__['Q']
    original = closure.FINAL.evaluate
    def evaluate(goals: Any, rows: Any, legal: Any, output: Any, state: Any) -> Any:
        receipt = state[KEY]
        assert receipt['installed'] and receipt['closed'] and receipt['restored']
        import json
        with (output/'atomic_journal.jsonl').open() as stream:
            journal = [json.loads(line) for line in stream]
        linked = checker.check(rows, journal)
        closure.K.write(output/'ROLLING_SAVED_LINK.json', linked)
        del journal
        joined = joining.check(state['private_suffix_factory'].controller.hidden_rolling_prepop_checks,
            linked, rows)
        closure.K.write(output/'ROLLING_PREPOP_JOIN.json', joined)
        result = original(goals, rows, legal, output, state)
        return result | dict(rolling_saved_link=linked, rolling_prepop_join=joined, quality_gate_clear=False)
    previous.patch(stack, closure.FINAL, 'evaluate', evaluate)


def configured(stack: Any) -> Any:
    import importlib.util
    spec = importlib.util.spec_from_file_location('_rolling_previous_adapter', PREVIOUS/'adapter.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    main = previous.configured(stack)
    imports(stack)
    paths = list(sys.path)
    stack.callback(sys.path.__setitem__, slice(None), paths)
    sys.path[:0] = [str(ROLLING), str(LINK)]
    connection = previous.load('_rolling_live_connection', ROLLING/'connection.py', stack)
    checker = previous.load('_rolling_live_saved_link', LINK/'saved_link.py', stack)
    joining = previous.load('_rolling_live_prepop_join', ROOT/'prepop_join.py', stack)
    bridge = previous.load('_rolling_live_stage1_bridge', BRIDGE/'bridge.py', stack)
    stage1 = previous.load('_rolling_live_stage1_connection', ROOT/'stage1_connection.py', stack)
    stage1.install(stack, previous, bridge)
    connect(stack, previous, connection)
    finalize(stack, previous, main, checker, joining)
    common, old_guards = main.__globals__['K'], main.__globals__['K'].guards
    paths = [p for folder in (ROOT, ROLLING, LINK, BRIDGE) for p in folder.glob('*.py')]
    pins = {str(p): common.sha(p) for p in paths}
    def guards() -> Any:
        assert all(common.sha(Path(path)) == digest for path,digest in pins.items()), 'rolling_source_changed'
        return old_guards() | pins
    helper = N(**(vars(common) | dict(ROOT=ROOT, guards=guards)))
    return FunctionType(main.__code__, dict(main.__globals__, K=helper),
        main.__name__, main.__defaults__, main.__closure__)
