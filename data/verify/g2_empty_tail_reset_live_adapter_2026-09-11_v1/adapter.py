"""既存live constructor/collectorへ原empty資格の自動予約を限定追加する。"""
from __future__ import annotations
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace as N
from typing import Any

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
PREVIOUS = VERIFY/'g2_empty_tail_live_adapter_2026-09-10_v1'
RESET = VERIFY/'g2_empty_tail_reset_integration_2026-09-11_v1'
AUTO = VERIFY/'g2_empty_tail_auto_reset_2026-09-11_v1'
GUARD = VERIFY/'g2_collector_continuous_guard_2026-09-11_v1'


def dependencies(stack: Any, old: Any) -> Any:
    before = list(sys.path)
    stack.callback(sys.path.__setitem__,slice(None),before)
    sys.path[:0] = [str(RESET),str(AUTO),str(GUARD)]
    Q = old.load('run_qualification',RESET/'run_qualification.py',stack)
    observer = old.load('_live_empty_desync_observer',VERIFY/'g2_empty_tail_desync_trigger_2026-09-11_v1/observer.py',stack)
    sys.path.insert(0,str(VERIFY/'g2_empty_tail_desync_trigger_2026-09-11_v1'))
    import arming_v2
    import tail_auth
    if '_live_empty_reset_proof' not in sys.modules: stack.callback(sys.modules.pop,'_live_empty_reset_proof',None)
    raw = old.load('_live_empty_reset_raw',ROOT/'raw_entry.py',stack)
    step = old.load('_live_empty_reset_step',VERIFY/'g2_private_suffix_fusion_2026-09-10_v1/samecall.py',stack).journal_step
    return N(Q=Q,observer=observer,arming=arming_v2,tail=tail_auth,raw=raw,proof=raw.P,step=step)


def attach(stack: Any, factory: Any, pipe: Any, state: Any, modules: Any) -> None:
    modules.proof.require(modules.proof.KEY not in state,'duplicate_install')
    modules.proof.require(state['private_suffix_factory'] is factory,'original_factory')
    actual = factory.controller.empty_completion_parts
    modules.parts = N(basis=sys.modules['_empty_live_basis'],completion=actual.module,C=actual.C)
    modules.tail.install(stack,state)
    observer = modules.observer.install(stack,state['directional_next_runtime']['adapter'])
    context = modules.proof.Context(stack,state,factory,pipe,modules)
    state[modules.proof.KEY] = context
    stack.callback(context.close)
    modules.raw.install(stack,state,pipe,modules.arming.E.V)
    state['live_empty_arming'] = modules.arming.install(stack,pipe,state,observer,context.perform)


def constructor(stack: Any, old: Any, main: Any, modules: Any) -> None:
    repeated = main.__globals__['REPEAT']
    previous = repeated.load
    common = sys.modules['_attach_shared_entry'].C.R.COMMON.B.K
    def load(inner: Any) -> Any:
        candidate,hook = previous(inner)
        def install(scope: Any, factory: Any, pipe: Any, state: Any, rows: Any) -> None:
            original = common.load
            def guarded(alias: str, path: Path, sub: Any) -> Any:
                value = original(alias,path,sub)
                return modules.Q.E.adapted_guard(value,modules.Q.guard_module()) if alias=='_combined_scope_stop' else value
            try:
                common.load = guarded
                candidate.install(scope,factory,pipe,state,rows)
            finally:
                common.load = original
            attach(scope,factory,pipe,state,modules)
        return N(**(vars(candidate)|dict(install=install))),hook
    old.patch(stack,repeated,'load',load)


def history_sink(stack: Any, old: Any, main: Any) -> None:
    recording = main.__globals__['R']
    original = recording.install
    def install(inner: Any, state: Any, *args: Any) -> Any:
        value = original(inner,state,*args)
        if 'live_history_sink' in state: raise ValueError('duplicate_live_history_sink')
        state['live_history_sink'] = value
        return value
    old.patch(stack,recording,'install',install)


def configured(stack: Any) -> Any:
    import importlib.util
    spec = importlib.util.spec_from_file_location('_live_reset_previous_adapter',PREVIOUS/'adapter.py')
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    stack.callback(sys.modules.pop,spec.name,None)
    spec.loader.exec_module(old)
    main = old.configured(stack)
    modules = dependencies(stack,old)
    history_sink(stack,old,main)
    constructor(stack,old,main,modules)
    common = main.__globals__['K']
    old_guards = common.guards
    files = [p for root in (ROOT,RESET,AUTO,GUARD) for p in root.glob('*.py')]
    pins = {str(p):common.sha(p) for p in files}
    def guards() -> Any:
        modules.proof.require(all(common.sha(Path(p))==h for p,h in pins.items()),'source_changed')
        return old_guards()|pins
    helper = N(**(vars(common)|dict(ROOT=ROOT,guards=guards)))
    return FunctionType(main.__code__,dict(main.__globals__,K=helper),main.__name__,main.__defaults__,main.__closure__)
