"""既存公開full CPUのsessionへ独立consumerだけを追加する。"""
from __future__ import annotations
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from types import FunctionType
from typing import Any

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'g2_history_publication_full_runtime_2026-09-09_v1'


def load_adapter() -> Any:
    alias = '_postcommit_outer_comparison_adapter'
    if alias in sys.modules:
        raise RuntimeError('consumer_adapter_collision')
    spec = importlib.util.spec_from_file_location(alias,ROOT/'adapter.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


A = load_adapter()


def clone(original: Any, **changes: Any) -> Any:
    value = FunctionType(original.__code__,dict(original.__globals__,**changes),original.__name__,
                         original.__defaults__,original.__closure__)
    value.__kwdefaults__ = original.__kwdefaults__
    return value


def loaded(stack: Any) -> Any:
    alias, path = '_postcommit_full_publication',OLD/'assembly_publication.py'
    A.require(alias not in sys.modules,'consumer_assembly_collision')
    spec = importlib.util.spec_from_file_location(alias,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    stack.callback(sys.modules.pop,alias,None)
    spec.loader.exec_module(module)
    return module


def checked(original: Any) -> Any:
    def run(output: Path) -> Any:
        result = original(output)
        status = json.loads((output/A.STATUS).read_bytes())
        comparison = json.loads((output/A.COMPARISON).read_bytes())
        rows = json.loads((output/A.ROWS).read_bytes())
        A.require(status['closed'] and not status['errors'] and status['updates']==len(rows)==10,
                  'consumer_full_close')
        A.require(comparison['all_consumers_ready'] and all(comparison['equal'].values()),
                  'consumer_full_no_event_difference')
        A.require(not any(row['changed_sides'] for row in rows),'consumer_unexpected_event')
        result['postcommit_consumer'] = dict(updates=len(rows),closed=True,all_equal=True,
            comparison_connected=True,actual_publication_event=False,actual_collector_append_verified=False)
        return result
    return run


def main() -> int:
    with ExitStack() as stack:
        previous_paths = list(sys.path)
        sys.path[:] = [p for p in sys.path if Path(p).resolve()!=ROOT]
        stack.callback(setattr,sys,'path',previous_paths)
        p = loaded(stack)
        runner = p.load(stack,'_postcommit_full_parent_runner',OLD/'run_cpu.py',{'assembly_publication':p})
        original_install, original_guards = p.install,p.guards
        own = [ROOT/n for n in ('adapter.py','test_adapter.py','run_full_cpu.py','PLAN.md')]
        extra = {str(path):A.sha(path) for path in own}
        def install(inner: Any, collector: Any, state: Any) -> Any:
            receiver = original_install(inner,collector,state)
            A.install(inner,state)
            return receiver
        def guards() -> Any:
            A.require(all(A.sha(Path(path))==value for path,value in extra.items()),'consumer_own_changed')
            return original_guards() | extra
        p.install,p.guards = install,guards
        stack.callback(setattr,p,'install',original_install)
        stack.callback(setattr,p,'guards',original_guards)
        execute = lambda original:checked(runner.execute(original))
        return clone(runner.main,ROOT=ROOT,execute=execute)()


if __name__ == '__main__':
    raise SystemExit(main())
