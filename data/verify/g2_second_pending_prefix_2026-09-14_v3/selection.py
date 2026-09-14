"""元cold loaderの最終2P Session選択へ私有型を接続。旧1P型は不変更。"""
from __future__ import annotations
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent/'g2_second_pending_prefix_2026-09-14_v1'
VERIFY = ROOT.parent
SECOND = VERIFY / 'g2_m1_completion_candidate_2026-09-12_v1/second_mode_binding.py'
PREFIX = VERIFY / 'g2_prefix_lane_integration_2026-09-13_v1'
BACK = VERIFY / 'g2_arrival_backlog_2026-09-13_v1'
NAMES = ('_g2_second_prefix_core','_g2_second_prefix_physical','_g2_second_prefix_session','_g2_second_prefix_saved','_g2_second_prefix_composition')


def dependencies(load: Any, owned: dict) -> Any:
    if owned:
        if set(owned)!=set(NAMES):
            raise ValueError('second_prefix_partial_dependencies')
        for alias,(module,path) in owned.items():
            if sys.modules.get(alias) is not module or Path(module.__file__).resolve()!=path:
                raise ValueError('second_prefix_private_owner_changed')
        return owned[NAMES[2]][0]
    math = sys.modules.get('_g2_prefix_math')
    decision = sys.modules.get('_g2_prefix_decision')
    if math is None or Path(math.__file__).resolve()!=BACK/'prefix_phase_math_v2.py':
        raise ValueError('second_prefix_original_math_missing')
    if decision is None or Path(decision.__file__).resolve()!=PREFIX/'stable_decision.py':
        raise ValueError('second_prefix_original_decision_missing')
    def private(alias: str, filename: str, injection: dict) -> Any:
        if alias in sys.modules:
            raise ValueError('second_prefix_foreign_alias')
        path = (ROOT if filename in ('session_binding.py','mode_composition.py') else BASE)/filename
        try:
            return load(alias,path,injection)
        finally:
            module = sys.modules.get(alias)
            source = getattr(module,'__file__',None)
            if isinstance(source,(str,Path)) and Path(source).resolve()==path:
                owned[alias] = (module,path)
    core = private(NAMES[0],'consumed_prefix.py',{'prefix_phase_math_v2':math})
    physical = private(NAMES[1],'physical_adapter.py',{'consumed_prefix':core,'stable_decision':decision})
    notice = sys.modules.get('_g2_second_notice_v21')
    if notice is None:
        raise ValueError('second_prefix_original_notice_missing')
    composition = private(NAMES[4],'mode_composition.py',{'settled_notice':notice})
    session = private(NAMES[2],'session_binding.py',{'physical_adapter':physical,'mode_composition':composition})
    private(NAMES[3],'saved_replay.py',{'consumed_prefix':core,'physical_adapter':physical,'stable_decision':decision})
    return session


def scope_owner(stack: Any, owned: dict, scope_getter: Any) -> Any:
    """生成と解放を同じ環境に限定し、環境を跨ぐ再利用は拒否する。"""
    selected,closed = [],[]
    def release(kind: Any, body: Any, trace: Any) -> bool:
        closed.append(True)
        changed = [name for name,(module,_) in owned.items() if sys.modules.get(name) is not module]
        for name,(module,_) in owned.items():
            if sys.modules.get(name) is module:
                sys.modules.pop(name)
        if changed and body is None:
            raise ValueError('second_prefix_cleanup_foreign_alias:'+','.join(changed))
        return False
    def claim() -> None:
        if closed:
            raise ValueError('second_prefix_scope_closed')
        scope = stack if scope_getter is None else scope_getter()
        if selected and selected[0] is not scope:
            raise ValueError('second_prefix_scope_changed')
        if not selected:
            scope.push(release)
            selected.append(scope)
    return claim


def install_load(stack: Any, bootstrap: Any, replace: Any, scope_getter: Any = None) -> None:
    original,owned,selected = bootstrap.load,{},[]
    if any(name in sys.modules for name in NAMES):
        raise ValueError('second_prefix_initial_foreign_alias')
    claim = scope_owner(stack,owned,scope_getter)
    def load(alias: str, path: Any, injection: Any = None) -> Any:
        value = original(alias,path,injection)
        if Path(path).resolve()==SECOND:
            if not selected:
                before = value.session_class
                def factory(session: Any, arrival: Any) -> type:
                    claim()
                    module = dependencies(original,owned)
                    return module.wrap_factory(before)(session,arrival)
                selected.append((value,before,factory))
            target,before,following = selected[0]
            if value is not target or value.session_class not in (before,following):
                raise ValueError('second_prefix_session_factory_changed')
            if value.session_class is before:
                replace(stack,value,'session_class',following)
        return value
    replace(stack,bootstrap,'load',load)


def install(stack: Any, owner: Any, replace: Any, scope_getter: Any = None) -> None:
    original,installed = owner.bootstrap,[]
    def bootstrap() -> Any:
        value = original()
        if not installed:
            install_load(stack,value,replace,scope_getter)
            installed.append(value)
        elif installed[0] is not value:
            raise ValueError('second_prefix_bootstrap_owner_changed')
        return value
    replace(stack,owner,'bootstrap',bootstrap)
