"""元paletteのcodeは不変。実legacy fixtureのロード後だけ入力を包む。"""
from __future__ import annotations
from contextlib import contextmanager
import importlib.util
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import tracker_input as T

ROOT = Path(__file__).resolve().parent.parent
COMMON = ROOT / 'g2_history_publication_probe_runtime_2026-09-10_v13/common.py'
SOURCE = ROOT / 'g2_chain_firing_continuous_cpu_2026-09-10_v1/constructor_input.py'


def patch(stack: Any, obj: Any, name: str, value: Any) -> None:
    old = getattr(obj, name)
    stack.callback(setattr, obj, name, old)
    setattr(obj, name, value)


def transport(original: Any, kwargs: Any, evidence: Any) -> Any:
    spec = importlib.util.spec_from_file_location('_g2_cascade_original_constructor', SOURCE)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    combined = T.transport(source.transport(original, kwargs, evidence), evidence)
    def real(frozen: Any, monkeypatch: Any) -> Any:
        with contextmanager(combined)(frozen, monkeypatch) as value:
            evidence['actual_constructor_seen'] = T.observe(value[0])
            if evidence.get('constructor_only_probe'):
                raise RuntimeError('cascade_constructor_probe_requested_stop')
            yield value
    return real


def attach_parent(parent: Any, stack: Any, kwargs: Any, evidence: Any) -> None:
    previous, calls = parent.load_previous, []
    def load_previous() -> Any:
        assert not calls, 'cascade_parent_loaded_twice'
        calls.append(True)
        latest, active = previous()
        original = active.P.load
        def load(alias: str, path: Any) -> Any:
            module = original(alias, path)
            if alias == '_normal_active_legacy_fixture':
                old = module.real
                patch(stack, module, 'real', N(__wrapped__=transport(old.__wrapped__, kwargs, evidence)))
                evidence['legacy_input_attached'] = True
            return module
        patch(stack, active.P, 'load', load)
        return latest, active
    patch(stack, parent, 'load_previous', load_previous)


def attach_dependencies(deps: Any, stack: Any, kwargs: Any, evidence: Any) -> None:
    original = deps.load
    def load(name: str, inner: Any) -> Any:
        module = original(name, inner)
        if name == '_history_latest_cpu':
            attach_parent(module, inner, kwargs, evidence)
        return module
    patch(stack, deps, 'load', load)


def installed(stack: Any, common: Any, evidence: Any) -> None:
    evidence['actual_common'] = str(Path(common.__file__).resolve())
    assert Path(common.__file__).resolve() == COMMON, 'cascade_common_source'
    plan = ROOT / 'video38_history_publication_probe_live_2026-09-10_v6/PLAN.json'
    names = ('enable_chain_formula_read_verify', 'enable_formula_chain_count_update', 'enable_formula_step_interlude')
    kwargs = {name: common.read(plan)['actual_collector_kwargs'][name] for name in names}
    assert all(type(value) is bool and value for value in kwargs.values())
    evidence.update(plan=str(plan), sha256=common.sha(plan), supplied=kwargs, all_live_flags_equal=False)
    original = common.load
    def load(alias: str, path: Any, inner: Any) -> Any:
        module = original(alias, path, inner)
        if path == common.CURRENT / 'run_cpu.py':
            previous = module.load
            def member(name: str, owned: Any) -> Any:
                value = previous(name, owned)
                if name == 'deps_full': attach_dependencies(value, owned, kwargs, evidence)
                return value
            patch(inner, module, 'load', member)
        return module
    patch(stack, common, 'load', load)


def wrap(previous: Any, evidence: Any) -> Any:
    def load(alias: str, path: Any, stack: Any) -> Any:
        module = previous(alias, path, stack)
        if alias == '_tail_suffix_shared_entry':
            installed(stack, module.C.R.H.TAIL.R.K, evidence)
        return module
    return load
