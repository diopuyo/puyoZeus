"""原normal生成/設置関数の私有globalsへhistoryを一度だけ供給する。"""
from __future__ import annotations

import ast
from contextlib import ExitStack, contextmanager
from types import FunctionType, SimpleNamespace
from typing import Any, Iterator
import deps_full as D


def clone(original: Any, **changes: Any) -> Any:
    value = FunctionType(original.__code__, dict(original.__globals__, **changes),
        original.__name__, original.__defaults__, original.__closure__)
    value.__kwdefaults__ = original.__kwdefaults__
    return value


def transforms(c: Any, h: Any, stack: Any) -> None:
    class View(c.View):
        def fix_missing_locations(self, tree: Any) -> Any:
            return h.add_history(super().fix_missing_locations(tree))
    class AfterGrace(c.AfterGraceView):
        def fix_missing_locations(self, tree: Any) -> Any:
            return h.add_history(super().fix_missing_locations(tree))
    builder = clone(c.transformed_builder, View=View)
    install = clone(c.install_transform, transformed_builder=builder, AfterGraceView=AfterGrace)
    original_derive = c.derive
    def derive() -> dict[str, Any]:
        return original_derive() | {'schema': 'directional-history-code-derivation/v1',
            'history_source_guards': h.guards() | D.guards()}
    c.G.patch(stack, c, 'install_transform', install)
    c.G.patch(stack, c, 'derive', derive)


class Factory:
    def __init__(self, c: Any, modules: Any) -> None:
        self.c, self.modules = c, modules
        self.captured: list[Any] | None = None
        self.types: Any = None
        self.provider: Any = None
        self.controller: Any = None

    def make_provider(self, journal: Any) -> Any:
        D.require(self.provider is None and self.captured is not None and len(self.captured) == 1, 'history_install_order')
        m, binding = self.modules, self.captured[0]
        o = binding['adapter'].__class__.__init__.__globals__
        import sys
        module = sys.modules.get(binding['adapter'].__class__.__module__)
        D.require(module is not None and vars(module) is o, 'history_occurrence_module')
        self.types = m['handoff'].make_types(m['history_controller'], m['history_provider'],
            m['history_state'], self.c.T, module)
        self.provider = self.types.Provider(journal, binding['adapter'])
        return self.provider

    def make_controller(self, provider: Any, legal: Any, *, enabled: bool = False) -> Any:
        D.require(self.controller is None and provider is self.provider, 'history_single_controller')
        self.controller = self.types.Controller(provider, legal, enabled=enabled,
            inventory=self.modules['history_dependencies'].P)
        return self.controller

    def compose(self, latest: Any) -> Any:
        # 元C.composeの原J/install/finishを保持し、二生成factoryだけを明示的に置換。
        provider = SimpleNamespace(Provider=self.make_provider)
        transaction = SimpleNamespace(Controller=self.make_controller)
        result = clone(self.c.compose, V=provider, T=transaction)(latest, enabled=True)
        addon, original = result[2], result[2].guards
        addon.guards = lambda: original() | D.guards()
        return result


@contextmanager
def directional(x: Any, runtime: Any, addon: Any, factory: Factory,
                source_id: str, run_id: str) -> Iterator[dict[str, Any]]:
    binding: dict[str, Any] = {}
    with ExitStack() as stack:
        loaded = {}
        for name in x.FIXED:
            loaded[name] = x.load(name)
            stack.callback(__import__('sys').modules.pop, name, None)
        o, p, r = (loaded[n] for n in ('_directional_occurrence', '_directional_provider', '_directional_runtime'))
        original = runtime.M.instrument
        def instrument(inner: Any, collector: Any, history: Any, receipt: Any, state: Any) -> None:
            with r.before_observation(o, p, source_id=source_id, run_id=run_id) as captured:
                factory.captured = captured
                original(inner, collector, history, receipt, state)
                r.bind_journal(state, captured[0], addon.journal)
                binding.update(captured[0])
                D.require(state['normal_completion_controller'] is factory.controller, 'actual_history_controller')
        factory.c.G.patch(stack, runtime.M, 'instrument', instrument)
        yield binding


@contextmanager
def session(c: Any, modules: Any) -> Iterator[Any]:
    with ExitStack() as stack, c.A.loaded_latest() as latest:
        transforms(c, modules['history_ast'], stack)
        factory = Factory(c, modules)
        prior, driver, addon, engine = factory.compose(latest)
        driver.validate_addon(addon)
        with latest.configuration(prior, driver):
            runtime, finisher = driver.load_runtime()
            with c.configured(latest, runtime, finisher, driver, addon, engine, enabled=True):
                yield runtime, addon, factory
