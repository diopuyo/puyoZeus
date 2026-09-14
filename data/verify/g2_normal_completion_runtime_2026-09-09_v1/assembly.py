"""検収済みFIFO/palette構成へ通常完了transactionを私有合成する。既定OFF。"""
from __future__ import annotations

import ast
from contextlib import ExitStack, contextmanager
import functools
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
COMPOSITION = ROOT.parent / 'g2_fifo_palette_composition_2026-09-09_v1'
TRANSACTION = ROOT.parent / 'g2_normal_completion_transaction_2026-09-09_v1'
FIXED = {COMPOSITION / 'adapter.py': '9c5cbf9fa5970f792f4b865e1b8248dc7eba8cef73f1c6f4fbea62e370100aed',
    COMPOSITION / 'codegen.py': 'a8d8ee6ec6021dbc5a978ecba27402e17db8473d4bff412017040ef08c2099cc',
    TRANSACTION / 'transaction.py': 'a1d67b6efcd21830109470a95da725a26beb21ddf5c5b6451b598a43b7dac553',
    TRANSACTION / 'live_provider.py': '5d0b400d965700943887b2bb833393689c84a46aa2a5d302e1ff7f6e600dca6f'}
OWN = ('assembly.py', 'reporting.py', 'test_assembly.py', 'test_reporting.py', 'run_cpu.py', 'PLAN.md')


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(alias: str, path: Path) -> Any:
    require(path in FIXED and sha(path) == FIXED[path], 'normal_dependency_changed')
    require(alias not in sys.modules, 'normal_alias_collision')
    spec = importlib.util.spec_from_file_location(alias, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[alias] = value
    spec.loader.exec_module(value)
    return value


def modules() -> tuple[Any, Any, Any]:
    sys.path[:0] = [str(COMPOSITION), str(TRANSACTION)]
    t = load('transaction', TRANSACTION / 'transaction.py')
    provider = load('_normal_completion_provider', TRANSACTION / 'live_provider.py')
    child = load('_normal_completion_fifo_composition', COMPOSITION / 'adapter.py')
    return child, t, provider


A, T, V = modules()
import reporting as R
G = A.G
BASE_DERIVE = G.derive


def guards() -> dict[str, str]:
    require(all(sha(p) == h for p, h in FIXED.items()), 'normal_dependency_guard')
    return A.guards() | {str(p): h for p, h in FIXED.items()} | {str(ROOT / n): sha(ROOT / n) for n in OWN}


class View:
    def __init__(self, previous: Any) -> None:
        self.previous = previous

    def __getattr__(self, name: str) -> Any:
        return getattr(self.previous, name)

    def fix_missing_locations(self, tree: Any) -> Any:
        return ast.fix_missing_locations(T.add_transaction(self.previous.fix_missing_locations(tree)))


class Router:
    def __init__(self) -> None:
        self.controller: Any = None

    def __getattr__(self, name: str) -> Any:
        require(self.controller is not None, 'completion_provider_not_attached')
        return getattr(self.controller, name)


class AfterGraceView:
    def __getattr__(self, name: str) -> Any:
        return getattr(ast, name)

    def fix_missing_locations(self, tree: Any) -> Any:
        return ast.fix_missing_locations(T.add_transaction(tree))


def transformed_builder(original: Any, router: Router) -> Any:
    fixed = G.module(G.FIFO)
    old_ast = original.__globals__['ast']
    require(type(old_ast).fix_missing_locations.__code__ == fixed.AstView.fix_missing_locations.__code__,
        'completion_requires_fixed_fifo_ast')
    require(hasattr(original, '__wrapped__') and original.__code__ is original.__wrapped__.__code__,
        'completion_fifo_builder_code')
    cloned = FunctionType(original.__code__, dict(original.__globals__, ast=View(old_ast)),
        original.__name__, original.__defaults__, original.__closure__)
    cloned.__kwdefaults__ = original.__kwdefaults__
    @functools.wraps(original)
    def build(current: Any, hook: Any) -> Any:
        result, receipt = cloned(current, hook)
        require(result.__globals__.get('__normal_completion', router) is router, 'completion_global_collision')
        result.__globals__['__normal_completion'] = router
        return result, receipt
    return build


def install_transform(stack: Any, runtime: Any, router: Router) -> None:
    pending, prior = runtime.pending, runtime.A.prior
    G.patch(stack, pending, '_build_transformed_step', transformed_builder(pending._build_transformed_step, router))
    old_load = prior.load
    def loaded(name: str, path: Path) -> Any:
        module = old_load(name, path)
        if path.resolve() != G.GRACE.resolve():
            return module
        original = module.compile_step
        require(Path(original.__code__.co_filename).resolve() == G.GRACE.resolve(), 'original_grace_compile')
        # 原graceは元sm.update ASTを一意検査する。その変換完了後にだけ新update入口を載せる。
        compiled = FunctionType(original.__code__, dict(original.__globals__, ast=AfterGraceView()),
            original.__name__, original.__defaults__, original.__closure__)
        compiled.__kwdefaults__ = original.__kwdefaults__
        G.patch(stack, module, 'compile_step', functools.update_wrapper(compiled, original))
        return module
    G.patch(stack, prior, 'load', loaded)


def derive() -> dict[str, Any]:
    old = BASE_DERIVE()
    t2, fifo = G.module(G.T2), G.module(G.FIFO)
    def install(stack: Any, runtime: Any, t2: Any, *, enabled: bool = False) -> None:
        fifo.install(stack, runtime, t2, enabled=enabled)
        if enabled:
            install_transform(stack, runtime, Router())
    on = G.derive_lane(t2, SimpleNamespace(install=install), True)
    require(all(old['on'][name]['bytecode_sha256'] != on[name]['bytecode_sha256']
        for name in ('pending', 'resolved')), 'completion_transform_not_applied')
    return old | {'schema': 'normal-completion-code-derivation/v1', 'off': old['on'], 'on': on,
        'transaction_source_sha256': {str(p): h for p, h in FIXED.items()}}


@contextmanager
def generated() -> Any:
    with ExitStack() as stack:
        G.patch(stack, G, 'derive', derive)
        yield


def compose(latest: Any, *, enabled: bool = False) -> tuple[Any, Any, Any, Any]:
    require(type(enabled) is bool, 'normal_enabled_type')
    if not enabled:
        return A.compose(latest)
    with generated():
        prior, driver, addon, engine = A.compose(latest, enabled=True)
    old_install = addon.install
    router = Router()
    def install(stack: Any, collector: Any, history: Any, state: Any) -> None:
        old_install(stack, collector, history, state)
        provider = V.Provider(state['atomic_journal_observer'])
        # 固定helperの純粋合法配置APIだけを使う。既存dataロードは呼ばない。
        path = ROOT.parent / 'g2_inventory_producer_diagnosis_2026-09-09_v1/probe.py'
        require(sha(path) == 'f3ea5e870d344436739e3ea706b39fcec58112d8fe79b1cc08c0739e4e1dc244', 'legal_source')
        spec = importlib.util.spec_from_file_location('_normal_completion_legal', path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        require(router.controller is None, 'completion_repeated_install')
        router.controller = T.Controller(provider, helper.placement_matches, enabled=True)
        provider.attach(stack, router.controller)
        state['normal_completion_controller'] = router.controller
        R.install(stack, state)
    addon.install, addon.normal_router = install, router
    finish, verify, old_guards = addon.finish, addon.verify, addon.guards
    def finished(state: Any) -> None:
        R.finish(state)
        with generated():
            finish(state)
    def verified(output: Path) -> None:
        R.verify(output)
        with generated():
            verify(output)
    addon.finish, addon.verify = finished, verified
    addon.REQUIRED = addon.REQUIRED | R.REQUIRED
    addon.guards = lambda: old_guards() | guards()
    return prior, driver, addon, engine


@contextmanager
def configured(latest: Any, runtime: Any, finisher: Any, driver: Any,
               addon: Any, engine: Any, *, enabled: bool = False) -> Any:
    require(type(enabled) is bool, 'normal_enabled_type')
    if not enabled:
        with A.configured(latest, runtime, finisher, driver, addon, engine):
            yield
        return
    with generated(), A.configured(latest, runtime, finisher, driver, addon, engine, enabled=True):
        t2 = latest.B.B.B.T
        original = t2.install
        def installed(stack: Any, target: Any, *, enabled: bool = False) -> None:
            require(enabled is True, 'completion_requires_t2_fifo')
            original(stack, target, enabled=enabled)
            install_transform(stack, target, addon.normal_router)
        with ExitStack() as stack:
            G.patch(stack, t2, 'install', installed)
            yield
