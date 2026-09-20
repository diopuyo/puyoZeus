"""非変異ASTの返値を原Moduleへ反映し、最終二経路の欠落を拒否する。"""
from __future__ import annotations
import ast
from types import SimpleNamespace
from typing import Any

REQUIRED = frozenset(('legacy_accounting', 'side_effect_gate', 'current_counter',
    'current_recovered', 'gate', 'infer', 'update'))


def require(value: bool, reason: str) -> None:
    if not value:
        raise RuntimeError(reason)


def apply_to_original(original: Any, tree: ast.Module) -> ast.Module:
    require(type(tree) is ast.Module, 'history_connection_module')
    result = original(tree)
    require(type(result) is ast.Module, 'history_connection_result')
    # 元codegenはfix_missing_locationsの返値を使わず、渡したModuleをcompileする。
    for name in tree._fields:
        setattr(tree, name, getattr(result, name))
    return tree


def validate(function: Any) -> None:
    require(REQUIRED <= set(function.__code__.co_names), 'history_connection_final_code_missing')


def checked_install(original: Any, core: Any, stack: Any, runtime: Any, router: Any) -> None:
    original(stack, runtime, router)
    pending, prior = runtime.pending, runtime.A.prior
    builder, loader = pending._build_transformed_step, prior.load
    def build(*args: Any, **kwargs: Any) -> Any:
        result, receipt = builder(*args, **kwargs)
        validate(result)
        return result, receipt
    def load(name: str, path: Any) -> Any:
        module = loader(name, path)
        if path.resolve() == core.G.GRACE.resolve():
            compiled = module.compile_step
            def compile_step(*args: Any, **kwargs: Any) -> Any:
                result = compiled(*args, **kwargs)
                validate(result)
                return result
            core.G.patch(stack, module, 'compile_step', compile_step)
        return module
    core.G.patch(stack, pending, '_build_transformed_step', build)
    core.G.patch(stack, prior, 'load', load)


def configure(stack: Any, assembly: Any, core: Any) -> None:
    original = assembly.transforms
    def transforms(c: Any, h: Any, inner: Any) -> None:
        proxy = SimpleNamespace(guards=h.guards,
            add_history=lambda tree: apply_to_original(h.add_history, tree))
        original(c, proxy, inner)
        installer = c.install_transform
        def install(target: Any, runtime: Any, router: Any) -> None:
            checked_install(installer, c, target, runtime, router)
        c.G.patch(inner, c, 'install_transform', install)
    core.G.patch(stack, assembly, 'transforms', transforms)
