"""同call実FIFO消費とinfer/着地vote色を結合する私有・既定OFF変換。"""
from __future__ import annotations

import ast
import functools
import hashlib
import marshal
from pathlib import Path
from types import FunctionType
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
PENDING = PROJECT / 'scripts/diagnose_video38_c6_pending_commit_shadow_v1.py'
SHADOW = ROOT.parent / 'g2_resolved_grace_guard_shadow_2026-09-08_v1/shadow.py'
OWN = ('adapter.py', 'test_binding.py', 'run_cpu.py', 'PLAN.md')
COMMITTED = "'committed' in locals()"
DIAG_SOURCE = 'fifo_committed_same_call'


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def located(source: str, reference: ast.AST) -> ast.stmt:
    node = ast.parse(source).body[0]
    for child in ast.walk(node):
        if 'lineno' in child._attributes:
            ast.copy_location(child, reference)
    return node


class Binder(ast.NodeTransformer):
    def __init__(self) -> None:
        self.diag_count = self.grace_count = 0

    def visit_Assign(self, node: ast.Assign) -> Any:
        names = [ast.unparse(target) for target in node.targets]
        if names == ['_landing_diag'] and isinstance(node.value, ast.Dict):
            self.diag_count += 1
            require([ast.literal_eval(key) for key in node.value.keys] ==
                ['falling_pair_old', 'falling_pair_new', 'source'], 'unknown_diag_schema')
            field = ast.parse('list(committed) if ' + COMMITTED + ' else None', mode='eval').body
            node.value.keys.append(ast.copy_location(ast.Constant('fifo_committed_pair'), node))
            node.value.values.append(ast.copy_location(field, node))
            source = f"if {COMMITTED}:\n    falling_pair = committed\n    _diag_source = '{DIAG_SOURCE}'"
            return [located(source, node), node]
        if names == ['(_, falling_pair_for_grace)'] and ast.unparse(node.value) == 'landing_pending':
            self.grace_count += 1
            return [node, located(f'if {COMMITTED}:\n    falling_pair_for_grace = committed', node)]
        return node


def add_binding(tree: ast.Module) -> ast.Module:
    """既存消費bodyは変えず、一意の推論/vote色入口だけへ同call値を渡す。"""
    pops = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and [ast.unparse(target) for target in node.targets] == ['committed']]
    require(len(pops) == 1 and ast.unparse(pops[0].value) == 'pending.popleft()', 'unique_real_fifo_pop')
    bind = Binder()
    result = bind.visit(tree)
    require((bind.diag_count, bind.grace_count) == (1, 1), 'unique_binding_sites')
    return result


class AstView:
    def __getattr__(self, name: str) -> Any:
        return getattr(ast, name)

    def fix_missing_locations(self, tree: ast.Module) -> ast.Module:
        return ast.fix_missing_locations(add_binding(tree))


def builder(original: Any, t2: Any) -> Any:
    """T2適用済み実code/globals/closureを保持し、原AST確定境界だけを私有化。"""
    require(Path(original.__code__.co_filename).resolve() == PENDING.resolve(), 'builder_source')
    require(hasattr(original, '__wrapped__'), 't2_builder_required')
    expected = t2.builder(original.__wrapped__)
    require(marshal.dumps(original.__code__) == marshal.dumps(expected.__code__), 'unknown_builder_code')
    require(original.__globals__.get('__t2_guard_tree') is t2.add_guard, 'unknown_t2_guard')
    namespace = dict(original.__globals__, ast=AstView())
    result = FunctionType(original.__code__, namespace, original.__name__,
        original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    return functools.update_wrapper(result, original)


def install(stack: Any, runtime: Any, t2: Any, *, enabled: bool = False) -> None:
    """T2直後・原runtime生成前に接続し、終了時はloader/builderを全復元する。"""
    require(type(enabled) is bool, 'enabled_type')
    if not enabled:
        return
    t2.guards()
    pending, prior, base = runtime.pending, runtime.A.prior, runtime.base
    base.patch(stack, pending, '_build_transformed_step', builder(pending._build_transformed_step, t2))
    old_load = prior.load

    def loaded(name: str, path: Path) -> Any:
        module = old_load(name, path)
        if path.resolve() != SHADOW.resolve():
            return module
        validate = module.validate_runtime

        @functools.wraps(validate)
        def checked(current: Any) -> Any:
            tree, first, mode = validate(current)
            require(mode == 'c6_pending', 'requires_existing_c6_t2')
            return add_binding(tree), first, mode
        base.patch(stack, module, 'validate_runtime', checked)
        return module
    base.patch(stack, prior, 'load', loaded)
