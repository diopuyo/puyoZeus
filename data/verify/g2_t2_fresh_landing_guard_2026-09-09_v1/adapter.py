"""今回生成した着地連鎖だけをT2旧色復元から除外する私有変換。"""
from __future__ import annotations
import ast
import functools
import hashlib
import inspect
from pathlib import Path
import textwrap
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
PENDING = PROJECT / 'scripts/diagnose_video38_c6_pending_commit_shadow_v1.py'
SHADOW = ROOT.parent / 'g2_resolved_grace_guard_shadow_2026-09-08_v1/shadow.py'
FIXED = {PENDING: '563fb383e9c0ceb804c3950522a67c77460a69f5e483a0c09c9becdd2f6cbab0',
    SHADOW: 'f20028f16c980825ee435ff57df649d1527b740c04861de0e621d62011af7561'}
OWN = ('adapter.py', 'test_guard.py', 'run_cpu.py', 'PLAN.md')
OLD_TEST = ('prev_stable is not None and _effective_chain_event is None '
    'and not in_grace and not _deferred_committed_this_frame')
FRESH = ("prev_state == BoardState.TSUMO_FALL and 'pseudo' in locals() "
    "and pseudo is not None and side in ('1P', '2P') "
    "and pseudo is (self._active_chain_1p if side == '1P' else self._active_chain_2p) "
    'and pseudo.trigger_sec == time_sec and pseudo.mechanism == CHAIN_MECHANISM_LANDING')


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value: bool, reason: str) -> None:
    if not value:
        raise RuntimeError(reason)


def guards() -> dict[str, str]:
    require(all(sha(path) == digest for path, digest in FIXED.items()), 't2_fixed_source_changed')
    return {str(path): digest for path, digest in FIXED.items()} | {
        str(ROOT / name): sha(ROOT / name) for name in OWN}


def add_guard(tree: ast.Module) -> ast.Module:
    """元T2条件一箇所だけに制限を追加し、writerの位置と本体は保つ。"""
    pattern = ast.dump(ast.parse(OLD_TEST, mode='eval').body)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
               and ast.dump(node.test) == pattern]
    require(len(matches) == 1, 't2_unique_original_condition')
    node = matches[0]
    extra = ast.parse('not (' + FRESH + ')', mode='eval').body
    for item in ast.walk(extra):
        if 'lineno' in item._attributes:
            ast.copy_location(item, node.test)
    node.test = ast.copy_location(ast.BoolOp(op=ast.And(), values=[node.test, extra]), node.test)
    return tree


def builder(original: Any) -> Any:
    """固定既存builderを私有cloneし、C6変換後にT2を一度追加する。"""
    require(Path(original.__code__.co_filename).resolve() == PENDING.resolve(), 't2_builder_path')
    lines, first = inspect.getsourcelines(original)
    tree = ast.parse(textwrap.dedent(''.join(lines)))
    function = tree.body[0]
    matches = [index for index, node in enumerate(function.body) if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'ast.fix_missing_locations']
    require(len(matches) == 1, 't2_builder_insertion')
    index = matches[0]
    statement = ast.parse('__t2_guard_tree(tree)').body[0]
    function.body.insert(index, ast.copy_location(statement, function.body[index]))
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, first - 1)
    namespace = dict(original.__globals__, __t2_guard_tree=add_guard)
    exec(compile(tree, str(PENDING), 'exec'), namespace)
    return functools.update_wrapper(namespace[original.__name__], original)


def install(stack: Any, runtime: Any, *, enabled: bool = False) -> None:
    """split初期化後・観測器生成前だけに接続し、旧loaderを必ず呼ぶ。"""
    require(type(enabled) is bool, 't2_enabled_type')
    if not enabled:
        return
    guards()
    pending, prior, base = runtime.pending, runtime.A.prior, runtime.base
    original = pending._build_transformed_step
    require(not hasattr(original, '__wrapped__'), 't2_builder_already_wrapped')
    base.patch(stack, pending, '_build_transformed_step', builder(original))
    old_load = prior.load
    def loaded(name: str, path: Path) -> Any:
        module = old_load(name, path)
        if path.resolve() != SHADOW.resolve():
            return module
        validate = module.validate_runtime
        require(Path(validate.__code__.co_filename).resolve() == SHADOW.resolve(), 't2_validator_path')
        @functools.wraps(validate)
        def checked(current: Any) -> Any:
            tree, first, mode = validate(current)
            require(mode == 'c6_pending', 't2_requires_existing_c6')
            return add_guard(tree), first, mode
        base.patch(stack, module, 'validate_runtime', checked)
        return module
    base.patch(stack, prior, 'load', loaded)
