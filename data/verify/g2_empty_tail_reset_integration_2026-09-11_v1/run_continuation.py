"""原76ループへ、prefix検収後のreset継続だけを追加する限定CPU入口。"""
from __future__ import annotations
import ast
from contextlib import ExitStack
import inspect
from types import FunctionType, SimpleNamespace as N
from typing import Any
import continuation as C
import post_inputs as I
import run_qualification as Q

OLD_CHECKED, OLD_LOADER = Q.checked_loader, Q.wrap_loader


def inputs(original: Any) -> Any:
    prefix = original.FRAMES
    def adapted(*args: Any) -> Any:
        old = original.adapted(*args)
        def install(q: Any, stack: Any, *rest: Any) -> Any:
            owned = stack.enter_context(ExitStack())
            Q.KEPT['prefix_input_stack'] = owned
            return old.install(q, owned, *rest)
        return N(**(vars(old) | dict(FRAMES=prefix+I.POST, PREFIX_FRAMES=prefix, install=install)))
    return N(**(vars(original) | dict(FRAMES=prefix+I.POST, adapted=adapted)))


def wrap_loader(original: Any) -> Any:
    previous = OLD_LOADER(original)
    def load(alias: str, path: Any, stack: Any) -> Any:
        value = previous(alias, path, stack)
        if alias == '_empty_fused_checks':
            return N(**(vars(value) | dict(I=inputs(value.I))))
        return value
    return load


def drive(original: Any, namespace: Any) -> Any:
    tree = ast.parse(inspect.getsource(original))
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and ast.unparse(node.iter) == 'I.FRAMES':
            node.iter = ast.parse('I.PREFIX_FRAMES', mode='eval').body
            changed += 1
        if isinstance(node, ast.With):
            for index, item in enumerate(list(node.body)):
                if isinstance(item, ast.Assign) and ast.unparse(item).startswith('result = verify('):
                    node.body.insert(index+1, ast.parse('empty_continue(locals(), result)').body[0])
                    changed += 1
    assert changed == 2, 'original_drive_anchors'
    values = dict(namespace, empty_continue=C.extend)
    exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), values)
    return values['drive']


def checked_loader(selected: Any, base: Any) -> Any:
    previous = OLD_CHECKED(selected, base)
    def load(alias: str, path: Any, stack: Any) -> Any:
        value = previous(alias, path, stack)
        if alias == '_tail_suffix_shared_entry':
            prefix = value.C.R.H.TAIL.R
            original = prefix.drive
            assert not hasattr(prefix, 'empty_continue')
            stack.callback(setattr, prefix, 'drive', original)
            stack.callback(delattr, prefix, 'empty_continue')
            prefix.empty_continue = C.extend
            prefix.drive = drive(original, vars(prefix))
        return value
    return load


if __name__ == '__main__':
    Q.wrap_loader, Q.checked_loader = wrap_loader, checked_loader
    raise SystemExit(FunctionType(Q.F.main.__code__, dict(vars(Q.F), ROOT=Q.ROOT, execute=Q.execute))())
