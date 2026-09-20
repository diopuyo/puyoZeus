"""返値無視の原caller契約と最終code欠落拒否のCPU境界。"""
from __future__ import annotations
import ast
from copy import deepcopy
from types import SimpleNamespace
import pytest
import ast_connection as A


def transform(tree: ast.Module) -> ast.Module:
    result = deepcopy(tree)
    result.body.append(ast.parse('second = 2').body[0])
    return result


def test_ignored_return_receives_full_transform() -> None:
    tree = ast.parse('first = 1')
    old = ast.dump(tree)
    transformed = transform(tree)
    assert ast.dump(tree) == old and transformed is not tree
    returned = A.apply_to_original(transform, tree)
    assert returned is tree and ast.dump(tree) == ast.dump(transformed)
    scope = {}
    exec(compile(tree, '<artificial-module>', 'exec'), scope)
    assert scope['first'] == 1 and scope['second'] == 2


@pytest.mark.parametrize('missing', sorted(A.REQUIRED))
def test_final_code_rejects_each_missing_method(missing: str) -> None:
    item = SimpleNamespace(__code__=SimpleNamespace(co_names=tuple(A.REQUIRED-{missing})))
    with pytest.raises(RuntimeError, match='final_code_missing'):
        A.validate(item)


def test_complete_method_set() -> None:
    A.validate(SimpleNamespace(__code__=SimpleNamespace(co_names=tuple(A.REQUIRED))))


def test_non_module_rejected() -> None:
    with pytest.raises(RuntimeError, match='connection_module'):
        A.apply_to_original(transform, ast.parse('1', mode='eval'))


def test_bad_result_rejected_before_transfer() -> None:
    tree = ast.parse('first = 1')
    before = ast.dump(tree)
    with pytest.raises(RuntimeError, match='connection_result'):
        A.apply_to_original(lambda value: None, tree)
    assert ast.dump(tree) == before
