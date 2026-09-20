"""v35専用finishのarchive検査先だけを寿命保持verifierへ差替える。"""
from __future__ import annotations
import ast
import inspect
import textwrap
from types import FunctionType
from typing import Any
import probabilistic_finish as OLD
import archive_lifetime as L


def retained(state: Any, lease: Any) -> None:
    anchor = OLD.read(state['output'], 'PROBABILISTIC_FACTORY_ANCHOR.json')
    OLD.require(anchor['old_archive_verified'] is True and anchor['archive_id'] == id(lease.archive), 'archive_anchor')
    state[L.KEY].verify(lease.archive)


def live(kept: Any) -> Any:
    tree = ast.parse(textwrap.dedent(inspect.getsource(OLD.live)))
    hits = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and ast.unparse(node.value) == 'lease.archive.verify()':
            node.value = ast.parse('retained(state, lease)', mode='eval').body
            hits += 1
    assert hits == 1, 'archive_finish_source_changed'
    namespace = dict(OLD.live.__globals__, retained=retained)
    exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), namespace)
    return namespace['live'](kept)


wrap = FunctionType(OLD.wrap.__code__, dict(vars(OLD), live=live))
