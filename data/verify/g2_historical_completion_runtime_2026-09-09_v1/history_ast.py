"""T適用済み原stepの会計・補助infer・graceを限定分岐する。状態所有はしない。"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
SOURCE = PROJECT / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/src/recognition_pipeline.py'
TRANSACTION = ROOT.parent / 'g2_normal_completion_transaction_2026-09-09_v1/transaction.py'
JOURNAL = ROOT.parent / 'g2_atomic_journal_capture_2026-09-09_v1/observer.py'
FIXED = {SOURCE: '6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02',
    TRANSACTION: 'a1d67b6efcd21830109470a95da725a26beb21ddf5c5b6451b598a43b7dac553',
    JOURNAL: 'b0d956a4fef51846a7baa2cee95a3679887bd2e3e9e620462e8375dc238d830c'}
POP = 'committed = pending.popleft()'
ROUTER = '__normal_completion'
METHODS = ('legacy_accounting', 'side_effect_gate')
SECONDARY = ('prev_state != BoardState.TSUMO_FALL and prev_state != BoardState.CHAIN '
    'and prev_state != BoardState.OJAMA_FALL and ctx.state == BoardState.STABLE '
    'and prev_confirmed is not None and ctx.confirmed_board is not None')
GRACE = 'landing_pending is not None and landing_pending[0] == frame_idx and ctx.confirmed_board is not None'


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guards() -> dict[str, str]:
    require(all(sha(path) == expected for path, expected in FIXED.items()), 'history_fixed_source')
    return {str(path): expected for path, expected in FIXED.items()} | {str(Path(__file__)): sha(Path(__file__))}


def key(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def expression(source: str) -> ast.expr:
    return ast.parse(source, mode='eval').body


def unique(tree: ast.AST, cls: type, predicate: Any, label: str) -> Any:
    values = [node for node in ast.walk(tree) if isinstance(node, cls) and predicate(node)]
    require(len(values) == 1, 'history_unique_' + label)
    return values[0]


def consume(tree: ast.AST) -> ast.If:
    pop = unique(tree, ast.Assign, lambda node: ast.unparse(node) == POP, 'pop')
    parent = unique(tree, ast.If, lambda node: pop in node.body, 'pop_parent')
    require(ast.unparse(parent.test) == 'pending' and parent.body[0] is pop
        and not parent.orelse, 'history_pending_body')
    return parent


def condition(tree: ast.AST, source: str, name: str) -> ast.If:
    expected = key(expression(source))
    return unique(tree, ast.If, lambda node: key(node.test) == expected, name)


def preflight(tree: ast.Module) -> tuple[ast.If, ast.If, ast.If]:
    require(type(tree) is ast.Module, 'history_module_type')
    guards()
    step = unique(tree, ast.FunctionDef, lambda node: node.name == '_step_side', 'step_name')
    calls = [node for node in ast.walk(step) if isinstance(node, ast.Call)]
    require(not any(ast.unparse(call.func) == ROUTER + '.' + name
        for call in calls for name in METHODS), 'history_already_applied')
    for name in ('update', 'infer'):
        require(sum(ast.unparse(call.func) == ROUTER + '.' + name for call in calls) == 1,
            'history_transaction_' + name)
    gates = [call for call in calls if ast.unparse(call.func) == ROUTER + '.gate']
    base = 'prev_state == BoardState.TSUMO_FALL and ctx.state == BoardState.STABLE'
    expected = [expression(ROUTER + ".gate('consume', " + base + ')'), expression(ROUTER
        + ".gate('infer', " + base + ' and prev_confirmed is not None and not _skip_infer_by_ojama_guard)')]
    require(sorted(map(key, gates)) == sorted(map(key, expected)), 'history_transaction_gates')
    original = ast.parse(SOURCE.read_bytes())
    selected = consume(tree), condition(tree, SECONDARY, 'secondary'), condition(tree, GRACE, 'grace')
    require([key(n) for n in selected[0].body] == [key(n) for n in consume(original).body],
        'history_accounting_body_changed')
    return selected


def call(name: str, arguments: list[ast.expr], reference: ast.AST) -> ast.Call:
    result = ast.Call(ast.Attribute(ast.Name(ROUTER, ast.Load()), name, ast.Load()), arguments, [])
    # 新規の外殻だけ位置を付け、元条件・元bodyの行は書き換えない。
    for node in (result, result.func, result.func.value):
        ast.copy_location(node, reference)
    return result


def add_history(tree: ast.Module) -> ast.Module:
    """一意性を先に検査し、入力ASTを非変異で3箇所だけ変換する。"""
    preflight(tree)
    result = copy.deepcopy(tree)
    accounting, secondary, grace = consume(result), condition(result, SECONDARY, 'secondary'), condition(result, GRACE, 'grace')
    tail = accounting.body[1:]
    wrapped = ast.copy_location(ast.If(call('legacy_accounting', [], tail[0]), tail, []), tail[0])
    wrapped.end_lineno, wrapped.end_col_offset = tail[-1].end_lineno, tail[-1].end_col_offset
    accounting.body = [accounting.body[0], wrapped]
    for name, node in (('secondary_infer', secondary), ('grace', grace)):
        node.test = call('side_effect_gate', [ast.copy_location(ast.Constant(name), node.test), node.test], node.test)
    return ast.fix_missing_locations(result)
