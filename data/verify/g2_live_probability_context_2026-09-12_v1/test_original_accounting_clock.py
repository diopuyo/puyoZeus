"""125の状態系列懸念を原collectorの代入順と実関数codeで確認する。"""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any

SOURCE = Path(__file__).resolve().parents[3] / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/scripts/collect_boards_lean.py'


def test_original_state_assignment_is_after_accounting() -> None:
    tree = ast.parse(SOURCE.read_bytes())
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collect_lean')
    loop = next(n for n in original.body if isinstance(n, ast.For) and ast.unparse(n.target) == 'local_i')
    index = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == '_drive_ojama_accounting_lean')
    call = loop.body[index].value
    assert [ast.unparse(arg) for arg in call.args[3:7]] == ['prev_bstate_p1', 'prev_bstate_p2', 'result.p1', 'result.p2']
    assert [ast.unparse(n) for n in loop.body[index + 1:index + 3]] == [
        'prev_bstate_p1 = result.p1.state', 'prev_bstate_p2 = result.p2.state']
    writes = [n for n in ast.walk(loop) if isinstance(n, ast.Assign)
              and any(ast.unparse(t) in ('prev_bstate_p1', 'prev_bstate_p2') for t in n.targets)]
    assert len(writes) == 2


def test_original_drive_passes_exact_current_and_previous_states() -> None:
    tree = ast.parse(SOURCE.read_bytes())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_drive_ojama_accounting_lean')
    module = ast.parse('from __future__ import annotations')
    module.body.append(function)
    seen: list[tuple] = []
    namespace: dict[str, Any] = {'_drain_ojama_by_tsumo_delta_lean': lambda *args: None}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)
    old1, old2, current1, current2 = (object() for _ in range(4))
    tracker = N(on_state_transition=lambda *args: seen.append(args), get_snapshot=lambda seconds: seconds)
    result = namespace['_drive_ojama_accounting_lean'](tracker, N(), N(), old1, old2,
        N(state=current1, score=6), N(state=current2, score=6), None, None, 1.0)
    assert result == 1.0
    assert seen == [('p1', old1, current1, 6, 1.0), ('p2', old2, current2, 6, 1.0)]
