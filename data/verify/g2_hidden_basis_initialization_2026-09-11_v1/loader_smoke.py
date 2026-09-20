"""既存4更新collectorを再用し、原factory生成後の新ローダーだけを検査する。"""
from __future__ import annotations
import ast
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any

ROOT = Path(__file__).resolve().parent
QROOT = ROOT.parent / 'g2_reset_inflight_quarantine_2026-09-11_v1'
SOURCE = ROOT.parent / 'g2_empty_tail_reset_live_adapter_2026-09-11_v1/attach_cpu.py'
sys.path.insert(0, str(QROOT))
import hidden_loader as L


def connect(stack: Any, factory: Any, pipe: Any, state: Any) -> None:
    parts = L.modules()  # この時点で原factory/pipeは生成済み。
    guard = state['repeat_scope_guard']
    assert guard.factory is factory and guard.pipe is pipe
    recovery = N(factory=factory, pipe=pipe, evidence=guard.reset_lease.evidence)
    observer = parts.actual.Observer(recovery, 0, 14)
    connection = parts.binding.Connection(recovery, observer, 30, io.StringIO())
    assert type(observer.gate) is parts.actual.G.SettledBasisGate
    assert parts.binding.OLD.A is parts.actual
    assert parts.mode.B is parts.binding.B is parts.belief
    assert observer.gate.candidate is None and connection.binding is None
    with (state['output'] / 'HIDDEN_LOADER_SMOKE.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(actual_factory=True, reset_executed=False, basis_issued=False,
            observer_module=type(observer).__module__, gate_module=type(observer.gate).__module__,
            binding_module=type(connection).__module__, same_core_objects=True,
            artificial_constructor_clocks=True, target_recovery_verified=False,
            quality_gate_clear=False), stream, indent=2)


def main() -> None:
    tree = ast.parse(SOURCE.read_bytes())
    imports = sites = outputs = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == 'adapter_v2':
            node.names[0].name = 'runtime_adapter'
            imports += 1
        if isinstance(node, ast.FunctionDef) and node.name == 'drive':
            node.body[1:1] = ast.parse("assert 'conditional_runtime_factory' not in state\nstate['conditional_runtime_factory']=factory").body
        if isinstance(node, ast.With):
            for index, item in enumerate(node.body):
                if ast.unparse(item).startswith('collector.collect_lean('):
                    node.body.insert(index + 1, ast.parse('CONNECT(stack,factory,pipe,state)').body[0])
                    sites += 1
                    break
        if isinstance(node, ast.Assign) and ast.unparse(node).startswith('output = A.ROOT'):
            node.value = ast.parse('OUTPUT / sys.argv[1]', mode='eval').body
            outputs += 1
    assert (imports, sites, outputs) == (1, 1, 1)
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE), 'exec'),
         dict(__name__='__main__', __file__=str(SOURCE), CONNECT=connect, OUTPUT=ROOT))


if __name__ == '__main__':
    main()
