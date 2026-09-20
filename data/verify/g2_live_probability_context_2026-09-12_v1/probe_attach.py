"""旧原4更新の非reset検査を再用する。新Contextのperform到達とは区別する。"""
from __future__ import annotations
import ast
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent


def main() -> None:
    source = ROOT.parent / 'g2_empty_tail_reset_live_adapter_2026-09-11_v1/attach_cpu.py'
    tree = ast.parse(source.read_bytes())
    imports, contexts, attachments = 0, 0, 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == 'adapter_v2':
            node.names[0].name = 'live_adapter'
            imports += 1
        if isinstance(node, ast.FunctionDef) and node.name == 'drive':
            node.body.insert(1, ast.parse("assert 'conditional_runtime_factory' not in state").body[0])
            node.body.insert(2, ast.parse("state['conditional_runtime_factory'] = factory").body[0])
            contexts += 1
        if isinstance(node, ast.With):
            for index, statement in enumerate(node.body):
                call = statement.value if isinstance(statement, ast.Expr) else None
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Name) and call.func.value.id == 'hook'
                        and call.func.attr == 'attach'):
                    checks = ast.parse("assert type(state['live_empty_reset_context']) is state['live_probability_context_class']\n"
                        "assert any(c.__module__ == '_g2_live_probability_context_core' for c in state['live_probability_context_class'].__mro__)\n"
                        "assert 'live_probability_context' not in state").body
                    node.body[index + 1:index + 1] = checks
                    attachments += 1
                    break
    assert (imports, contexts, attachments) == (1, 1, 1) and not torch.cuda.is_initialized()
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'),
         dict(__name__='__main__', __file__=str(source)))


if __name__ == '__main__':
    main()
