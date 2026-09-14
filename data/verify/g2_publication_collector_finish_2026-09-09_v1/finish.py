"""旧finishの単一prefix条件だけを、新しい限定採否検証に差し替える。"""
from __future__ import annotations
import ast
import contextlib
import hashlib
from pathlib import Path
from types import CodeType, FunctionType
from typing import Any
import proof as P

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / 'g2_split_runtime_evidence_adapter_2026-09-08_v1/runner.py'
RUNTIME_SHA = '78bdc5fdc58e2b2300ddff2215bc0276be4d810ce36bfa87aeff7505924fd93e'
PRIVATE = '_bounded_publication_prefix_validate'
EXPECTED = '''if not all(r["pre_mutation_prefix_bit_exact"] for r in report["streams"].values()):
    raise ValueError("pre_intervention_prefix_changed")'''


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def transformed_code() -> CodeType:
    require(sha(RUNTIME) == RUNTIME_SHA, 'fixed_runner_changed')
    tree = ast.parse(RUNTIME.read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'finish')
    expected = ast.dump(ast.parse(EXPECTED).body[0], include_attributes=False)
    found = [i for i, n in enumerate(node.body) if ast.dump(n, include_attributes=False) == expected]
    require(len(found) == 1, 'single_prefix_branch_required')
    replacement = ast.parse(PRIVATE + '(output, report, state)').body[0]
    node.body[found[0]] = ast.copy_location(replacement, node.body[found[0]])
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    code = compile(module, str(RUNTIME), 'exec')
    return next(item for item in code.co_consts if isinstance(item, CodeType) and item.co_name == 'finish')


def install(stack: contextlib.ExitStack, runtime: Any, *, enabled: bool = False) -> None:
    """driver wrapperより前に設置し、run/write/保存順序は元globalsで保つ。"""
    if not enabled:
        return
    require(type(enabled) is bool and Path(runtime.__file__).resolve() == RUNTIME, 'runtime_identity')
    require(not hasattr(runtime, PRIVATE), 'finish_reentry')
    code = compile(RUNTIME.read_bytes(), str(RUNTIME), 'exec')
    original_code = next(n for n in code.co_consts if isinstance(n, CodeType) and n.co_name == 'finish')
    original = runtime.finish
    require(type(original) is FunctionType and original.__code__ == original_code
        and original.__globals__ is vars(runtime), 'original_finish_identity')
    changed = FunctionType(transformed_code(), vars(runtime), original.__name__, original.__defaults__)
    def restore() -> None:
        runtime.finish = original
        delattr(runtime, PRIVATE)
    setattr(runtime, PRIVATE, P.validate)
    runtime.finish = changed
    stack.callback(restore)
