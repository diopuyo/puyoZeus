"""親providerの人工契約試験を排他保存し、終了と品質を区別する。"""
from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
OWN = ('stream.py', 'provider.py', 'dependencies.py', 'test_stream.py', 'test_provider.py', 'test_dependencies.py', 'run_cpu.py', 'PLAN.md')
MAX_FUNCTION_LINES = 50


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def style() -> list[dict[str, Any]]:
    result = []
    for name in OWN:
        if not name.endswith('.py'):
            continue
        tree = ast.parse((ROOT / name).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno + 1
                arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                if length > MAX_FUNCTION_LINES or node.returns is None:
                    raise ValueError(f'style:{name}:{node.name}')
                if any(arg.annotation is None and arg.arg not in ('self', 'cls') for arg in arguments):
                    raise ValueError(f'argument_hint:{name}:{node.name}')
                result.append({'file': name, 'function': node.name, 'lines': length})
    return result


def main() -> int:
    output = ROOT / sys.argv[1]
    if output.parent != ROOT or not output.name.startswith('cpu_v'):
        raise ValueError('output_scope')
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    guards = {str(ROOT / name): sha(ROOT / name) for name in OWN}
    import dependencies as dependency
    guards.update({str(ROOT.parent / relative): digest for relative, digest in dependency.FIXED.values()})
    if any(sha(Path(path)) != digest for path, digest in guards.items()):
        raise ValueError('input_changed')
    functions = style()
    import pytest
    with (output / 'PYTEST.log').open('x', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            status = int(pytest.main([str(ROOT / 'test_stream.py'), str(ROOT / 'test_provider.py'), str(ROOT / 'test_dependencies.py'),
                '-q', '-p', 'no:cacheprovider', '--junitxml=' + str(output / 'PYTEST.xml')]))
    unchanged = all(sha(Path(path)) == digest for path, digest in guards.items())
    result = {'pid': os.getpid(), 'test_exit': status, 'seconds': time.perf_counter() - started,
        'guards_unchanged': unchanged, 'style': functions, 'torch_imported': 'torch' in sys.modules,
        'artificial_motion_fixture': True, 'actual_pipeline_caller': False, 'quality_gate_clear': False}
    save(output / 'RESULT.json', result)
    save(output / 'GUARDS.json', guards)
    save(output / 'INDEX.json', {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(result, flush=True)
    return status if unchanged else 1


if __name__ == '__main__':
    raise SystemExit(main())
