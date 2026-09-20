"""専用CPUを排他的に保存。自己SHAは完成索引へ含めない。"""
from __future__ import annotations
import ast
import contextlib
import os
from pathlib import Path
import sys
import time
import traceback
import adapter as A
import cpu_fixture as C


def main() -> int:
    if not __debug__:
        raise RuntimeError('optimized_python_unsupported')
    output = A.ROOT / sys.argv[1]
    output.mkdir(exist_ok=False)
    os.environ['PALETTE_PROOF_CPU_OUTPUT'] = str(output)
    started, before = time.perf_counter(), A.guards()
    code, error = 1, None
    try:
        import pytest
        with (output / 'PYTEST.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            code = int(pytest.main(['-q', str(A.ROOT / 'test_adapter.py'), '-p', 'no:cacheprovider']))
    except BaseException:
        error = traceback.format_exc()
    after = {p: A.sha(Path(p)) for p in before}
    violations = [(n, v.name, v.end_lineno - v.lineno + 1) for n in A.OWN if n.endswith('.py')
        for v in ast.walk(ast.parse((A.ROOT / n).read_bytes())) if isinstance(v, ast.FunctionDef)
        and v.end_lineno - v.lineno + 1 > 50]
    result = {'pid': os.getpid(), 'actual_pytest_exit': code, 'error': error,
        'seconds': time.perf_counter() - started, 'before': before, 'after': after,
        'function_over_50': violations, 'quality_gate_clear': False}
    C.write(output / 'RESULT.json', result)
    code = code if before == after and not violations else 1
    if code == 0:
        artifacts = {str(p.relative_to(output)): A.sha(p) for p in output.rglob('*') if p.is_file()}
        C.write(output / 'CPU_COMPLETE.json', {'sha256': artifacts, 'actual_pytest_exit': code})
    print({k: v for k, v in result.items() if k not in ('before', 'after')}, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
