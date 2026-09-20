"""小CPUのみ。最終部品と原画像/途中票を固定して再現する。"""
from __future__ import annotations
import ast
import contextlib
import os
from pathlib import Path
import sys
import time
import adapter as A


def main() -> int:
    output = A.ROOT / sys.argv[1]
    output.mkdir(exist_ok=False)
    started, before = time.perf_counter(), A.guards()
    for name in ('probe.py', 'partial_cpu_v1/CAPTURED_ROW.json', 'partial_cpu_v1/REPRODUCTION.json'):
        before[str(A.ROOT / name)] = A.sha(A.ROOT / name)
    image = A.ROOT.parent / 'g2_first_inventory_break_2026-09-09_v1/physical_v1/source_32798.png'
    before[str(image)] = A.sha(image)
    import pytest
    with (output / 'PYTEST.log').open('x') as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            code = int(pytest.main([str(A.ROOT / 'test_adapter.py'), '-q', '--tb=short',
                '--confcutdir', str(A.ROOT), '-p', 'no:cacheprovider', '--basetemp', str(output / 'tmp')]))
    after = {p: A.sha(Path(p)) for p in before}
    long = [n.name for file in A.OWN if file.endswith('.py') for n in ast.walk(ast.parse((A.ROOT / file).read_bytes()))
            if isinstance(n, ast.FunctionDef) and n.end_lineno - n.lineno + 1 > 50]
    cuda = bool(sys.modules.get('torch') and sys.modules['torch'].cuda.is_initialized())
    good = code == 0 and before == after and not long and not cuda
    result = {'pid': os.getpid(), 'actual_exit': 0 if good else 1, 'pytest_exit': code,
        'seconds': time.perf_counter() - started, 'before': before, 'after': after,
        'functions_over_50': long, 'cuda_initialized': cuda, 'quality_gate_clear': False}
    A.write(output / 'RESULT.json', result)
    A.write(output / ('COMPLETE.json' if good else 'FAILED.json'), {'sha256':
        {str(p.relative_to(output)): A.sha(p) for p in output.rglob('*') if p.is_file()}})
    print({k: v for k, v in result.items() if k not in ('before', 'after')}, flush=True)
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
