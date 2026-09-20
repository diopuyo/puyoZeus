"""二thread CPU専用。既存prepare fixtureと全byte再照合を再利用する。"""
from __future__ import annotations
import ast
import contextlib
import importlib.util
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent


def main() -> int:
    if not __debug__:
        raise RuntimeError('optimized_python_forbidden')
    spec = importlib.util.spec_from_file_location('_atomic_journal', ROOT / 'observer.py')
    journal = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = journal
    spec.loader.exec_module(journal)
    output = ROOT / sys.argv[1]
    output.mkdir(exist_ok=False)
    os.environ['MANUFACTURING_OUTPUT'] = str(output)
    started, before = time.perf_counter(), journal.guards()
    import test_observer as tests
    before.update(tests.T.A.guards())
    before.update(tests.P.H.guards())
    for module in (tests.P, tests.T):
        before[module.__file__] = journal.sha(Path(module.__file__))
    import pytest
    with (output / 'PYTEST.log').open('x') as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            code = int(pytest.main([str(ROOT / 'test_observer.py'), '-q', '--tb=short', '--confcutdir',
                str(ROOT), '-p', 'no:cacheprovider', '--basetemp', str(output / 'tmp')]))
    after = {path: journal.sha(Path(path)) for path in before}
    long = [node.name for name in journal.OWN if name.endswith('.py')
        for node in ast.walk(ast.parse((ROOT / name).read_text()))
        if isinstance(node, ast.FunctionDef) and node.end_lineno - node.lineno + 1 > 50]
    cuda = bool(sys.modules.get('torch') and sys.modules['torch'].cuda.is_initialized())
    good = code == 0 and before == after and not long and not cuda
    result = {'pid': os.getpid(), 'actual_exit': 0 if good else 1, 'pytest_exit': code,
        'seconds': time.perf_counter() - started, 'before': before, 'after': after,
        'functions_over_50': long, 'cuda_initialized': cuda, 'quality_gate_clear': False}
    journal.write(output / 'RESULT.json', result)
    files = {str(p.relative_to(output)): journal.sha(p) for p in output.rglob('*') if p.is_file()}
    journal.write(output / ('COMPLETE.json' if good else 'FAILED.json'), {'sha256': files})
    print({k: v for k, v in result.items() if k not in ('before', 'after')}, flush=True)
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
