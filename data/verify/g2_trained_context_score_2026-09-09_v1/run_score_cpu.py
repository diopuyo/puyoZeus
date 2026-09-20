"""親scorer境界のCPU検収。子loaderの稼働中sourceはguard対象にしない。"""
from __future__ import annotations
import ast
import contextlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import score as P
import test_score as T


def write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def main() -> int:
    output = P.ROOT / sys.argv[1]
    output.mkdir(exist_ok=False)
    paths = [Path(m.__file__).resolve() for m in list(sys.modules.values())
        if getattr(m, '__file__', None) and Path(m.__file__).is_absolute()]
    paths = [p for p in paths if p.suffix == '.py' and p.is_relative_to(P.PROJECT)]
    paths += [T.T.SOURCE, P.ROOT / 'SCORER_CONTRACT.md']
    before, started = {str(p): P.sha(p) for p in paths}, time.perf_counter()
    import pytest
    with (output / 'PYTEST.log').open('x') as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            code = int(pytest.main([str(P.ROOT / 'test_score.py'), '-q', '--tb=short',
                '--confcutdir', str(P.ROOT), '-p', 'no:cacheprovider', '--basetemp', str(output / 'tmp')]))
    after = {p: P.sha(Path(p)) for p in before}
    long = [name + ':' + n.name for name in ('score.py', 'test_score.py', 'run_score_cpu.py')
        for n in ast.walk(ast.parse((P.ROOT / name).read_text()))
        if isinstance(n, ast.FunctionDef) and n.end_lineno - n.lineno + 1 > 50]
    import torch
    good = code == 0 and before == after and not long and not torch.cuda.is_initialized()
    result = {'pid': os.getpid(), 'actual_exit': 0 if good else 1, 'pytest_exit': code,
        'seconds': time.perf_counter() - started, 'before': before, 'after': after,
        'functions_over_50': long, 'cuda_initialized': torch.cuda.is_initialized(),
        'artificial_models': True, 'trained_loader_tested': False, 'quality_gate_clear': False}
    write(output / 'RESULT.json', result)
    write(output / ('COMPLETE.json' if good else 'FAILED.json'), {'sha256':
        {str(p.relative_to(output)): P.sha(p) for p in output.rglob('*') if p.is_file()}})
    print({k: v for k, v in result.items() if k not in ('before', 'after')}, flush=True)
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
