"""合成前後の全sourceを固定してCPU結果を排他保存する。"""
from __future__ import annotations
import contextlib
import os
from pathlib import Path
import sys
import time
import live_cli as L


def main() -> int:
    output = L.ROOT / sys.argv[1]
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    before = L.guards() | L.C.A.guards() | L.C.L.H.guards()
    import pytest
    with (output / 'PYTEST.log').open('x') as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            code = int(pytest.main([str(L.ROOT / 'test_runtime.py'), '-q', '--tb=short',
                '--confcutdir', str(L.ROOT), '-p', 'no:cacheprovider', '--basetemp', str(output / 'tmp')]))
    after = {p: L.F.sha(Path(p)) for p in before}
    cuda = bool(sys.modules.get('torch') and sys.modules['torch'].cuda.is_initialized())
    good = code == 0 and before == after and not cuda
    result = {'pid': os.getpid(), 'actual_exit': 0 if good else 1, 'pytest_exit': code,
        'seconds': time.perf_counter() - started, 'before': before, 'after': after,
        'cuda_initialized': cuda, 'quality_gate_clear': False}
    L.F.P.write(output / 'RESULT.json', result)
    L.F.P.write(output / ('COMPLETE.json' if good else 'FAILED.json'), {'sha256':
        {str(p.relative_to(output)): L.F.sha(p) for p in output.rglob('*') if p.is_file()}})
    print({k: v for k, v in result.items() if k not in ('before', 'after')}, flush=True)
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
