"""実採録metadata observerのCPU検収を専用rootへ保存する。"""
from __future__ import annotations
import contextlib
import os
from pathlib import Path
import sys
import time
import observer as O


def main() -> int:
    output = O.ROOT / sys.argv[1]
    output.mkdir(exist_ok=False)
    before, started = O.guards(), time.perf_counter()
    import pytest
    with (output / 'PYTEST.log').open('x') as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            code = int(pytest.main([str(O.ROOT / 'test_observer.py'), '-q', '--tb=short',
                '--confcutdir', str(O.ROOT), '-p', 'no:cacheprovider', '--basetemp', str(output / 'tmp')]))
    after = {p: O.sha(Path(p)) for p in before}
    cuda = bool(sys.modules.get('torch') and sys.modules['torch'].cuda.is_initialized())
    good = code == 0 and before == after and not cuda
    result = {'pid': os.getpid(), 'actual_exit': 0 if good else 1, 'pytest_exit': code,
        'seconds': time.perf_counter() - started, 'before': before, 'after': after,
        'cuda_initialized': cuda, 'quality_gate_clear': False}
    O.write(output / 'RESULT.json', result)
    O.write(output / ('COMPLETE.json' if good else 'FAILED.json'), {'sha256':
        {str(p.relative_to(output)): O.sha(p) for p in output.rglob('*') if p.is_file()}})
    print({k: v for k, v in result.items() if k not in ('before', 'after')}, flush=True)
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
