"""上限化観測のCPUを排他保存する。"""
from __future__ import annotations
import contextlib
import os
from pathlib import Path
import sys
import time
import bounded as B


def main() -> int:
    output = B.ROOT / sys.argv[1]
    output.mkdir(exist_ok=False)
    before, started = B.guards(), time.perf_counter()
    import pytest
    with (output / 'PYTEST.log').open('x') as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            code = int(pytest.main([str(B.ROOT / 'test_bounded.py'), '-q', '--tb=short', '--confcutdir',
                str(B.ROOT), '-p', 'no:cacheprovider', '--basetemp', str(output / 'tmp')]))
    after = {p: B.O.sha(Path(p)) for p in before}
    cuda = bool(sys.modules.get('torch') and sys.modules['torch'].cuda.is_initialized())
    good = code == 0 and before == after and not cuda
    result = dict(pid=os.getpid(), actual_exit=0 if good else 1, pytest_exit=code,
        seconds=time.perf_counter() - started, before=before, after=after,
        cuda_initialized=cuda, quality_gate_clear=False)
    B.O.write(output / 'RESULT.json', result)
    B.O.write(output / ('COMPLETE.json' if good else 'FAILED.json'), {'sha256':
        {str(p.relative_to(output)): B.O.sha(p) for p in output.rglob('*') if p.is_file()}})
    print({k: v for k, v in result.items() if k not in ('before', 'after')}, flush=True)
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
