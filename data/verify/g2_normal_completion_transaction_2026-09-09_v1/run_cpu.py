"""通常完了の縮小実経路・CPU成果を排他保存する。モデル/動画は読まない。"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import transaction as T

OWN = ('transaction.py', 'live_provider.py', 'test_transaction.py', 'run_cpu.py', 'PLAN.md')


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def evidence(test: Any) -> Any:
    parts = test.setup()
    pipe, sm, raw, provider, controller, step, detector = parts
    history = []
    for frame in (2, 4, 6, 8, 10, 12, 14, 16, 18):
        detector.target = test.BoardState.OJAMA_FALL if frame == 8 else test.BoardState.STABLE
        if frame == 16:
            provider.append_new()
        elif frame == 18:
            provider.added = ()
        result = test.invoke(parts, frame)
        history.append({'frame': frame, 'state': result.state.value, 'confirmed': T.board_key(result.confirmed_board),
            'counter': dict(pipe._tsumo_count_1p), 'pending': list(pipe._pending_tsumo_1p)})
    T.require(history[-2]['state'] == 'stable' and history[-1]['counter'] == history[-2]['counter'], 'finite_return')
    return {'records': controller.records, 'history': history,
        'actual_sm_consume_infer': True, 'full_step_executed': False,
        'next_and_detector_proposal_artificial': True, 'physical_certified': False, 'quality_gate_clear': False}


def main() -> int:
    output = T.ROOT / sys.argv[1]
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    import test_transaction as test
    import pytest
    paths = [T.ROOT / n for n in OWN] + [T.PREVIEW, test.SOURCE, Path(test.A.__file__), Path(test.D.__file__)]
    sources = [Path(module.__file__).resolve() for name, module in tuple(sys.modules.items())
        if name == 'src' or name.startswith('src.')]
    T.require(all(p.is_relative_to(test.SNAPSHOT) for p in sources), 'nonfrozen_src_import')
    before = {str(p): sha(p) for p in paths + sources}
    with (output / 'PYTEST.log').open('x', encoding='utf-8') as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            code = int(pytest.main([str(T.ROOT / 'test_transaction.py'), '-q', '--tb=short',
                '--confcutdir', str(T.ROOT), '-p', 'no:cacheprovider', '--junitxml=' + str(output / 'pytest.xml')]))
    if code == 0:
        write(output / 'EVIDENCE.json', evidence(test))
    after = {path: sha(Path(path)) for path in before}
    long = [(name, n.name) for name in OWN if name.endswith('.py')
        for n in ast.walk(ast.parse((T.ROOT / name).read_bytes()))
        if isinstance(n, ast.FunctionDef) and n.end_lineno - n.lineno + 1 > 50]
    cuda = bool(sys.modules.get('torch') and sys.modules['torch'].cuda.is_initialized())
    good = code == 0 and before == after and not long and not cuda
    result = {'pid': os.getpid(), 'actual_exit': 0 if good else 1, 'pytest_exit': code,
        'seconds': time.perf_counter() - started, 'before': before, 'after': after,
        'functions_over_50': long, 'cuda_initialized': cuda, 'quality_gate_clear': False}
    write(output / 'RESULT.json', result)
    files = {str(p.relative_to(output)): sha(p) for p in output.iterdir() if p.is_file()}
    write(output / ('COMPLETE.json' if good else 'FAILED.json'), {'sha256': files})
    print({k: v for k, v in result.items() if k not in ('before', 'after')}, flush=True)
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
