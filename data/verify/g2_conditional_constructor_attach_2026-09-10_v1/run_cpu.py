"""冷 import の共有 loader と原4更新を一回実行し、失敗も排他保存する。"""
from __future__ import annotations
from contextlib import ExitStack
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any
import attach_probe as P

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
SHARED = VERIFY/'g2_conditional_shared_runtime_candidate_2026-09-10_v1'
RUNTIME = VERIFY/'g2_history_publication_probe_runtime_2026-09-10_v13'
SCOPE = VERIFY/'g2_same_scope_stop_guard_2026-09-10_v1'
HOOK = VERIFY/'g2_repeated_firing_runtime_candidate_2026-09-10_v1/constructor_v1/hook.py'
THREADS, MAX_FUNCTION_LINES = 2, 50


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load(alias: str, path: Path, stack: Any) -> Any:
    assert alias not in sys.modules
    spec = importlib.util.spec_from_file_location(alias, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[alias] = value
    stack.callback(sys.modules.pop, alias, None)
    spec.loader.exec_module(value)
    return value


def style() -> dict[str, int]:
    sizes = {}
    for path in ROOT.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            size = node.end_lineno-node.lineno+1
            assert size <= MAX_FUNCTION_LINES and node.returns is not None
            args = node.args.posonlyargs+node.args.args+node.args.kwonlyargs
            assert all(arg.annotation is not None for arg in args)
            sizes[path.name+':'+str(node.lineno)] = size
    return sizes


def modules(stack: Any) -> tuple[Any, ...]:
    old = list(sys.path)
    stack.callback(sys.path.__setitem__, slice(None), old)
    sys.path.insert(0, str(SHARED))
    shared = load('_attach_shared_entry', SHARED/'run_shared.py', stack)
    base = shared.load(stack)
    assert Path(base.K.__file__).resolve() == RUNTIME/'common.py', 'attach_cold_common'
    assert not any(name == 'src' or name.startswith('src.') for name in sys.modules)
    sys.path.insert(0, str(SCOPE))
    scope = load('_attach_scope_references', SCOPE/'scope_stop.py', stack)
    hook = load('_attach_original_hook', HOOK, stack)
    smoke = load('_attach_original_smoke', RUNTIME/'cpu_smoke.py', stack)
    assert smoke.FRAMES == (29052, 29054, 29056, 29058)
    return P.facade(base, scope), scope, hook, smoke


def execute(output: Path, preflight: bool) -> dict[str, Any]:
    before_path, before_profile = list(sys.path), sys.getprofile()
    with ExitStack() as stack:
        candidate, scope, hook, smoke = modules(stack)
        guards = candidate.guards() | smoke.K.guards()
        paths = [HOOK, RUNTIME/'cpu_smoke.py', *SCOPE.glob('*.py'), *ROOT.glob('*.py'), ROOT/'PLAN.md']
        guards.update({str(path): sha(path) for path in paths})
        write(output/'INPUTS.json', guards)
        for path in ROOT.glob('*.py'):
            (output/'source_snapshot'/path.name).write_bytes(path.read_bytes())
        runner = P.derived(smoke, hook, candidate, scope, output)
        write(output/'PREFLIGHT.json', dict(style=style(), cold_common=str(smoke.K.__file__),
            execute_same_code=True, only_drive_changed=True, scope_reference_count=6))
        try:
            result = {} if preflight else runner(output/'full')
            write(output/'FULL_RESULT.json', result)
        finally:
            changed = [path for path, digest in guards.items() if sha(Path(path)) != digest]
            write(output/'GUARDS.json', dict(changed=changed, count=len(guards)))
    assert sys.path == before_path and sys.getprofile() is before_profile
    assert not changed, 'attach_input_changed'
    return dict(sys_path_restored=True, profile_restored=True, guards_unchanged=True,
        actual_updates=0 if preflight else result['actual_update_count'])


def main() -> int:
    name = sys.argv[1]
    assert name.startswith('cpu_') or name.startswith('preflight_')
    output = ROOT/name
    output.mkdir(exist_ok=False)
    (output/'full').mkdir(); (output/'source_snapshot').mkdir()
    started = time.monotonic()
    record: dict[str, Any] = dict(pid=os.getpid(), error=None, cpu_threads=THREADS,
        actual_load_default=False, video_read=False, actual_append=False, quality_pass=False)
    try:
        import cv2
        import torch
        cv2.setNumThreads(THREADS)
        torch.set_num_threads(THREADS); torch.set_num_interop_threads(THREADS)
        record.update(execute(output, name.startswith('preflight_')))
        assert not torch.cuda.is_initialized()
        record.update(exit=0, cuda_initialized=False)
    except BaseException:
        record.update(exit=1, error=traceback.format_exc())
    record['seconds'] = time.monotonic()-started
    write(output/'RESULT.json', record)
    write(output/'INDEX.json', {str(p.relative_to(output)): sha(p)
        for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps(record, ensure_ascii=False))
    return record['exit']


if __name__ == '__main__':
    raise SystemExit(main())
