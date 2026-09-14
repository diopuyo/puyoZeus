"""固定video38診断区間の原run/全終了を使う専用入口。本番live入口は変更しない。"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
import time
import traceback
from types import FunctionType
from typing import Any
import owned_adapter as A

ROOT = Path(__file__).resolve().parent
GO = ROOT / 'TARGET_GO.json'
OUTPUT_NAME = 'video38_m1_completion_candidate_v11'
THREADS = 1
FINALIZER_ROOT = ROOT.parent / 'g2_history_publication_probe_runtime_2026-09-10_v13'


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('target_entry:' + reason)


def runtime_sources() -> tuple[Path, ...]:
    """このrootから動的に使う確率層も原SHA票へ含める。検査専用ファイルは除く。"""
    paths = [p for root in (ROOT, A.PRIOR) for p in root.glob('*.py')
             if not p.name.startswith(('test_', 'probe_'))]
    return tuple(sorted(set(paths) | set(A.sources()) | {ROOT / 'run_whole_target.sh'}))


def approved(output: Path) -> None:
    value = json.loads(GO.read_bytes())
    require(value['decision'] == 'GO_fixed_whole_target_observation', 'target_not_approved')
    require(value['quality_gate_clear'] is False and value['production_permission'] is False, 'target_authority')
    require(value['known_connection_failures'] == [] and bool(value['stop_conditions']), 'target_conditions')
    require(output == ROOT.parent / OUTPUT_NAME == Path(value['output']), 'target_scope')
    pins = value['frozen_files']
    require(all(str(p) in pins for p in runtime_sources()), 'target_source_coverage')
    require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in pins.items()), 'target_changed')


def finalize(output: Path, child_exit: int, resource_exit: int) -> dict:
    """生成層やGO再検査に依存せず、実wait結果を先に保存して原finalizeへ渡す。"""
    require(output == ROOT.parent / OUTPUT_NAME and output.is_dir() and not output.is_symlink(), 'finalize_scope')
    require(all(type(v) is int and 0 <= v <= 255 for v in (child_exit, resource_exit)), 'actual_exit_codes')
    with (output / 'TARGET_PARENT_WAIT.json').open('x') as stream:
        json.dump(dict(child_exit_code=child_exit, resource_guard_exit=resource_exit, source='actual_wait'), stream)
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(FINALIZER_ROOT))
        closure = importlib.import_module('closure')
        require(Path(closure.__file__).resolve() == FINALIZER_ROOT / 'closure.py', 'finalizer_module')
        return closure.finalize(output, child_exit, resource_exit, Path(str(output) + '.resources.jsonl'))
    finally:
        sys.path[:] = previous


def selected_run(main: Any) -> Any:
    """原run_liveのcodeを維持し、検収済みcollectと同じ設定参照へ結ぶ。"""
    namespace = main.__globals__
    original = namespace['run_live']
    bindings = {key: namespace[key] for key in ('collect', 'S', 'K')}
    run = FunctionType(original.__code__, dict(original.__globals__, **bindings),
                       original.__name__, original.__defaults__, original.__closure__)
    run.__kwdefaults__ = original.__kwdefaults__
    return run


def observe(main: Any, output: Path) -> int:
    common, closure = main.__globals__['K'], main.__globals__['Q']
    assert not output.exists() and not output.is_symlink(), 'exclusive_target'
    started, code = time.perf_counter(), 0
    try:
        import cv2
        import torch
        cv2.setNumThreads(THREADS)
        torch.set_num_threads(THREADS)
        torch.set_num_interop_threads(THREADS)
        result = selected_run(main)(output, [])
    except BaseException:
        result, code = dict(error=traceback.format_exc()), 1
        output.mkdir(exist_ok=True)
    result.update(exit_code=code, seconds=time.perf_counter() - started, pid=os.getpid(),
                  production_permission=False, quality_gate_clear=False)
    common.write(output / 'ENTRY_RESULT.json', result)
    if code == 0:
        closure.seal(output, common.read(output / 'SUMMARY.json'), common.read(output / 'PLAN.json'))
    print(json.dumps(result), flush=True)
    return code


def main() -> int:
    require(sys.flags.optimize == 0, 'python_optimization_forbidden')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('observe', 'finalize', 'review-m1'), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--child-exit', type=int)
    parser.add_argument('--resource-exit', type=int)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.mode == 'finalize':
        print(json.dumps(finalize(output, args.child_exit, args.resource_exit)), flush=True)
        return 0
    approved(output)
    if args.mode == 'review-m1':
        import review_m1
        result = review_m1.inspect(output)
        approved(output)
        result['runtime_source_verification_required'] = False
        result['runtime_sources_verified'] = True
        with (output / 'M1_SAVED_REVIEW.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result), flush=True)
        return 0
    with ExitStack() as stack:
        selected = A.configured(stack)
        common = selected.__globals__['K']
        pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_sources()}
        A.protect(stack, common, pins)
        if selected.__globals__['S'].K is not common:
            A.protect(stack, selected.__globals__['S'].K, pins)
        return observe(selected, output)


if __name__ == '__main__':
    raise SystemExit(main())
