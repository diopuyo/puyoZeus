"""固定G3初回を既存supervisorで一度だけ進め、失敗・未対応を保存する。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any, Iterator

from scripts.g3_agent_review import ROOT, VERIFY, read, save

SUPERVISOR = ROOT / 'data/verify/g2_m1_paced_runtime_2026-09-13_v12/supervise.py'
GUARD = ROOT / 'data/verify/g2_bounded_publication_runtime_2026-09-09_v1/resource_guard.py'
ENTRY = ROOT / 'scripts/g3_video38_entry.py'
OTHER_SOURCES = ('video_39', 'video_c74', 'video_c50', 'video_c80', 'video_c138')
MIN_FREE_BYTES = 20 * 1024 ** 3


def supervisor() -> Any:
    """既存のpidfd・実wait・早期guard終了処理をそのまま使う。"""
    original = list(sys.path)
    try:
        sys.path.insert(0, str(SUPERVISOR.parent))
        spec = importlib.util.spec_from_file_location('_g3_existing_supervisor', SUPERVISOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original


@contextmanager
def log_output(path: Path) -> Iterator[None]:
    """既存Popenのstdout/stderrもarm別の原ログへ保存する。"""
    sys.stdout.flush()
    sys.stderr.flush()
    before = (os.dup(1), os.dup(2))
    try:
        with path.open('xb', buffering=0) as stream:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        for descriptor, saved in zip((1, 2), before):
            os.dup2(saved, descriptor)
            os.close(saved)


def run_arm(api: Any, root: Path, arm: str, plan: Path, digest: str) -> dict:
    """再試行なし。entry失敗と実親waitを別票として保存する。"""
    output = root / ('video_38_' + arm)
    command = [sys.executable, '-u', str(ENTRY), '--plan', str(plan), '--plan-sha', digest,
               '--arm', arm, '--output', str(output)]
    save(root / (arm + '_REQUEST.json'), dict(command=command, output=str(output)))
    waited = None
    try:
        with log_output(root / (arm + '.log')):
            waited = api.run_pair(command, str(GUARD), str(ENTRY), root / (arm + '.resources.jsonl'))
        save(root / (arm + '_ACTUAL_WAIT.json'), waited)
        validate_wait(waited)
        return arm_artifacts(root, arm, output, waited)
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        save(root / (arm + '_CONTROL_FAILURE.json'), dict(error=failure, returned_wait=waited,
             process_state='UNKNOWN_DO_NOT_RETRY', actual_wait_claimed=False))
        return dict(arm=arm, wait=waited, wait_verified=False, entry=dict(status='CONTROL_FAILED',
                    error=failure), output=str(output), resource_observed=False, resource_stop=None)


def validate_wait(waited: Any) -> None:
    """欠落した実wait値を成功や0へ補完しない。"""
    if not isinstance(waited, dict):
        raise ValueError('wait_not_object')
    for key in ('child_exit_code', 'resource_guard_exit'):
        if type(waited.get(key)) is not int or not 0 <= waited[key] <= 255:
            raise ValueError('wait_exit_missing_or_invalid:' + key)
    for key in ('guard_exited_first', 'forced_child_kill'):
        if type(waited.get(key)) is not bool:
            raise ValueError('wait_flag_missing_or_invalid:' + key)
    if 'supervisor_error' not in waited:
        raise ValueError('wait_supervisor_error_field_missing')


def arm_artifacts(root: Path, arm: str, output: Path, waited: dict) -> dict:
    """原票の破損も保存済みwaitと一緒に集約し、元ファイルは保持する。"""
    errors, rows = [], []
    entry_path = output / 'ENTRY_RESULT.json'
    try:
        entry = read(entry_path)
        if not isinstance(entry, dict) or not isinstance(entry.get('error') or {}, dict):
            raise ValueError('entry_schema')
    except Exception as error:
        errors.append(dict(path=str(entry_path), error=repr(error)))
        entry = dict(status='ENTRY_ARTIFACT_UNREADABLE', error=dict(message=repr(error)))
    resources = root / (arm + '.resources.jsonl')
    try:
        rows = [json.loads(line) for line in resources.read_text().splitlines()]
        if any(not isinstance(row, dict) or type(row.get('safety_stop')) is not bool for row in rows):
            raise ValueError('resource_row_schema')
    except Exception as error:
        errors.append(dict(path=str(resources), error=repr(error)))
        rows = []
    return dict(arm=arm, wait=waited, wait_verified=True, entry=entry, output=str(output),
                artifact_errors=errors, resource_observed=bool(rows),
                resource_stop=any(row['safety_stop'] for row in rows))


def shared_failure(result: dict) -> bool:
    """確定した共有入力/資源失敗だけで依存armを止める。"""
    if not result.get('wait_verified'):
        return True
    wait = result['wait']
    message = str((result['entry'].get('error') or {}).get('message', ''))
    reasons = ('source_scope', 'source_metadata_changed', 'entry_changed', 'actual_gpu_not_RTX4060',
               'plan_sha', 'runtime_pin_coverage', 'explicit_gpu_zero')
    return bool(wait['guard_exited_first'] or wait['forced_child_kill'] or wait['supervisor_error']
                or wait['resource_guard_exit'] != 0 or not result['resource_observed'] or result['resource_stop']
                or any('g3_video38:' + reason in message for reason in reasons))


def dispatch_arms(api: Any, root: Path, args: Any, results: list) -> None:
    """個別失敗後の独立armだけを一度開始する。"""
    for arm in ('candidate', 'baseline'):
        if results and shared_failure(results[-1]):
            results.append(dict(arm=arm, status='NOT_RUN_SHARED_FAILURE'))
            break
        results.append(run_arm(api, root, arm, args.plan, args.plan_sha))


def main() -> int:
    """一巡後に停止。G4・修復再走・学習へは進まない。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if not root.is_relative_to(VERIFY.resolve()) or shutil.disk_usage(VERIFY).free < MIN_FREE_BYTES:
        raise ValueError('G3_D_storage_boundary_or_capacity')
    if hashlib.sha256(args.plan.read_bytes()).hexdigest() != args.plan_sha:
        raise ValueError('G3_plan_changed')
    root.mkdir(parents=True, exist_ok=False)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ.update(CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
    save(root / 'START.json', dict(plan_sha=args.plan_sha, free_bytes=shutil.disk_usage(root).free,
        code_sha={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), ENTRY, SUPERVISOR, GUARD)},
        source_order=['video_38', *OTHER_SOURCES], one_gpu_process=True, llm_polling=False))
    results, failure = [], None
    try:
        dispatch_arms(supervisor(), root, args, results)
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        save(root / 'FIRST_PASS_RESULT.json', dict(video_38=results, batch_error=failure,
            other_sources=[dict(source=s, status='NOT_RUN_UNSUPPORTED_CANDIDATE_SOURCE_POLICY')
                           for s in OTHER_SOURCES], g3_complete=False, gt_scored=False,
            stop_boundary='USER_REVIEW_NO_G4', retries=0))
    return int(failure is not None)


if __name__ == '__main__':
    raise SystemExit(main())
