"""G3の登録source/有理時計を元モデル子processへ接続する。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator
import uuid

from scripts import g3_source_models as M
from scripts.g3_agent_review import VERIFY

PROCESS = M.VERIFY / 'g2_model_process_bridge_2026-09-11_v1'
WORKER_SHA = 'fe12bd2dbb44c8f2798f782895e1f48fcbc39b9a8c2af0de406acdf0b89d3439'
CLIENT_SHA = '974683915ff27d1a77e7af583d78128996a772326fd31e128e681c8f8d1798ea'
SCORE_SHA = '90550cf638f331e8eb9ade46607262519e78503a943d2730d890068cf0cc6e06'
FPS_SHA = 'da0a6ff3c59ce1055b053424c05c309ad0969f599c1c7cde493dcbf03c4c3066'
MAX_BYTES = 32 * 1024 * 1024
TIMEOUT_SECONDS = 180


def encoded(value: Any) -> bytes:
    """元wireと同じcanonical表現。"""
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(',', ':')).encode()


def route(path: Path, sha: str, source: str) -> dict[str, Any]:
    """順序で結合せず、元source IDで時計/model登録を照合する。"""
    registry = M.document(path, sha)
    M.require(registry['schema'] == 'g3-model-process-registry/1', 'registry_schema')
    item = M.one(registry['sources'], 'source', source)
    source_sha = item['source_id'].removeprefix('sha256:')
    contract = M.source_contract(source, source_sha)
    M.require(all(item.get(k) == v for k, v in contract.items()), 'registry_source_contract')
    registered = M.one(M.document(M.PREREG, M.PREREG_SHA)['sources'], 'source_video_id', source)
    clock, prefix = registered['source'], registered['required_continuous_processing']
    M.require(clock['source_video_id'] == source and clock['source_video_sha256'] == source_sha,
              'registry_clock_source')
    M.require(prefix['start_frame'] == 0 and prefix['seek_between_windows'] is False, 'registry_history')
    numerator, denominator = clock['time_base_numerator'], clock['time_base_denominator']
    with loaded(M.ROOT / 'src/fps_normalize.py', FPS_SHA) as fps:
        stride = fps.resolve_normalize_fps_30_stride(denominator / numerator)
    M.require(item['time_base'] == [numerator, denominator]
              and item['stride'] == stride and item['end_frame_exclusive'] == prefix['end_frame_exclusive'],
              'registry_clock')
    fold = registry['folds'][str(contract['fold'])]
    M.require(hashlib.sha256(encoded(fold['members'])).hexdigest()
              == fold['manifest_sha256'] == item['model_manifest_sha256'], 'registry_manifest')
    for name, expected in (fold['guards'] | fold['checkpoint_guards']).items():
        target = Path(name).resolve()
        M.require(target.is_relative_to(M.ROOT), 'registry_guard_path')
        M.require(M.digest(target) == expected, 'registry_guard_changed:' + name)
    return item


def request_gate(request: dict[str, Any], item: dict[str, Any]) -> None:
    """子側でもsourceと範囲を確認し、外来rowへ登録時計を適用しない。"""
    M.require(type(request) is dict and type(request.get('row')) is dict, 'request_mapping')
    row = request['row']
    M.require(row.get('source_id') == item['source_id'], 'request_source')
    frame = request.get('frame')
    M.require(type(frame) is int and 0 <= frame < item['end_frame_exclusive']
              and frame % item['stride'] == 0 and row.get('frame_idx') == frame, 'request_frame')
    numerator, denominator = item['time_base']
    M.require(type(row.get('time_sec')) in (int, float)
              and row['time_sec'] == frame * numerator / denominator, 'request_clock')


@contextmanager
def loaded(path: Path, sha: str) -> Iterator[ModuleType]:
    """専用moduleのみ登録/解放し、原ファイルを変更しない。"""
    M.require(M.digest(path) == sha, 'process_source_changed')
    name = '_g3_process_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    M.require(spec is not None and spec.loader is not None, 'process_spec')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        if sys.modules.get(name) is module:
            del sys.modules[name]


def identity_bridge(original: Any, item: dict[str, Any]) -> Any:
    """旧workerの固定registrationを認証してから元identityへ新時計を渡す。"""
    def identity(row: dict[str, Any], old: dict[str, Any]) -> None:
        request_gate(dict(row=row, frame=row.get('frame_idx')), item)
        expected = dict(source_id=item['source_id'], run_id=row['run_id'],
                        time_base_numerator=1, time_base_denominator=60, ledger_connection='NOT_CONNECTED')
        M.require(old == expected, 'original_worker_registration')
        numerator, denominator = item['time_base']
        original(row, expected | dict(time_base_numerator=numerator, time_base_denominator=denominator))
    return identity


def evaluate_worker(request: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    """モデルimportは専用子Pythonだけで行う。元bound/evaluateを一回呼ぶ。"""
    request_gate(request, item)
    paths = list(sys.path)
    try:
        with loaded(PROCESS / 'worker.py', WORKER_SHA) as worker:
            with M.source_loader(item['source'], item['source_id'].removeprefix('sha256:')) as loader:
                with loaded(M.LOADER.parent / 'score.py', SCORE_SHA) as score:
                    score.fixed_sources()
                    original_backend = worker.T.T.backend
                    original_identity = score.B._identity
                    score.B = SimpleNamespace(**vars(score.B))
                    score.B._identity = identity_bridge(original_identity, item)
                    score.load_loader = lambda: loader
                    def backend() -> ModuleType:
                        score.fixed_sources()
                        return score
                    worker.T.T.backend = backend
                    try:
                        result = worker.evaluate(request)
                        M.require(result['packet']['model_artifact_sha256']
                                  == item['model_manifest_sha256'], 'worker_manifest')
                        return result
                    finally:
                        worker.T.T.backend = original_backend
    finally:
        sys.path[:] = paths


def exchange(request: dict[str, Any], registry: Path, registry_sha: str,
             source: str, output: Path) -> dict[str, Any]:
    """元client検査の後に保存。失敗stdout/stderrと実waitを残し再送しない。"""
    item = route(registry, registry_sha, source)
    request_gate(request, item)
    M.require(output.resolve().is_relative_to(VERIFY.resolve()), 'process_output_root')
    output.mkdir()
    raw, started = encoded(request), time.monotonic()
    M.require(len(raw) <= MAX_BYTES, 'process_request_size')
    (output / 'REQUEST.json').write_bytes(raw)
    args = [sys.executable, '-m', 'scripts.g3_model_process', '--registry', str(registry),
            '--registry-sha', registry_sha, '--source', source, '--code-sha', M.digest(Path(__file__))]
    environment = dict(os.environ, PYTHONPATH=str(M.ROOT), CUDA_VISIBLE_DEVICES='-1',
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    status: dict[str, Any] = dict(command=args, quality_gate_clear=False)
    try:
        child = subprocess.run(args, input=raw, capture_output=True, cwd=M.ROOT,
                               env=environment, timeout=TIMEOUT_SECONDS)
        status.update(exit_code=child.returncode, actual_wait=True)
        (output / 'stdout.log').write_bytes(child.stdout)
        (output / 'stderr.log').write_bytes(child.stderr)
        M.require(child.returncode == 0 and len(child.stdout) <= MAX_BYTES, 'model_process_failed')
        with loaded(PROCESS / 'client.py', CLIENT_SHA) as client:
            client.MODEL_MANIFEST = item['model_manifest_sha256']
            packet = client.validated(json.loads(child.stdout), request)
        (output / 'PACKET.json').write_bytes(encoded(packet))
        status['status'] = 'VALIDATED'
        return packet
    except BaseException as error:
        status.update(status='FAILED', error=repr(error))
        if isinstance(error, subprocess.TimeoutExpired):
            (output / 'stdout.log').write_bytes(error.stdout or b'')
            (output / 'stderr.log').write_bytes(error.stderr or b'')
            status.update(timeout=True, actual_wait=True)
        raise
    finally:
        status['seconds'] = time.monotonic() - started
        (output / 'RESULT.json').write_bytes(encoded(status))


def main() -> None:
    """親から固定されたsource/registryだけを一回処理する。"""
    parser = argparse.ArgumentParser()
    for name in ('registry', 'registry-sha', 'source', 'code-sha'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    M.require(M.digest(Path(__file__)) == args.code_sha, 'worker_code_changed')
    raw = sys.stdin.buffer.read(MAX_BYTES + 1)
    M.require(len(raw) <= MAX_BYTES, 'worker_request_size')
    with redirect_stdout(sys.stderr):
        item = route(Path(args.registry), args.registry_sha, args.source)
        result = evaluate_worker(json.loads(raw), item)
    sys.stdout.buffer.write(encoded(result) + b'\n')


if __name__ == '__main__':
    main()
