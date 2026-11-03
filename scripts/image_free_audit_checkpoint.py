"""一致証明済み監査制御のSHAをキャッシュへ加え、時系列CPだけを引き継ぐ。"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch


def control_sha(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def rebind_cache(state: dict, directory: Path, key: str) -> None:
    from scripts.image_free_cache import CachedOld
    runtime = state['runtimes'][0]
    if not isinstance(runtime, CachedOld):
        return
    if runtime.cached is not None:
        raise ValueError('キャッシュ読出し途中のCPは監査改版へ移行できません')
    state['runtimes'] = [CachedOld(runtime.runtime, directory/key, key, runtime.verify_frames)]


@contextmanager
def cache_provenance(digest: str, proof_sha: str,
                     predecessors: tuple[str, ...] = ()) -> Iterator[None]:
    from scripts import image_free_stream as stream
    from scripts.image_free_cache import cache_key
    original_identity, original_restore = stream.identity, stream.restore
    legacy: dict[str, Any] = {}

    def identified(video: Path, source: Path, config: dict) -> dict:
        value = original_identity(video, source, config)
        legacy['identity'] = dict(value)
        return dict(value, execution_control_sha256=digest)

    def restored(job: dict, config: dict, output: Path, signature: str, key: str) -> dict:
        manifest = output/'checkpoint.json'
        if not manifest.exists() or json.loads(manifest.read_text())['fingerprint'] == signature:
            return original_restore(job, config, output, signature, key)
        record = dict(legacy['identity'], format='continuous_full_outputs_v1',
                      origin=job['start_frame'], seed_sha256=job.get('seed_sha256'))
        saved = json.loads(manifest.read_text())['fingerprint']
        candidate = stream.comparison.identity(job, config)
        for previous in (None, *predecessors):
            identity = dict(record, execution_control_sha256=previous) if previous else record
            prior_key = cache_key(identity)
            prior = hashlib.sha256((prior_key+candidate).encode()).hexdigest()
            if prior == saved:
                break
        else:
            raise ValueError('未検証のコード・モデル・入力変更です')
        # 元のrestore自身が、ソース・モデル・入力の完全一致を検査する。
        state = original_restore(job, config, output, prior, prior_key)
        rebind_cache(state, Path(config['old_cache']), key)
        state['fingerprint'] = signature
        stream.comparison.save_json(output/'audit_cp_migration.json', dict(
            frame=state['frame'], old_fingerprint=prior, new_fingerprint=signature,
            execution_control_sha256=digest, proof_sha256=proof_sha, continuous_history=True))
        return state

    with ExitStack() as stack:
        stack.enter_context(patch.object(stream, 'identity', identified))
        stack.enter_context(patch.object(stream, 'restore', restored))
        yield
