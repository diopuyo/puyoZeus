"""監査改版時も別コード・モデルのCPを受け入れず、旧キャッシュを混ぜない。"""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import image_free_stream as stream
from scripts.image_free_audit_checkpoint import cache_provenance, rebind_cache
from scripts.image_free_cache import CachedOld, cache_key


@pytest.mark.parametrize('previous', [None, 'validated-audit-v1'])
def test_migration_checks_original_fingerprint_and_preserves_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, previous: str | None) -> None:
    identity = dict(code_sha='code', model_sha='model', video_sha='video')
    job = dict(start_frame=0)
    config = dict(old_cache=str(tmp_path/'cache'))
    monkeypatch.setattr(stream, 'identity', lambda *args: dict(identity))
    monkeypatch.setattr(stream.comparison, 'identity', lambda *args: 'candidate')
    legacy = dict(identity, format='continuous_full_outputs_v1', origin=0, seed_sha256=None)
    if previous is not None:
        legacy['execution_control_sha256'] = previous
    key = cache_key(legacy)
    signature = hashlib.sha256((key+'candidate').encode()).hexdigest()
    (tmp_path/'checkpoint.json').write_text(json.dumps(dict(fingerprint=signature)))
    saved = dict(frame=12300, runtimes=[object()], fingerprint=signature)
    def restore(job: dict, config: dict, output: Path, supplied: str, old_key: str) -> dict:
        assert supplied == signature and old_key == key
        return dict(saved)
    monkeypatch.setattr(stream, 'restore', restore)
    with cache_provenance('controls', 'proof', ('validated-audit-v1',)):
        assert stream.identity(None, None, {})['execution_control_sha256'] == 'controls'
        result = stream.restore(job, config, tmp_path, 'new-signature', 'new-key')
    assert result['frame'] == 12300 and result['runtimes'] == saved['runtimes']
    assert result['fingerprint'] == 'new-signature'
    receipt = json.loads((tmp_path/'audit_cp_migration.json').read_text())
    assert receipt['old_fingerprint'] == signature and receipt['proof_sha256'] == 'proof'


def test_inflight_cache_is_rejected_and_computed_state_is_rebound(tmp_path: Path) -> None:
    runtime = object()
    cached = CachedOld(runtime, tmp_path/'old', 'old')
    state = dict(frame=300, runtimes=[cached])
    cached.cached = {'runtime': object()}
    with pytest.raises(ValueError):
        rebind_cache(state, tmp_path, 'new')
    cached.cached = None
    rebind_cache(state, tmp_path, 'new')
    assert state['runtimes'][0].runtime is runtime
    assert state['runtimes'][0].key == 'new'


def test_unrelated_checkpoint_is_not_migrated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stream, 'identity', lambda *args: dict(code_sha='a', model_sha='b', video_sha='c'))
    monkeypatch.setattr(stream.comparison, 'identity', lambda *args: 'candidate')
    (tmp_path/'checkpoint.json').write_text(json.dumps(dict(fingerprint='unrelated')))
    with cache_provenance('controls', 'proof'):
        stream.identity(None, None, {})
        with pytest.raises(ValueError, match='変更'):
            stream.restore(dict(start_frame=0), dict(old_cache=str(tmp_path)), tmp_path, 'new', 'new-key')
    assert not (tmp_path/'audit_cp_migration.json').exists()
