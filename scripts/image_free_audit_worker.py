"""固定計算本体へ証明済み監査再利用を接続する独立ワーカー入口。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from contextlib import nullcontext
from unittest.mock import patch

from image_free_audit_checkpoint import cache_provenance, control_sha
from image_free_audit_reuse import FrameMemo, reuse_audit
from image_free_reference_fast import fast_reference

FILES = ('image_free_audit_worker.py', 'image_free_audit_checkpoint.py', 'image_free_audit_reuse.py',
         'image_free_reference_fast.py')


def verify_gate(root: Path, config: dict, source: Path) -> tuple[str, str, bool]:
    digest = control_sha([root/name for name in FILES])
    if config.get('audit_proof_mode'):
        return digest, 'validation-in-progress', True
    path = root/'audit_gate.json'
    gate = json.loads(path.read_text())
    if gate['control_sha256'] != digest or not gate['passed']:
        raise ValueError('監査再利用の一致証明/制御SHAが不正です')
    manifest = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for base in ('src', 'scripts') for p in (source/base).rglob('*.py')}
    job_id = sys.argv[sys.argv.index('--worker')+1]
    route = next(j['route'] for j in config['videos'] if j['id'] == job_id)
    applicable = manifest == gate['source_manifest'] and route in gate['routes']
    return digest, hashlib.sha256(path.read_bytes()).hexdigest(), applicable


def main() -> None:
    root = Path(__file__).resolve().parent
    source = Path(sys.argv[sys.argv.index('--source')+1])
    sys.path.insert(0, str(source))
    from scripts import image_free_parallel as parallel, image_free_stream as stream
    config = json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
    job_id = sys.argv[sys.argv.index('--worker')+1]
    digest, proof, applicable = verify_gate(root, config, source)
    gate = json.loads((root/'audit_gate.json').read_text()) if (root/'audit_gate.json').exists() else {}
    route = next(j['route'] for j in config['videos'] if j['id'] == job_id)
    directory = Path(config['output'])/job_id
    if not applicable:
        parallel.save_json(directory/'audit_fallback.json', dict(reason='この経路/ソースでは再利用を採用していません', control_sha256=digest))
        parallel.main()
        return
    original = stream.run
    def run(job: dict, config: dict, guard: object) -> None:
        # 検証だけ: 同居版の実測前史CPを、変更前と同じ保存形式へ載せる。
        if config.get('audit_proof_mode') and job_id.endswith('_resume') and not (directory/'checkpoint.json').exists():
            signature, key = stream.fingerprint(job, config, directory)
            state = stream.restore(job, config, directory, signature, key)
            stream.comparison.checkpoint(directory, state, [])
        with cache_provenance(digest, proof, tuple(gate.get('predecessors', []))):
            original(job, config, guard)
    audit = reuse_audit() if route == 'old' else nullcontext(FrameMemo())
    with audit as memo, fast_reference(), patch.object(stream, 'run', run):
        try:
            parallel.main()
        finally:
            parallel.save_json(directory/'audit_reuse.json', dict(counts=dict(memo.counts),
                control_sha256=digest, proof_sha256=proof, limit_bytes=memo.limit))


if __name__ == '__main__':
    main()
