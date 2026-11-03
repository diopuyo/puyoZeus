"""照合済みL200Hと正式EX資産だけを24hへ同梱する。"""

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

from hashlib import sha256
import json
from pathlib import Path
import shutil

JOB = Path(_public_dev_text('<work>/codex_jobs/live_24h_20261027'))
MODEL_SHA = '11a611abaf571022a05cdd0779129713d4190d5f80059056d1fd8d2321139119'
SEED_SHA = '5aec04e78d2e3128b4706978969c669b5cb27ee79f7e22ea9f7d40f219d4f851'
POLICY_SHA = 'b32374fb8bfd07ee739f9559f9e6f38b88d5ac3c000e3a71dedb9fcb66c309c4'
SOURCE_EXTRAS = ('src/phase_j/_live_ex_frozen/SOURCE.json',
                 'src/phase_j/live_ex_native_golden.json', 'src/phase_j/overlay_24h.js')


def files() -> dict[str, tuple[Path, str]]:
    manifest = JOB/'models_L200H/ensemble.json'
    assert sha256(manifest.read_bytes()).hexdigest() == MODEL_SHA
    data = json.loads(manifest.read_text(encoding='utf-8'))
    assert len(data['members']) == 3 and data['aggregate'] == 'mean_logit'
    result = {'app/models/b3_ex_L200H/ensemble.json': (manifest, MODEL_SHA)}
    for member in data['members']:
        source = manifest.parent/member['name']
        assert sha256(source.read_bytes()).hexdigest() == member['sha256']
        result['app/models/b3_ex_L200H/'+member['name']] = (source, member['sha256'])
    policy = JOB/'policy_locked.json'
    canonical = json.dumps(json.loads(policy.read_text(encoding='utf-8-sig')),
                           sort_keys=True, separators=(',', ':')).encode()
    assert sha256(canonical).hexdigest() == POLICY_SHA
    seed = JOB/'seed_table.npz'
    assert sha256(seed.read_bytes()).hexdigest() == SEED_SHA
    result.update({'app/models/live_ex/policy_locked.json': (policy, sha256(policy.read_bytes()).hexdigest()),
                   'app/models/live_ex/seed_table.npz': (seed, SEED_SHA)})
    return result


def stage(bundle: Path) -> dict:
    receipt = {}
    for relative, (source, digest) in files().items():
        target = bundle/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha256(target.read_bytes()).hexdigest() == digest
        receipt[relative] = digest
    return receipt


def verify_archive(archive: object) -> dict:
    manifest = json.loads(archive.read('PuyoLive/app/MANIFEST.json'))['files']
    checks = {}
    for relative, (_, digest) in files().items():
        record = manifest.get(relative.removeprefix('app/'), {})
        checks[relative] = (sha256(archive.read('PuyoLive/'+relative)).hexdigest() == digest
                            and record.get('sha256') == digest)
    return dict(files=checks, passed=all(checks.values()))
