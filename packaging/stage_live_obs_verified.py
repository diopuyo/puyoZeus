"""完成ZIPと一致する既存資産だけを共有し、実データの重複を避ける。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import zipfile

from build_bundle import collect_app_files, source_of

RELEASE = Path('D:/puyo_analyzer/packaging/build_20261020c_release')
BASE = Path(_public_dev_text('<work>/build_20261020c_validation'))
WORK = Path('D:/puyo_analyzer/packaging/build_20261020d_work')
ASSETS = Path(_public_dev_text('<work>/assets/baseline_b/app'))


def source_plan() -> dict:
    inventory = json.loads((RELEASE/'files_sha256.json').read_text(encoding='utf-8'))
    changed = []
    for name in collect_app_files(ASSETS):
        data = source_of(name, ASSETS).read_bytes()
        if sha256(data).hexdigest() != inventory.get('PuyoLive/app/'+name):
            changed.append(dict(name=name, bytes=len(data)))
    return dict(changed=changed, source_bytes=sum(r['bytes'] for r in changed))


def link_stage(destination: Path, base: Path = BASE) -> None:
    """書換え時はbuild_live_obs_20261020b.break_linkで旧配布から切り離す。"""
    assert destination.name == 'PuyoLive' and not destination.exists()
    assert destination.resolve() != (base/'PuyoLive').resolve()
    receipt = json.loads((RELEASE/'archive_receipt.json').read_text(encoding='utf-8'))
    raw = Path(receipt['zip']).read_bytes()
    assert sha256(raw).hexdigest() == receipt['sha256']
    inventory = json.loads((RELEASE/'files_sha256.json').read_text(encoding='utf-8'))
    restored, linked_bytes = [], 0
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        for name, digest in inventory.items():
            source, target = base/name, destination.parent/name
            assert target.resolve().is_relative_to(destination.resolve()), name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_file() and sha256(source.read_bytes()).hexdigest() == digest:
                os.link(source, target)
                linked_bytes += source.stat().st_size
            else:
                data = archive.read(name)
                assert sha256(data).hexdigest() == digest
                target.write_bytes(data)
                restored.append(dict(name=name, bytes=len(data)))
    value = dict(destination=str(destination), files=len(inventory), linked_bytes=linked_bytes,
                 restored_from_completed_zip=restored, source_zip_sha256=receipt['sha256'])
    (WORK/'link_stage_receipt.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(value, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.destination is None:
        value = source_plan()
        (WORK/'reuse_source_plan.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
        print(value, flush=True)
    else:
        link_stage(args.destination)
