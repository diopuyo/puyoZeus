"""未起動のステージを封印し、10/17bとの差分と別展開先を作る。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import hashlib
import json
from pathlib import Path
import zipfile
import sys
from io import BytesIO
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bundle_archive import write_bundle_zip, extract_smoke_copy, is_runtime_file

OUT = Path('D:/puyo_analyzer/packaging/build_20261019a_release')
OLD = Path('D:/puyo_analyzer/packaging/build_20261017b/PuyoLive_20261017b.zip')


def inventory(path: Path, preload: bool = False) -> dict[str, str]:
    with zipfile.ZipFile(BytesIO(path.read_bytes()) if preload else path) as archive:
        return {i.filename: hashlib.sha256(archive.read(i)).hexdigest()
                for i in archive.infolist() if not i.is_dir()}


def main() -> None:
    archive = write_bundle_zip(Path(_public_dev_text('<work>/build_20261019a_stage/PuyoLive')), OUT/'PuyoLive_20261019a.zip')
    before, after = inventory(OLD), inventory(archive)
    runtime = [n for n in after if is_runtime_file(n.removeprefix('PuyoLive/'))]
    assert not runtime, runtime
    changes = dict(added=sorted(after.keys()-before.keys()), removed=sorted(before.keys()-after.keys()),
        changed=[n for n in sorted(before.keys() & after.keys()) if before[n] != after[n]])
    for name, value in [('files_sha256', after), ('diff_20261017b', changes)]:
        (OUT/(name+'.json')).write_text(json.dumps(value, indent=2), encoding='utf-8')
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    receipt = dict(zip=str(archive), sha256=digest, files=len(after), runtime_files=runtime,
        changes={k: len(v) for k, v in changes.items()})
    (OUT/'archive_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (OUT/'SHA256SUMS.txt').write_text(f'{digest}  {archive.name}\n', encoding='ascii')
    extract_smoke_copy(archive, Path(_public_dev_text('<work>/build_20261019a_stage/smoke_copy_r2')))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
