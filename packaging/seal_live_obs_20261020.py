"""本人用ZIPを新規作成し、前2配布との差分と除外規則を検収する。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bundle_archive import write_bundle_zip, extract_smoke_copy, is_runtime_file
from seal_live_v49f import inventory
from build_live_obs_20261020 import BUNDLE, OUT, STAGE, sha


def main() -> None:
    archive = write_bundle_zip(BUNDLE,OUT/'PuyoLive_20261020a_USER_ONLY_DO_NOT_DISTRIBUTE.zip')
    after, diffs = inventory(archive), {}
    assert not [n for n in after if is_runtime_file(n.removeprefix('PuyoLive/'))]
    for name, path in [('20261017b','build_20261017b/PuyoLive_20261017b.zip'),
                       ('20261019a','build_20261019a_release/PuyoLive_20261019a.zip')]:
        before = inventory(Path('D:/puyo_analyzer/packaging')/path)
        changes = dict(added=sorted(after.keys()-before.keys()),removed=sorted(before.keys()-after.keys()),
            changed=[n for n in sorted(before.keys()&after.keys()) if before[n]!=after[n]])
        (OUT/f'diff_{name}.json').write_text(json.dumps(changes,indent=2),encoding='utf-8')
        diffs[name] = {k:len(v) for k,v in changes.items()}
    digest = sha(archive)
    receipt = dict(zip=str(archive),sha256=digest,files=len(after),runtime_files=[],changes=diffs)
    (OUT/'files_sha256.json').write_text(json.dumps(after,indent=2),encoding='utf-8')
    (OUT/'archive_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (OUT/'SHA256SUMS.txt').write_text(f'{digest}  {archive.name}\n',encoding='ascii')
    extract_smoke_copy(archive,STAGE/'zip_smoke_copy')
    print(receipt,flush=True)


if __name__ == '__main__':
    main()
