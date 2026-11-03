"""10/17bと同じビルダーでOBS試験用の新規ステージを作る。完成ZIPでのみsmokeする。"""
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
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('D:/puyo_analyzer/packaging/build_20261019a_release')
NATIVE = Path('D:/puyo_analyzer/packaging/build_20261019a/native/puyo_core.pyd')
OLD = Path(_public_dev_text('<work>/build_20261017b_stage/smoke_copy/PuyoLive'))
BUNDLE = Path(_public_dev_text('<work>/build_20261019a_stage/PuyoLive'))
ASSETS = Path(_public_dev_text('<work>/assets'))


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
    from build_bundle import build, write_manifest, precompile
    assert not BUNDLE.exists(), '既存ビルドは上書きしない'
    shutil.copytree(OLD/'python', BUNDLE/'python')
    print('Python copied', flush=True)
    receipt = build(ASSETS/'packaging', ASSETS/'baseline_b/app', False, True, with_onnx=True,
        onnx_cache=OLD/'app/models/onnx', b3_model=OLD/'app/models/b3/model.joblib',
        b3_timing_table=OLD/'app/models/b3/timing_table.json', bundle_dir=BUNDLE)
    native = next((BUNDLE/'python/Lib/site-packages/puyo_core').glob('*.pyd'))
    shutil.copy2(NATIVE, native)
    target = BUNDLE/'app/models/v49f'
    target.mkdir()
    for source, name in (('exch_review_v310/policy_locked.json', 'policy.json'),
                         ('prefire_v44/model.json', 'pre_model.json'), ('prefire_v48/model.json', 'model.json')):
        shutil.copy2(ROOT/'docs/verification'/source, target/name)
    settings = json.loads((OLD/'puyo_live.json').read_text(encoding='utf-8'))
    settings.update(b3_async=True, live_v49f=True, live_stage_timing=True)
    (BUNDLE/'puyo_live.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    files = [p for p in (BUNDLE/'app').rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and p.name != 'MANIFEST.json']
    write_manifest(BUNDLE, files)
    precompile(BUNDLE/'python', BUNDLE/'app')
    receipt.update(settings=settings, native_sha256=sha(native),
                   base_model_sha256=sha(BUNDLE/'app/models/b3/model.joblib'))
    (OUT/'build_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('Stage ready', BUNDLE, flush=True)


if __name__ == '__main__':
    main()
