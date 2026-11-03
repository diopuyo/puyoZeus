"""この作業で新規作成した未起動ステージだけを最新import閉包で再生成する。"""

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
from build_bundle import copy_app, write_manifest, precompile


def main() -> None:
    bundle = Path(_public_dev_text('<work>/build_20261019a_stage/PuyoLive'))
    assert bundle.resolve().parent == Path(_public_dev_text('<work>/build_20261019a_stage')).resolve()
    copy_app(bundle, Path(_public_dev_text('<work>/assets/baseline_b/app')))
    files = [p for p in (bundle/'app').rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and p.name != 'MANIFEST.json']
    write_manifest(bundle, files)
    precompile(bundle/'python', bundle/'app')


if __name__ == '__main__':
    main()
