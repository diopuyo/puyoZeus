"""旧ZIPを保持し、今回の検証専用コピーだけへ修正を反映する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_bundle import precompile, write_manifest

ROOT = Path(__file__).resolve().parents[1]
COPY = Path(_public_dev_text('<work>/build_20261020a_stage/zip_smoke_copy_fast/PuyoLive'))


def main() -> None:
    assert COPY.resolve().is_relative_to(Path(_public_dev_text('<work>/build_20261020a_stage')).resolve())
    changed = []
    for relative in sys.argv[1:]:
        source, target = ROOT/relative, COPY/'app'/relative
        assert source.is_file() and target.resolve().is_relative_to((COPY/'app').resolve())
        shutil.copy2(source,target)
        changed.append(target)
    if changed:
        precompile(COPY/'python',*changed)
    write_manifest(COPY,[p for p in (COPY/'app').rglob('*') if p.is_file()
        and p.name != 'MANIFEST.json' and 'device_calibration' not in p.parts])
    print([str(p) for p in changed],flush=True)


if __name__ == '__main__':
    main()
