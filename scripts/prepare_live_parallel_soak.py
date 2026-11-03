"""m基準と候補の内部試験版をCへ作る。配布pと既存d〜mは変更しない。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'logs/live_parallel'
ASSETS = Path(_public_dev_text('<work>/assets'))
PYTHON = Path('C:/PuyoLive_20261012a/PuyoLive/python')
BASE_COMMIT = 'dc24ee44'


def archive_baseline() -> Path:
    """Gitの指定commitだけを別領域へ展開し、候補OFFをmの代用にしない。"""
    target = OUT/'baseline_m'
    if target.exists():
        assert (target/'BASE_COMMIT.txt').read_text() == BASE_COMMIT
        return target
    target.mkdir(exist_ok=False)
    archive = OUT/'baseline_m.zip'
    subprocess.run(['git', 'archive', '--format=zip', '-o', str(archive), BASE_COMMIT,
                    'src', 'scripts', 'packaging', 'config', 'LICENSE'], cwd=ROOT, check=True)
    with zipfile.ZipFile(archive) as saved:
        for name in saved.namelist():
            if not (target/name).resolve().is_relative_to(target.resolve()):
                raise ValueError('展開先が内部試験領域の外')
        saved.extractall(target)
    (target/'BASE_COMMIT.txt').write_text(BASE_COMMIT)
    return target


def prepare(mode: str, source: Path) -> None:
    """同梱Python・モデル・設定を揃え、各版のMANIFESTを照合する。"""
    from live_parallel_probe import guard
    import build_bundle as builder
    from src.phase_j.launcher_manifest import verify_manifest
    guard()
    bundle = OUT/f'soak_{mode}'/'PuyoLive'
    if bundle.exists():
        raise FileExistsError(bundle)
    assert bundle.resolve().is_relative_to(OUT.resolve())
    shutil.copytree(PYTHON, bundle/'python')
    builder.ROOT = source
    result = builder.build(ASSETS/'packaging', ASSETS/'baseline_b/app', False, True,
        with_onnx=True, onnx_cache=ASSETS/'baseline_b/app/models/onnx',
        b3_model=ASSETS/'baseline_b/app/models/b3/model.joblib',
        b3_timing_table=ASSETS/'baseline_b/app/models/b3/timing_table.json', bundle_dir=bundle)
    report = verify_manifest(bundle/'app', bundle/'app/MANIFEST.json', bundle/'python/Lib/site-packages')
    assert report.ok, report.summary()
    config = json.loads(Path(_public_dev_text('<work>/coalesce2/build.json')).read_text())['settings']
    config.update(device_index=2, port=18843, live_public_match_number=True,
        output_dir=str(OUT/f'camera_{mode}'/'output'),
        b3_model=str(bundle/'app/models/b3/model.joblib'),
        b3_timing_table=str(bundle/'app/models/b3/timing_table.json'))
    if mode == 'on':
        config.update(live_onnx_parallel=True, live_color_buffer=True)
    (bundle/'puyo_live.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    result.update(manifest_checked=report.checked, baseline_commit=BASE_COMMIT,
                  source=str(source), settings=config)
    (OUT/f'build_{mode}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    subprocess.run([str(bundle/'python/python.exe'), '-B', '-m', 'src.phase_j.launcher',
        '--config', str(bundle/'puyo_live.json'), '--require-manifest', '--check-only'],
        cwd=bundle, check=True, creationflags=0x4000)


def main() -> None:
    sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'packaging')]
    from live_parallel_probe import guard
    guard()
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    baseline = archive_baseline()
    prepare('off', baseline)
    prepare('on', ROOT)


if __name__ == '__main__':
    main()
