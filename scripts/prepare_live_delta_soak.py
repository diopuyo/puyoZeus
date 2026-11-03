"""既存配布版を変更せず、l基準と差分候補の内部試験台をCに組み立てる。"""
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
import sys

REPO = Path(__file__).resolve().parents[1]
OUT = Path(_public_dev_text('<work>/delta'))
ASSETS = Path(_public_dev_text('<work>/assets'))
sys.path[:0] = [str(REPO), str(REPO / 'packaging')]


def prepare(mode: str) -> None:
    from scripts.bench_live_delta import check_resources
    import build_bundle as builder
    from src.phase_j.launcher_manifest import verify_manifest
    check_resources()
    bundle = OUT / ('soak_' + mode) / 'PuyoLive'
    if bundle.exists():
        raise FileExistsError(bundle)
    source = OUT / 'baseline' if mode == 'off' else REPO
    if mode == 'off':
        shutil.copy2(REPO / 'LICENSE', source / 'LICENSE')
    builder.ROOT = source
    shutil.copytree('C:/PuyoLive_20261012a/PuyoLive/python', bundle / 'python')
    check_resources()
    result = builder.build(ASSETS / 'packaging', ASSETS / 'baseline_b/app', False, True,
        with_onnx=True, onnx_cache=ASSETS / 'local_onnx',
        b3_model=ASSETS / 'baseline_b/app/models/b3/model.joblib',
        b3_timing_table=ASSETS / 'baseline_b/app/models/b3/timing_table.json', bundle_dir=bundle)
    report = verify_manifest(bundle / 'app', bundle / 'app/MANIFEST.json', bundle / 'python/Lib/site-packages')
    assert report.ok, report.summary()
    settings = json.loads(Path(_public_dev_text('<work>/native_fallback/build.json')).read_text())['settings']
    run = OUT / ('run_' + mode)
    settings.update(device_index=2, port=18823, output_dir=str(run / 'output'))
    settings.update(b3_model=str(bundle / 'app/models/b3/model.joblib'),
                    b3_timing_table=str(bundle / 'app/models/b3/timing_table.json'))
    if mode == 'on':
        settings.update(live_delta_cells=True, live_cnn_dedup=False)
    run.mkdir(parents=True)
    (run / 'puyo_live.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    (OUT / ('soak_build_' + mode + '.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    for mode in ('off', 'on'):
        prepare(mode)
