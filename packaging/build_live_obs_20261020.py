"""本人用OBS試験の未起動ステージ。同一ビルダー・除外規則を使用する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import json
import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
from build_bundle import build, write_manifest, precompile
from build_live_v49f import sha

STAGE = Path('D:/puyo_analyzer/packaging/build_20261020a_work')
BUNDLE = STAGE/'PuyoLive'
OLD = Path(_public_dev_text('<work>/build_20261019a_stage/PuyoLive'))
OUT = Path('D:/puyo_analyzer/packaging/build_20261020a_release')
ASSETS = Path(_public_dev_text('<work>/assets'))
WARNING = '本人試験専用・配布禁止。puyoai-private由来のamaを含むため、第三者へ渡さないでください。\n'


def settings() -> dict:
    values = json.loads((OLD/'puyo_live.json').read_text(encoding='utf-8'))
    values.update(b3_m3=True, live_ama_plain=True, b3_time_gate=True,
        b3_skip_icon_reading=False, live_reference_times_only=True,
        live_near_future_cache=True,live_exact_connectivity=True,live_exact_native=True,
        overlay=dict(layout='l',scale=75,font='small',debug_panel=False,details=False,mascot=False))
    return values


def finish() -> None:
    for name in ('ama_plain', 'v49f'):
        target = BUNDLE/'app/models'/name
        target.mkdir(exist_ok=True)
        if name == 'v49f':
            shutil.copytree(OLD/'app/models/v49f', target, dirs_exist_ok=True)
        shutil.copytree(ROOT/'models'/name, target, dirs_exist_ok=True)
    shutil.copy2(ROOT/'models/b3_m3/b3_ledger_experiment.json', BUNDLE/'app/models/b3/b3_ledger_experiment.json')
    (BUNDLE/'puyo_live.json').write_text(json.dumps(settings(), ensure_ascii=False, indent=2), encoding='utf-8')
    (BUNDLE/'USER_ONLY_DO_NOT_DISTRIBUTE.txt').write_text(WARNING, encoding='utf-8-sig')
    readme = BUNDLE/'README_ja.txt'
    readme.write_text(WARNING+'\n'+readme.read_text(encoding='utf-8-sig'), encoding='utf-8-sig')
    precompile(BUNDLE/'python', BUNDLE/'app')
    files = [p for p in (BUNDLE/'app').rglob('*') if p.is_file() and p.name != 'MANIFEST.json']
    write_manifest(BUNDLE, files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-own-stage',action='store_true')
    args = parser.parse_args()
    assert args.resume_own_stage or not BUNDLE.exists(), '既存ステージを上書きしない'
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'build_receipt.json').exists()
    if not args.resume_own_stage:
        shutil.copytree(OLD/'python', BUNDLE/'python')
    receipt = build(ASSETS/'packaging', ASSETS/'baseline_b/app', False, True, with_onnx=True,
        onnx_cache=OLD/'app/models/onnx', b3_model=ROOT/'models/b3_m3/model.joblib',
        b3_timing_table=OLD/'app/models/b3/timing_table.json', bundle_dir=BUNDLE)
    native = next((BUNDLE/'python/Lib/site-packages/puyo_core').glob('*.pyd'))
    shutil.copy2(next((OLD/'python/Lib/site-packages/puyo_core').glob('*.pyd')), native)
    finish()
    receipt.update(settings=settings(), native_sha256=sha(native),
        m3_sha256=sha(BUNDLE/'app/models/b3/model.joblib'),
        ama_sha256=sha(BUNDLE/'app/models/ama_plain/ama_reference.exe'))
    (OUT/'build_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('Stage ready', BUNDLE, flush=True)


if __name__ == '__main__':
    main()
