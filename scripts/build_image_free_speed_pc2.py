"""既存PC2一式の資産を流用し、未合格時は起動不能な新版を作る。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tarfile

from scripts.image_free_cache import file_sha

DIRECTORIES = ('assets', 'inputs', 'videos', 'wheels', 'source')
FILES = ('ui.joblib', 'digit_student.joblib', 'requirements-pc1.txt', 'requirements-pc2.txt')


def freeze(previous: Path, destination: Path, source: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    for name in DIRECTORIES:
        shutil.copytree(previous/name, destination/name, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__'), copy_function=shutil.copyfile)
    for name in FILES:
        shutil.copyfile(previous/name, destination/name)
    for base in ('scripts', 'src'):
        for path in (source/base).rglob('*.py'):
            target = destination/'source'/path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    manifest = {str(p.relative_to(destination/'source')): file_sha(p)
                for base in ('scripts', 'src') for p in (destination/'source'/base).rglob('*.py')}
    (destination/'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    return json.loads((previous/'config.template.json').read_text())


def entrypoints(destination: Path, config: dict) -> None:
    config['old_cache'] = 'old_cache'
    (destination/'config.template.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    # 合格を推定しない。候補確定後に合格SHAへ固定した一式を再生成する。
    (destination/'quality_gate.json').write_text(json.dumps(dict(gate_passed=False,
        reason='fix1段3未合格・区間分割不一致。全区間比較は未起動。'), ensure_ascii=False))
    prepare = '''import json
from pathlib import Path
root=Path(__file__).resolve().parent
config=json.loads((root/'config.template.json').read_text())
for key in ('output','asset_root','old_cache','mask_model','digit_model','b3_model','b3_table'):
    config[key]=str((root/config[key]).resolve())
for video in config['videos']:
    video['path']=str((root/video['path']).resolve())
(root/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
'''
    (destination/'prepare.py').write_text(prepare, encoding='utf-8')
    launch = '''#!/bin/bash
set -eu
cd "$(dirname "$0")"
: "${PUYO_PYTHON:?Python 3.12環境の絶対パスが必要です}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/source"
"$PUYO_PYTHON" prepare.py
exec "$PUYO_PYTHON" -m scripts.image_free_speed_launch --config config.json --source source --gate quality_gate.json --workers 24 --execute verified-full-comparison
'''
    (destination/'run_pc2.sh').write_text(launch, encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--instructions', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    config = freeze(args.previous, args.destination, source)
    entrypoints(args.destination, config)
    shutil.copy2(args.instructions, args.destination/'PC2手順.md')
    target = args.destination.with_suffix('.tar')
    if target.exists():
        raise FileExistsError(target)
    with tarfile.open(target, 'w') as archive:
        archive.add(args.destination, arcname=args.destination.name)
    digest = file_sha(target)
    target.with_suffix('.tar.sha256').write_text(f'{digest}  {target.name}\n')
    print(json.dumps(dict(path=str(target), sha256=digest)), flush=True)


if __name__ == '__main__':
    main()
