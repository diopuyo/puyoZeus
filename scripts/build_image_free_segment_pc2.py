"""検証済みの分割比較を、旧PC2一式の依存資産で移送可能にする。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tarfile

from scripts.image_free_cache import file_sha
from scripts.image_free_speed_launch import verify

INPUT_KEYS = ('mask_model', 'digit_model', 'b3_model', 'b3_table')
MODEL_FILES = ('ui.joblib', 'digit_student.joblib')
EVIDENCE_FILES = ('quality_gate.json', 'split_proof.json', 'segment_plan.json')
DEPENDENCY_FILES = ('requirements-pc1.txt', 'requirements-pc2.txt')


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def copy_assets(previous: Path, run: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('assets', 'wheels'):
        shutil.copytree(previous/name, destination/name, symlinks=True)
    shutil.copytree(run/'source', destination/'source',
                    ignore=shutil.ignore_patterns('__pycache__'))
    for name in MODEL_FILES + EVIDENCE_FILES:
        shutil.copyfile(run/name, destination/name)
    for name in DEPENDENCY_FILES:
        shutil.copyfile(previous/name, destination/name)


def portable_config(config: dict, run: Path, destination: Path) -> dict:
    plan = read(run/'segment_plan.json')
    result = dict(config, output='.', host='pc2', asset_root='assets', old_cache='old_cache')
    (destination/'inputs').mkdir()
    for key in INPUT_KEYS:
        source = Path(config[key])
        target = Path('inputs')/(key+source.suffix)
        shutil.copyfile(source, destination/target)
        result[key] = str(target)
    # 同じ動画の複数区間を、一つの入力ファイルへ向ける。
    (destination/'videos').mkdir()
    result['videos'], videos = [], {}
    for segment in (s for s in plan['segments'] if s['host'] == 'pc2'):
        target = Path('videos')/(segment['video_id']+'.mp4')
        if segment['video_id'] not in videos:
            shutil.copyfile(segment['path'], destination/target)
            videos[segment['video_id']] = dict(path=str(target), sha256=file_sha(destination/target))
        result['videos'].append(dict(segment, path=str(target)))
    result['cache_asset_files'] = [str(Path('assets')/Path(p).relative_to(config['asset_root']))
                                    for p in config['cache_asset_files']]
    (destination/'video_manifest.json').write_text(json.dumps(videos, indent=2))
    return result


def entrypoints(destination: Path, config: dict) -> None:
    (destination/'config.template.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    prepare = '''import json
from pathlib import Path
root=Path(__file__).resolve().parent
config=json.loads((root/'config.template.json').read_text())
for key in ('output','asset_root','old_cache','mask_model','digit_model','b3_model','b3_table'):
    config[key]=str((root/config[key]).resolve())
config['cache_asset_files']=[str((root/p).resolve()) for p in config['cache_asset_files']]
for video in config['videos']:
    video['path']=str((root/video['path']).resolve())
(root/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
'''
    launch = '''#!/bin/bash
set -eu
cd "$(dirname "$0")"
: "${PUYO_PYTHON:?Python 3.12環境の絶対パスが必要です}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/source"
"$PUYO_PYTHON" prepare.py
exec "$PUYO_PYTHON" -m scripts.image_free_speed_launch --config config.json --source source --gate quality_gate.json --workers 24 --execute verified-full-comparison
'''
    (destination/'prepare.py').write_text(prepare, encoding='utf-8')
    (destination/'run_pc2.sh').write_text(launch, encoding='utf-8')


def build(run: Path, previous: Path, destination: Path, instructions: Path) -> dict:
    config = read(run/'config.json')
    verify(config, run/'source', read(run/'quality_gate.json'))
    archive_path = destination.with_suffix('.tar')
    if destination.exists() or archive_path.exists():
        raise FileExistsError('新しいPC2一式の名前が必要です')
    copy_assets(previous, run, destination)
    entrypoints(destination, portable_config(config, run, destination))
    shutil.copyfile(instructions, destination/'PC2手順.md')
    with tarfile.open(archive_path, 'w') as archive:
        archive.add(destination, arcname=destination.name)
    digest = file_sha(archive_path)
    archive_path.with_suffix('.tar.sha256').write_text(f'{digest}  {archive_path.name}\n')
    return dict(path=str(archive_path), sha256=digest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--instructions', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run, args.previous, args.destination, args.instructions)))


if __name__ == '__main__':
    main()
