"""検証段階ごとに新しい出力先へソースとモデルを固定する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.compare_image_free_all import read_config, save_json

SOURCE_DIRS = ('src', 'scripts')
MODELS = ('ui.joblib', 'digit_student.joblib')
INPUT_KEYS = ('mask_model', 'b3_model', 'b3_table')
START_MARGIN_SEC = 2.
END_MARGIN_SEC = 2.
C68_START_MARGIN_SEC = 4.
C68_END_MARGIN_SEC = 3.


def freeze_source(previous: Path, destination: Path, repository: Path) -> dict:
    """既存の資産配置を保ち、今回のPythonソースを固定する。"""
    shutil.copytree(previous/'source', destination/'source', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    for directory in SOURCE_DIRS:
        for path in (repository/directory).rglob('*.py'):
            target = destination/'source'/path.relative_to(repository)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    manifest = {str(path.relative_to(destination/'source')): hashlib.sha256(path.read_bytes()).hexdigest()
                for directory in SOURCE_DIRS for path in sorted((destination/'source'/directory).rglob('*.py'))}
    save_json(destination/'source_manifest.json', manifest)
    shutil.copy2(repository/'scripts/image_free_parallel.py', destination/'parallel.py')
    return manifest


def short_intervals(config: dict, receipts: Path) -> list[dict]:
    """既存の実測試合境界を根拠に、開始を含む2試合を選ぶ。"""
    import cv2
    videos = []
    for original in config['videos']:
        video = dict(original)
        cap = cv2.VideoCapture(video['path'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if video['id'] == 'c68':
            mapping = Path(video['path']).parents[1]/'game_mapping.json'
            games = json.loads(mapping.read_text())
            start = games[0]['source_start_sec']-C68_START_MARGIN_SEC
            end = games[1]['source_end_sec']+C68_END_MARGIN_SEC
            evidence = str(mapping)
        else:
            receipt = json.loads((receipts/(video['id']+'.json')).read_text())
            path = Path(receipt['attempt'])/'record/lean_event_observations_v1.json'
            boundaries = json.loads(path.read_text())['accepted_boundary_evidence']
            start = boundaries[0]['observed_sec']-START_MARGIN_SEC
            end = boundaries[2]['observed_sec']+END_MARGIN_SEC
            evidence = str(path)
        video.update(start_frame=max(0, int(start*fps)), end_frame=int(end*fps),
                     interval_evidence=evidence, intended_matches=2)
        videos.append(video)
    return videos


def prepare(config_path: Path, destination: Path, short: bool, gate: Path | None) -> None:
    config = read_config(config_path)
    origin = Path(config['output'])
    if not short:
        if gate is None or not json.loads(gate.read_text()).get('gate_passed'):
            raise ValueError('全区間準備には段3の合格受領票が必要です')
        accepted = json.loads(gate.read_text())
        for name in MODELS:
            if hashlib.sha256((origin/name).read_bytes()).hexdigest() != accepted['model_sha256'][name]:
                raise ValueError('合格後にモデルが変わっています')
        for key in INPUT_KEYS:
            if hashlib.sha256(Path(config[key]).read_bytes()).hexdigest() != accepted['input_sha256'][key]:
                raise ValueError('合格後に評価モデル・設定が変わっています')
    if not all((origin/name).is_file() for name in MODELS):
        raise FileNotFoundError('学習済みモデルが揃っていません')
    destination.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[1]
    manifest = freeze_source(Path(config['previous_output']), destination, repository)
    if not short and manifest != accepted['source_manifest']:
        raise ValueError('合格後にソースが変わっています')
    for name in MODELS:
        shutil.copy2(origin/name, destination/name)
    freeze_inputs(config, destination)
    config['output'] = str(destination.resolve())
    if short:
        receipts = Path('/mnt/d/puyo_analyzer/wt_b2/data/verify/recollect_v2_newlist/receipts')
        config['videos'] = short_intervals(config, receipts)
    save_json(destination/'config.json', config)
    if gate is not None:
        shutil.copy2(gate, destination/'quality_gate.json')
    python = _public_dev_text('<repo>/venv/bin/python')
    command = ('#!/bin/bash\nset -eu\ncd "$(dirname "$0")"\n'
               'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1\n'
               f'exec {python} parallel.py --config config.json --source source --workers 4\n')
    (destination/'run_parallel.sh').write_text(command, encoding='utf-8')


def freeze_inputs(config: dict, destination: Path) -> None:
    """稼働アプリのモデル更新が比較の途中に混入しないようコピーする。"""
    (destination/'inputs').mkdir()
    for key in INPUT_KEYS:
        source = Path(config[key])
        target = destination/'inputs'/(key+source.suffix)
        shutil.copy2(source, target)
        config[key] = str(target.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--gate', type=Path)
    args = parser.parse_args()
    prepare(args.config, args.destination, args.short, args.gate)


if __name__ == '__main__':
    main()
