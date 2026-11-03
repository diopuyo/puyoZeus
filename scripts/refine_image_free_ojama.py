"""カラー照合の教師に合わせ、予告へBGR特徴を追加する。評価動画は使わない。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import joblib

from scripts.compare_image_free_all import read_config, save_json
from scripts.obs_priority_guard import ObsPriorityGuard
from scripts.train_image_free_fix import fit_task, validate_sources
from src.image_free_ui_ensemble import task_features

FRAME_STRIDE = 3
CHECKPOINT_SAMPLES = 32


def collect(video: dict, config: dict, remaining: bool = False) -> list:
    from src.ojama_warning import OjamaWarningDetector, P1_BOARD_X, P2_BOARD_X, CELL_COUNT
    output = Path(config['output'])
    checkpoint = output/(video['id']+'.teacher')/'checkpoint.json'
    all_frames = json.loads(checkpoint.read_text())['frames']
    frames = ([frame for index, frame in enumerate(all_frames) if index % FRAME_STRIDE]
              if remaining else all_frames[::FRAME_STRIDE])
    cache = output/(video['id']+('.color_extra' if remaining else '.color'))
    cache.mkdir(exist_ok=True)
    manifest = cache/'checkpoint.json'
    state = json.loads(manifest.read_text()) if manifest.exists() else dict(completed=0, chunks=[])
    rows = []
    cap = cv2.VideoCapture(video['path'], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    reader, guard = OjamaWarningDetector(use_cnn=False), ObsPriorityGuard()
    for position in range(state['completed'], len(frames)):
        guard.checkpoint()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[position])
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"復号失敗: {video['id']}:{frames[position]}")
        frame = cv2.resize(frame, (1920, 1080))
        for board_x in (P1_BOARD_X, P2_BOARD_X):
            for idx in range(CELL_COUNT):
                cell = reader._extract_cell(frame, board_x, idx)
                rows.append(('ojama', task_features('ojama', cell, True), reader._match_templates(cell), None))
        state['completed'] = position+1
        if state['completed'] % CHECKPOINT_SAMPLES == 0 or state['completed'] == len(frames):
            name = f"{len(state['chunks']):05d}.joblib"
            temporary = cache/(name+'.tmp')
            joblib.dump(rows, temporary, compress=3)
            temporary.replace(cache/name)
            state['chunks'].append(name)
            save_json(manifest, state)
            rows.clear()
            print(video['id'], state['completed'], len(frames), flush=True)
    cap.release()
    return [row for name in state['chunks'] for row in joblib.load(cache/name)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--collect-only')
    parser.add_argument('--all-frames', action='store_true')
    args = parser.parse_args()
    config = read_config(args.config)
    validate_sources(config)
    os.chdir(config['asset_root'])
    os.environ['PUYO_IMAGE_FREE_ALL'] = '0'
    cv2.setNumThreads(1)
    if args.collect_only:
        collect(next(v for v in config['videos'] if v['id'] == args.collect_only), config, args.all_frames)
        return
    from src.ojama_warning import OjamaWarningDetector
    rows = [row for video in config['videos'] for row in collect(video, config)]
    if args.all_frames:
        rows.extend(row for video in config['videos'] for row in collect(video, config, True))
    model = joblib.load(Path(config['output'])/'ui.joblib')
    expected = list(OjamaWarningDetector(use_cnn=False)._templates)
    model['tasks']['ojama'], receipt = fit_task('ojama', rows, expected, color=True)
    receipt.update(sources=config['videos'], frame_stride=1 if args.all_frames else FRAME_STRIDE,
                   color_features=True)
    suffix = '_all' if args.all_frames else ''
    save_json(Path(config['output'])/f'color{suffix}_receipt.json', receipt)
    if not receipt['coverage_passed']:
        raise RuntimeError('カラー予告の教師被覆が不足しています')
    joblib.dump(model, Path(config['output'])/f'ui_color{suffix}.joblib', compress=3)


if __name__ == '__main__':
    main()
