"""評価動画を除外し、陽性数とモデル構造を受領票に残すUI再学習。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from scripts.compare_image_free_all import local_path, read_config, save_json
from scripts.obs_priority_guard import ObsPriorityGuard
from scripts.train_image_free_all import teachers
from src.image_free_ui_ensemble import UIEnsemble, task_features

EXCLUDED_IDS = {'c68', 'video_c68', 'video_d01', 'video_d02', 'video_d04'}
EXCLUDED_YOUTUBE = {'DBfHe3dA244', 'ok1fKDR_6Ho', 'hqIAMefbeyk', 'sLaZharJBvI'}
TASKS = ('death_0', 'death_1', 'match_end', 'ojama', 'chain')
TREE_COUNT = 64
SEED = 20261008
MIN_POSITIVES = 50
MIN_FEATURES = 8
SAMPLE_SECONDS = 2.
NEIGHBOR_SECONDS = (-0.3, -0.15, 0.15, 0.3)
TEACHER_CHUNK_FRAMES = 32


def validate_sources(config: dict) -> None:
    """別名でも既知の評価動画を学習へ混ぜない。"""
    evaluation = {Path(local_path(v['path'])).resolve() for v in config['evaluation_videos']}
    for video in config['videos']:
        if (video['id'] in EXCLUDED_IDS or video.get('youtube_id') in EXCLUDED_YOUTUBE
                or Path(local_path(video['path'])).resolve() in evaluation):
            raise ValueError(f"評価動画の学習利用は禁止: {video['id']}")
        if not any(tier in video['title'] for tier in ('マスター', 'チャレンジャー', 'S級', 'A級')):
            raise ValueError(f"ティア未確認: {video['id']}")


def collect_frame(frame: np.ndarray, readers: dict) -> list[tuple]:
    from src.chain_count_ocr import _crop_search_roi
    from src.match_end_detector import DEFAULT_NCC_THRESHOLD
    from src.ojama_warning import P1_BOARD_X, P2_BOARD_X, CELL_COUNT
    rows = []
    for idx in range(2):
        key = f'death_{idx}'
        label = readers['death']._score(frame, None, idx) >= DEFAULT_NCC_THRESHOLD
        rows.append((key, task_features(key, frame), label, None))
    ending = readers['match_end'].detect(frame)
    rows.append(('match_end', task_features('match_end', frame),
                 ending.template_name if ending.detected else None, None))
    for board_x in (P1_BOARD_X, P2_BOARD_X):
        for idx in range(CELL_COUNT):
            cell = readers['ojama']._extract_cell(frame, board_x, idx)
            rows.append(('ojama', task_features('ojama', cell), readers['ojama']._match_templates(cell), None))
    for side in ('1P', '2P'):
        result = readers['chain'].read_side(frame, side)
        roi = _crop_search_roi(frame, side)
        rows.append(('chain', task_features('chain', roi), result.chain_count, result.location))
    return rows


def collect_video(video: dict, output: Path, readers: dict, guard: Any) -> list:
    cache = output / (video['id'] + '.teacher')
    cache.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(video['path'], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    if count <= 0 or fps <= 0:
        raise ValueError(f"動画を開けません: {video['path']}")
    manifest = cache/'checkpoint.json'
    state = json.loads(manifest.read_text()) if manifest.exists() else dict(video=video,
        pending=list(range(0, count, max(1, round(fps*SAMPLE_SECONDS)))), frames=[], chunks=[], classes={})
    if state['video'] != video:
        raise ValueError('教師キャッシュの動画情報が不一致')
    while state['pending']:
        rows = collect_chunk(cap, state, fps, count, readers, guard)
        name = f"{len(state['chunks']):05d}.joblib"
        temporary = cache/(name+'.tmp')
        joblib.dump(rows, temporary, compress=3)
        temporary.replace(cache/name)
        state['chunks'].append(name)
        save_json(manifest, state)
        save_json(output/f"training.{video['id']}.json", dict(video=video['id'], frames=len(state['frames']),
                                             pending=len(state['pending']), classes=state['classes']))
        print(video['id'], len(state['frames']), len(state['pending']), flush=True)
    cap.release()
    return [row for name in state['chunks'] for row in joblib.load(cache/name)]


def collect_chunk(cap: Any, state: dict, fps: float, count: int, readers: dict, guard: Any) -> list:
    pending, scanned, rows = set(state['pending']), set(state['frames']), []
    for _ in range(min(TEACHER_CHUNK_FRAMES, len(pending))):
        index = min(pending)
        pending.remove(index)
        guard.checkpoint()
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"復号失敗: {state['video']['id']}:{index}")
        observations = collect_frame(cv2.resize(frame, (1920, 1080)), readers)
        rows.extend(observations)
        scanned.add(index)
        for key, _, label, _ in observations:
            counts = state['classes'].setdefault(key, {})
            target = json.dumps(label)
            counts[target] = counts.get(target, 0) + 1
        if index % max(1, round(fps*SAMPLE_SECONDS)) == 0 and any(r[2] for r in observations[:3]):
            pending.update(i for delta in NEIGHBOR_SECONDS
                           if 0 <= (i := index+round(delta*fps)) < count and i not in scanned)
    state.update(pending=sorted(pending), frames=sorted(scanned))
    return rows


def fit_task(key: str, rows: list, expected_classes: list | None = None,
             color: bool = False) -> tuple[UIEnsemble, dict]:
    data = np.stack([row[1] for row in rows])
    labels = [json.dumps(row[2]) for row in rows]
    counts = dict(Counter(labels))
    # 予告のNoneはHSVへ渡す棄却クラス。希少クラスの増幅で棄却を潰さない。
    class_weight = None if key == 'ojama' else 'balanced'
    classifier = ExtraTreesClassifier(n_estimators=TREE_COUNT, random_state=SEED,
        class_weight=class_weight, max_features='sqrt', min_samples_leaf=2, n_jobs=1).fit(data, labels)
    coordinates = None
    if key == 'chain':
        positive = [i for i, row in enumerate(rows) if row[3] is not None]
        coordinates = ExtraTreesRegressor(n_estimators=TREE_COUNT, random_state=SEED,
            min_samples_leaf=2, n_jobs=1).fit(data[positive], [rows[i][3] for i in positive])
    used = set(int(f) for tree in classifier.estimators_ for f in tree.tree_.feature if f >= 0)
    depths = [tree.get_depth() for tree in classifier.estimators_]
    positive_count = sum(n for label, n in counts.items() if label not in ('null', 'false'))
    negative_count = len(rows) - positive_count
    passed = (min(positive_count, negative_count) >= MIN_POSITIVES
              and len(used) >= MIN_FEATURES and min(depths) > 1)
    missing = [label for label in expected_classes or [] if counts.get(json.dumps(label), 0) < MIN_POSITIVES]
    receipt = dict(samples=len(rows), classes=counts, positives=positive_count,
        negatives=negative_count, depths=depths, used_features=len(used),
        class_weight=class_weight, underrepresented_classes=missing, coverage_passed=passed and not missing)
    return UIEnsemble(key, classifier, coordinates, color=color), receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--collect-only', choices=('video_d03', 'video_d05', 'video_d06', 'video_d07'))
    args = parser.parse_args()
    config = read_config(args.config)
    validate_sources(config)
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(config['asset_root'])
    os.environ['PUYO_IMAGE_FREE_ALL'] = '0'
    cv2.setNumThreads(1)
    guard, readers, rows = ObsPriorityGuard(), teachers(), []
    if args.collect_only:
        video = next(video for video in config['videos'] if video['id'] == args.collect_only)
        collect_video(video, output, readers, guard)
        return
    for video in config['videos']:
        rows.extend(collect_video(video, output, readers, guard))
    model = joblib.load(config['base_ui_model'])
    receipt = dict(sources=config['videos'], evaluation_excluded=config['evaluation_videos'], tasks={})
    from src.ojama_warning import TEMPLATE_FILES
    receipt['ojama_teacher_classes'] = list(readers['ojama']._templates)
    receipt['unobservable_teacher_classes'] = sorted(set(TEMPLATE_FILES)-set(readers['ojama']._templates))
    receipt['source_frames'] = {v['id']: len(json.loads(
        (output/(v['id']+'.teacher')/'checkpoint.json').read_text())['frames']) for v in config['videos']}
    receipt['legacy_tasks_retained'] = sorted(set(model['tasks'])-set(TASKS))
    for key in TASKS:
        guard.checkpoint(force=True)
        expected = list(readers['ojama']._templates) if key == 'ojama' else None
        print('fit', key, flush=True)
        model['tasks'][key], receipt['tasks'][key] = fit_task(key, [r for r in rows if r[0] == key], expected)
        save_json(output/'receipt.json', receipt)
    joblib.dump(model, output/'ui.joblib', compress=3)
    if not all(task['coverage_passed'] for task in receipt['tasks'].values()):
        raise RuntimeError('陽性または複数特徴の被覆不足。比較の合格には進めません')


if __name__ == '__main__':
    main()
