"""画像非同梱 UI 候補の教師採取と学習。画像は保存しない。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from scripts.compare_image_free_digits import cv2, np, joblib, fit, samples
from scripts.obs_priority_guard import ObsPriorityGuard
from src.image_free_ui import features, StructuredModel, MODEL_SCHEMA

SAMPLES_PER_VIDEO = 64
CONFIDENCE_FIELDS = ('score', 'confidence', 'score_1p', 'score_2p')
DETECTION_THRESHOLD = 0.55
SCORE_EPSILON = 0.01


def teachers() -> dict:
    from src.match_end_detector import MatchEndDetector
    from src.telop_detector import TelopDetector
    from src.win_panel import WinPanelDetector
    from src.score_zero import ScoreZeroDetector
    from src.chain_count_ocr import ChainCountOcr
    from src.ojama_warning import OjamaWarningDetector
    from src.exchange_event_terminal import ObservedDeathDetector
    from src.review_terminal_outcome import _load_templates
    root = Path('models/ui_templates')
    return dict(match_end=MatchEndDetector.load_default(), telop=TelopDetector.load_default(),
                panel=WinPanelDetector.load_default(), zero=ScoreZeroDetector.load_default(),
                chain=ChainCountOcr.load_default(), ojama=OjamaWarningDetector(),
                death=ObservedDeathDetector(), review=_load_templates(root),
                review_crop=_load_templates(root, yatta_logo_only=True))


def observations(frame: np.ndarray, readers: dict) -> list[tuple]:
    from src.chain_count_ocr import _crop_search_roi
    from src.ojama_warning import P1_BOARD_X, P2_BOARD_X, CELL_COUNT
    from src.review_terminal_outcome import _template_score
    data, full = [], features(frame)
    for key in ('match_end', 'telop', 'zero'):
        data.append((key, full, asdict(readers[key].detect(frame))))
    panel = readers['panel'].detect(frame)
    data.append(('panel', full, dict(present=panel.present, score=panel.score)))
    for side in ('1P', '2P'):
        roi = _crop_search_roi(frame, side)
        data.append(('chain', features(roi, False), asdict(readers['chain'].read_side(frame, side))))
    for board_x in (P1_BOARD_X, P2_BOARD_X):
        for idx in range(CELL_COUNT):
            cell = readers['ojama']._extract_cell(frame, board_x, idx)
            data.append(('ojama', features(cell, False), readers['ojama']._match_templates(cell)))
    for idx in range(2):
        score = readers['death']._score(frame, None, idx)
        data.append((f'death_{idx}', full, decision_score(score)))
    for mode in ('review', 'review_crop'):
        for name in ('match_end_batan', 'match_end_yatta'):
            for side in ('p1', 'p2'):
                score = _template_score(frame, readers[mode], name, side)
                data.append((f'{mode}_{name}_{side}', full, decision_score(score)))
    return data


def decision_score(score: float) -> float:
    """閾値の上下を直接学習し、回帰による採否反転を避ける。"""
    return DETECTION_THRESHOLD + (SCORE_EPSILON if score >= DETECTION_THRESHOLD else -SCORE_EPSILON)


def fit_task(rows: list, guard: ObsPriorityGuard) -> StructuredModel:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    data = np.stack([r[0] for r in rows])
    targets = [r[1] for r in rows]
    scalar = not isinstance(targets[0], dict)
    keys = [] if scalar else [k for k in CONFIDENCE_FIELDS if k in targets[0]]
    labels = [json.dumps(t if scalar else {k: v for k, v in t.items() if k not in keys},
                         sort_keys=True) for t in targets]
    guard.checkpoint(force=True)
    classifier = DecisionTreeClassifier(random_state=20261007).fit(data, labels)
    regressor = None
    if keys:
        guard.checkpoint(force=True)
        regressor = DecisionTreeRegressor(random_state=20261007).fit(data, [[t[k] for k in keys] for t in targets])
    return StructuredModel(classifier, regressor, keys, scalar)


def train_ui(config: dict, output: Path, guard: ObsPriorityGuard) -> None:
    readers, tasks = teachers(), {}
    for video in config['videos']:
        checkpoint = output / (video['id'] + '.teacher.joblib')
        if checkpoint.exists():
            rows = joblib.load(checkpoint)
        else:
            cap = cv2.VideoCapture(video['path'])
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            rows = []
            for index in np.linspace(0, count - 1, SAMPLES_PER_VIDEO, dtype=int):
                guard.checkpoint()
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'教師動画を読めません: {video}, {index}')
                rows.extend(observations(cv2.resize(frame, (1920, 1080)), readers))
                progress(output, 'teacher', video=video['id'], frame=int(index), total=count)
            cap.release()
            joblib.dump(rows, checkpoint, compress=3)
        for key, feature, target in rows:
            tasks.setdefault(key, []).append((feature, target))
    models = {}
    for key, rows in tasks.items():
        progress(output, 'fit_ui', task=key, samples=len(rows))
        models[key] = fit_task(rows, guard)
    joblib.dump(dict(schema=MODEL_SCHEMA, tasks=models), output / 'ui.joblib', compress=3)


def progress(output: Path, stage: str, **values: object) -> None:
    values.update(stage=stage)
    temporary = output / 'training.tmp'
    temporary.write_text(json.dumps(values, ensure_ascii=False), encoding='utf-8')
    temporary.replace(output / 'training.json')
    print(json.dumps(values, ensure_ascii=False), flush=True)


def refine_digits(config: dict, output: Path, guard: ObsPriorityGuard) -> None:
    from src.score_ocr import ScoreOcr
    from src.image_free_digit import ImageFreeScoreOcr
    teacher = ScoreOcr.load_default()
    cap = cv2.VideoCapture(config['videos'][0]['path'])
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # 原票の既知不一致と、ゼロの背景変動を連続近傍で補う。独立評価とは呼ばない。
    times = np.unique(np.concatenate((np.linspace(1, duration-1, 240)[:144],
        np.arange(121.5, 122.2, 1/60), np.array([2119.091, 2119.191, 2119.291, 2119.391, 2119.491]))))
    data = samples(cap, teacher, times, guard)
    model = joblib.load(config['digit_model'])
    for key, column, regression in (('labels', 1, False), ('confidence', 2, True), ('multiply', 3, True)):
        progress(output, 'fit_digits', task=key)
        model[key] = fit(data[0], data[column], regression, guard)
    path = output / 'digit_student.joblib'
    joblib.dump(model, path, compress=3)
    cap.set(cv2.CAP_PROP_POS_MSEC, 121817)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError('既知ゼロフレームを読めません')
    old, new = teacher.read_side_detail(frame, '2P'), ImageFreeScoreOcr(path).read_side_detail(frame, '2P')
    progress(output, 'zero_regression', old=old, new=new)
    (output / 'zero_regression.json').write_text(json.dumps(dict(old=old, new=new)), encoding='utf-8')
    if old[0] != new[0]:
        raise RuntimeError('既知ゼロ欠測が未修正です')
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    from scripts.compare_image_free_all import read_config
    config = read_config(args.config)
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(config['asset_root'])
    os.environ['PUYO_IMAGE_FREE_ALL'] = '0'
    cv2.setNumThreads(1)
    guard = ObsPriorityGuard()
    guard.checkpoint(force=True)
    if not (output / 'digit_student.joblib').exists():
        refine_digits(config, output, guard)
    if not (output / 'ui.joblib').exists():
        train_ui(config, output, guard)
    progress(output, 'trained')


if __name__ == '__main__':
    main()
