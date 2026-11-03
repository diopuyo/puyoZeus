"""画像を同梱しない OCR 候補を時間分離した動画標本で比較する（公開ゲート）。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from scripts.obs_priority_guard import ObsPriorityGuard
from src.image_free_digit import (ImageFreeScoreOcr, digit_features, UNKNOWN_LABEL,
                                  FEATURE_SIZE, MODEL_SCHEMA)
from src.score_ocr import ScoreOcr, _crop_score_roi, _crop_digit_cell

SAMPLES = 240
TRAIN_FRACTION = 0.6
TREE_COUNT = 48
MAX_DEPTH = 14
SEED = 20261007
SIDES = ('1P', '2P')


def frame_at(cap: cv2.VideoCapture, seconds: float) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, image = cap.read()
    if not ok:
        raise RuntimeError(f'動画読込失敗: {seconds}')
    return cv2.resize(image, (1920, 1080))


def samples(cap: cv2.VideoCapture, teacher: ScoreOcr, times: np.ndarray,
            guard: ObsPriorityGuard) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, labels, confidence, multiply = [], [], [], []
    for index, seconds in enumerate(times):
        guard.checkpoint()
        image = frame_at(cap, float(seconds))
        for side in SIDES:
            roi = _crop_score_roi(image, side)
            for cell_index in range(8):
                cell = _crop_digit_cell(roi, cell_index, side)
                label, conf = teacher._classify_digit(cell)
                features.append(digit_features(cell))
                labels.append(UNKNOWN_LABEL if label is None else label)
                confidence.append(conf)
                tpl = teacher._mult_template_gray
                gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                multiply.append(float(cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED).max()))
        if index % 20 == 0:
            print(f'教師標本 {index + 1}/{len(times)} t={seconds:.1f}', flush=True)
    return np.asarray(features), np.asarray(labels), np.asarray(confidence), np.asarray(multiply)


def fit(features: np.ndarray, labels: np.ndarray, regression: bool,
        guard: ObsPriorityGuard) -> object:
    cls = ExtraTreesRegressor if regression else ExtraTreesClassifier
    estimator = cls(n_estimators=1, max_depth=MAX_DEPTH, n_jobs=1,
                    random_state=SEED, warm_start=True)
    for count in range(1, TREE_COUNT + 1):
        guard.checkpoint()
        estimator.n_estimators = count
        started = time.monotonic()
        estimator.fit(features, labels)
        if time.monotonic() - started > 20:
            raise RuntimeError('学習単位が20秒を超過。OBS監視間隔を守るため中断')
    return estimator


def evaluate(cap: cv2.VideoCapture, times: np.ndarray, teacher: ScoreOcr,
             student: ImageFreeScoreOcr, output: Path, guard: ObsPriorityGuard) -> dict:
    counts = {name: dict(total=0, equal=0) for name in ('digits', 'scores', 'formulas')}
    coverage = dict(old_scores=0, new_scores=0, old_formulas=0, new_formulas=0)
    differences = []
    for seconds in times:
        guard.checkpoint()
        image = frame_at(cap, float(seconds))
        for side in SIDES:
            old = teacher.read_side_detail(image, side)
            new = student.read_side_detail(image, side)
            for a, b in zip(old[2], new[2]):
                counts['digits']['total'] += 1
                counts['digits']['equal'] += int(a == b)
            old_formula = teacher.read_formula_side(image, side, old[2], old[3])
            new_formula = student.read_formula_side(image, side, new[2], new[3])
            coverage['old_scores'] += int(old[0] is not None)
            coverage['new_scores'] += int(new[0] is not None)
            coverage['old_formulas'] += int(old_formula.valid)
            coverage['new_formulas'] += int(new_formula.valid)
            # 信頼度も比較記録へ残すが、式の同値判定では読取値とvalidを比較する。
            old_f = asdict(old_formula)
            new_f = asdict(new_formula)
            semantic = lambda value: {k: value[k] for k in ('valid', 'left', 'right', 'product')}
            same_formula = semantic(old_f) == semantic(new_f)
            for name, same in (('scores', old[0] == new[0]), ('formulas', same_formula)):
                counts[name]['total'] += 1
                counts[name]['equal'] += int(same)
            if old[0] != new[0] or old[2] != new[2] or not same_formula:
                differences.append(dict(t_sec=float(seconds), side=side,
                    old_score=old[0], new_score=new[0], old_digits=old[2], new_digits=new[2],
                    old_formula=old_f, new_formula=new_f))
    for values in counts.values():
        values['rate'] = values['equal'] / values['total'] if values['total'] else None
    output.write_text(json.dumps(differences, ensure_ascii=False, indent=2), encoding='utf-8')
    return dict(metrics=counts, coverage=coverage, difference_scenes=len(differences),
                gate_passed=all(v['equal'] == v['total'] and v['total'] > 0 for v in counts.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--templates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=SAMPLES)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    guard = ObsPriorityGuard()
    guard.checkpoint(force=True)
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    teacher = ScoreOcr.load_default(args.templates)
    cap = cv2.VideoCapture(str(args.video))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    times = np.linspace(1, duration - 1, args.samples)
    split = int(args.samples * TRAIN_FRACTION)
    model_path = args.output / 'digit_student.joblib'
    if not args.evaluate_only:
        data = samples(cap, teacher, times[:split], guard)
        model = dict(schema=MODEL_SCHEMA, feature_size=FEATURE_SIZE,
            labels=fit(data[0], data[1], False, guard),
            confidence=fit(data[0], data[2], True, guard),
            multiply=fit(data[0], data[3], True, guard))
        joblib.dump(model, model_path, compress=3)
    student = ImageFreeScoreOcr(model_path)
    report = evaluate(cap, times[split:], teacher, student, args.output / 'differences.json', guard)
    report.update(training_frames=split, evaluation_frames=len(times) - split,
                  evaluation_start_sec=float(times[split]), duration_sec=duration,
                  method='時間順60%学習/40%評価・等間隔標本。全フレーム試験ではない',
                  downstream='確定盤面・B3特徴・勝率への影響は未測定')
    (args.output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    cap.release()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
