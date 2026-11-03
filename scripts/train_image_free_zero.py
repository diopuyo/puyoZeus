"""評価4動画を除外し、得点欄の連続スコアを位置ずれ付きで蒸留する。"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from scripts.compare_image_free_all import read_config, save_json
from scripts.obs_priority_guard import ObsPriorityGuard
from scripts.train_image_free_fix import validate_sources
from src.image_free_zero import LocalZeroModel, SCORE_REGIONS, zero_features
from src.score_zero import ScoreZeroDetector, ZERO_NCC_THRESHOLD

FRAME_SIZE = (1920, 1080)
SIDE_COUNT = len(SCORE_REGIONS)
UNIFORM_SAMPLES = 384
INTRO_SECONDS = 180
INTRO_STRIDE_SECONDS = 0.5
PROGRESS_SAMPLES = 50
MIN_CLASS_SAMPLES = 50
AUGMENT_UNIFORM_SAMPLES = 64
AUGMENT_SCORE_MIN = 0.83
SHIFT_RADIUS = 2
FEATURE_NORM_FLOOR = 1e-12
REGULARIZATION = 1e-6
FIT_THREADS = 3
CACHE_SCHEMA = 'centered-gray-l2-v1'
SAMPLING = dict(uniform=UNIFORM_SAMPLES, intro_seconds=INTRO_SECONDS,
                intro_stride_seconds=INTRO_STRIDE_SECONDS)


def frame_indices(cap: Any) -> np.ndarray:
    """全体と開始付近の標本を動画のフレーム番号で固定する。"""
    count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    if count <= 0 or fps <= 0:
        raise ValueError('学習動画を開けません')
    return np.unique(np.concatenate((np.linspace(0, count - 1, UNIFORM_SAMPLES, dtype=int),
        np.arange(0, min(count, int(INTRO_SECONDS * fps)),
                  max(1, round(fps * INTRO_STRIDE_SECONDS))))))


def collect(video: dict, output: Path, teacher: ScoreZeroDetector,
            guard: ObsPriorityGuard) -> tuple[np.ndarray, np.ndarray]:
    """画像は保存せず、動画・採取条件を照合できる特徴キャッシュを残す。"""
    cache = output / (video['id'] + '.zero_score.joblib')
    identity = dict(video=video, sampling=SAMPLING, schema=CACHE_SCHEMA)
    if cache.exists():
        saved = joblib.load(cache)
        if saved['identity'] != identity:
            raise ValueError('ゼロ教師キャッシュの動画・採取条件が不一致')
        return saved['features'], saved['scores']
    cap = cv2.VideoCapture(video['path'], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    data, scores = [], []
    try:
        frames = frame_indices(cap)
        for index, frame_id in enumerate(frames):
            guard.checkpoint()
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"復号失敗: {video['id']}:{frame_id}")
            frame = cv2.resize(frame, FRAME_SIZE)
            result = teacher.detect(frame)
            data.append([zero_features(frame, side) for side in range(SIDE_COUNT)])
            scores.append([result.score_1p, result.score_2p])
            if index % PROGRESS_SAMPLES == 0:
                print(video['id'], index, len(frames), flush=True)
    finally:
        cap.release()
    arrays = np.asarray(data), np.asarray(scores)
    temporary = cache.with_suffix('.tmp')
    joblib.dump(dict(identity=identity, frames=frames, features=arrays[0], scores=arrays[1]),
                temporary, compress=3)
    temporary.replace(cache)
    return arrays


def shifted_sample(values: np.ndarray, side: int, dy: int, dx: int,
                   template: np.ndarray) -> tuple[np.ndarray, float]:
    """学習用得点欄の微小移動だけを生成し、移動後の旧スコアを教師にする。"""
    y1, y2, x1, x2 = SCORE_REGIONS[side]
    patch = values.reshape(y2 - y1, x2 - x1)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(patch, matrix, (x2 - x1, y2 - y1),
                            borderMode=cv2.BORDER_REFLECT_101)
    score = float(cv2.matchTemplate(shifted, template, cv2.TM_CCOEFF_NORMED)[0, 0])
    feature = shifted.ravel().copy()
    feature -= feature.mean()
    feature /= max(float(np.linalg.norm(feature)), FEATURE_NORM_FLOOR)
    return feature, score


def augment(sample: tuple[np.ndarray, np.ndarray], templates: tuple[np.ndarray, ...],
            guard: ObsPriorityGuard) -> tuple[np.ndarray, np.ndarray]:
    """動画内の標本とゼロ近傍に位置ずれを加える。評価動画は入力しない。"""
    data, scores = sample
    indices = np.unique(np.concatenate((
        np.linspace(0, len(data) - 1, AUGMENT_UNIFORM_SAMPLES, dtype=int),
        np.flatnonzero((scores >= AUGMENT_SCORE_MIN).any(axis=1)))))
    shifts = [(dy, dx) for dy in range(-SHIFT_RADIUS, SHIFT_RADIUS + 1)
              for dx in range(-SHIFT_RADIUS, SHIFT_RADIUS + 1) if (dy, dx) != (0, 0)]
    features, targets = [], []
    for index in indices:
        guard.checkpoint()
        for dy, dx in shifts:
            rows = [shifted_sample(data[index, side], side, dy, dx, templates[side])
                    for side in range(SIDE_COUNT)]
            features.append([row[0] for row in rows])
            targets.append([row[1] for row in rows])
    return (np.concatenate((data, np.asarray(features))),
            np.concatenate((scores, np.asarray(targets))))


def fit_side(samples: list[tuple[np.ndarray, np.ndarray]], side: int,
             guard: ObsPriorityGuard) -> Ridge:
    """画素特徴を画像として複製せず、教師の連続スコアへ回帰する。"""
    guard.checkpoint()
    data = np.concatenate([sample[0][:, side] for sample in samples]).astype(np.float64)
    scores = np.concatenate([sample[1][:, side] for sample in samples])
    with threadpool_limits(limits=FIT_THREADS):
        return Ridge(alpha=REGULARIZATION, solver='cholesky').fit(data, scores)


def validate_model(samples: list[tuple[np.ndarray, np.ndarray]],
                   validation: tuple[np.ndarray, np.ndarray], guard: ObsPriorityGuard) -> dict:
    """最後の学習用動画を分離した検証。比較4動画による設定調整はしない。"""
    data, scores = validation
    predictions = np.stack([fit_side(samples, side, guard).predict(data[:, side])
                            for side in range(SIDE_COUNT)], axis=1)
    expected, actual = scores >= ZERO_NCC_THRESHOLD, predictions >= ZERO_NCC_THRESHOLD
    return dict(frames=len(scores), side_errors=int(np.count_nonzero(expected != actual)),
        both_errors=int(np.count_nonzero(expected.all(axis=1) != actual.all(axis=1))),
        mean_absolute_error=float(np.mean(abs(predictions - scores))),
        maximum_error=float(np.max(abs(predictions - scores))))


def train(config: dict, output: Path) -> None:
    validate_sources(config)
    if len(config['videos']) < 2:
        raise ValueError('動画を分離した検証には学習用動画が2本以上必要です')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(config['asset_root'])
    os.environ['PUYO_IMAGE_FREE_ALL'] = '0'
    cv2.setNumThreads(1)
    guard, teacher = ObsPriorityGuard(), ScoreZeroDetector.load_default()
    samples = [collect(video, output, teacher, guard) for video in config['videos']]
    labels = np.concatenate([sample[1] for sample in samples]) >= ZERO_NCC_THRESHOLD
    counts = [np.bincount(labels[:, side].astype(int), minlength=2).tolist()
              for side in range(SIDE_COUNT)]
    if min(min(count) for count in counts) < MIN_CLASS_SAMPLES:
        raise ValueError('ゼロの陽性または陰性が不足しています')
    templates = tuple(t.astype(np.float32) for t in (teacher._tpl_1p, teacher._tpl_2p))
    augmented = [augment(sample, templates, guard) for sample in samples]
    validation = validate_model(augmented[:-1], samples[-1], guard)
    regressors = tuple(fit_side(augmented, side, guard) for side in range(SIDE_COUNT))
    model = joblib.load(config['base_ui_model'])
    model['tasks']['zero'] = LocalZeroModel(regressors)
    target = output / 'ui_fix2.joblib'
    temporary = target.with_suffix('.tmp')
    joblib.dump(model, temporary, compress=3)
    temporary.replace(target)
    save_json(output / 'zero_training_receipt.json', dict(sources=config['videos'],
        evaluation_excluded=config['evaluation_videos'], counts=counts,
        samples=len(labels), augmented_samples=[len(sample[0]) for sample in augmented],
        validation=validation, validation_video=config['videos'][-1]['id'],
        regularization=REGULARIZATION, shift_radius=SHIFT_RADIUS, sampling=SAMPLING,
        model_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        scope='学習用動画の分離検証。段2/3の合格判定は別途必要'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    train(config, Path(config['output']))


if __name__ == '__main__':
    main()
