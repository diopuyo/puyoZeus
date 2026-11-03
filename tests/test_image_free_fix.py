"""希少事象の学習契約と候補の画像読込禁止を検証する。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.compare_image_free_all import mode
from scripts.train_image_free_fix import validate_sources
from src.image_free_ui_ensemble import UIEnsemble, task_features


class ConstantEstimator:
    def __init__(self, value: Any) -> None:
        self.value = value

    def predict(self, data: np.ndarray) -> np.ndarray:
        return np.asarray([self.value] * len(data))


@pytest.mark.parametrize('key', ['death_0', 'death_1', 'match_end', 'ojama', 'chain'])
@pytest.mark.parametrize('positive', [False, True])
def test_positive_negative_without_image_reads(key: str, positive: bool) -> None:
    labels = dict(death_0=positive, death_1=positive,
                  match_end='match_end_batan' if positive else None,
                  ojama='rock' if positive else None, chain=5 if positive else None)
    model = UIEnsemble(key, ConstantEstimator(json.dumps(labels[key])), ConstantEstimator([17, 29]))
    with mode(True):
        result = model.predict_image(np.zeros((1080, 1920, 3), np.uint8))
    if key.startswith('death_'):
        assert (result >= .55) == positive
    elif key == 'match_end':
        assert result['detected'] == positive
        assert (result['score'] >= .55) == positive
    elif key == 'ojama':
        assert result == labels[key]
    else:
        assert result['chain_count'] == labels[key]
        assert result['location'] == ((17, 29) if positive else None)


@pytest.mark.parametrize('kind', [None, 'small', 'line', 'rock', 'star', 'moon', 'crown', 'big_crown', 'supercrown'])
def test_ojama_class_is_preserved(kind: str | None) -> None:
    model = UIEnsemble('ojama', ConstantEstimator(json.dumps(kind)))
    assert model.predict_image(np.zeros((40, 40, 3), np.uint8)) == kind


def test_death_does_not_use_opposite_board() -> None:
    image = np.zeros((1080, 1920, 3), np.uint8)
    expected = task_features('death_0', image)
    image[:, 1150:] = np.random.default_rng(1).integers(0, 256, image[:, 1150:].shape, dtype=np.uint8)
    np.testing.assert_array_equal(task_features('death_0', image), expected)
    assert np.any(task_features('death_1', image) != 0)


def test_ojama_retains_color_lost_by_gray_normalization() -> None:
    red, blue = np.zeros((40, 40, 3), np.uint8), np.zeros((40, 40, 3), np.uint8)
    red[:, :, 2], blue[:, :, 0] = 255, 255
    np.testing.assert_array_equal(task_features('ojama', red), task_features('ojama', blue))
    assert np.any(task_features('ojama', red, True) != task_features('ojama', blue, True))


def test_gray_model_without_color_attribute_remains_loadable() -> None:
    model = UIEnsemble('ojama', ConstantEstimator(json.dumps('rock')))
    del model.color
    assert model.predict_image(np.zeros((40, 40, 3), np.uint8)) == 'rock'


@pytest.mark.parametrize('identity', ['c68', 'video_c68', 'video_d01', 'video_d02', 'video_d04'])
def test_evaluation_id_excluded(identity: str) -> None:
    config = dict(videos=[dict(id=identity, path='train.mp4', title='マスター')], evaluation_videos=[])
    with pytest.raises(ValueError, match='評価動画'):
        validate_sources(config)


def test_alias_path_and_youtube_excluded(tmp_path: Path) -> None:
    video = dict(id='alias', path=str(tmp_path/'eval.mp4'), title='マスター')
    with pytest.raises(ValueError, match='評価動画'):
        validate_sources(dict(videos=[video], evaluation_videos=[video]))
    video['youtube_id'] = 'DBfHe3dA244'
    with pytest.raises(ValueError, match='評価動画'):
        validate_sources(dict(videos=[video], evaluation_videos=[]))


def test_unconfirmed_tier_excluded() -> None:
    video = dict(id='training', path='train.mp4', title='初心者')
    with pytest.raises(ValueError, match='ティア未確認'):
        validate_sources(dict(videos=[video], evaluation_videos=[]))


def test_comparison_counts_only_active_games() -> None:
    from scripts.compare_image_free_all import compare
    state = dict(counts={}, display_scenes=0, previous_display_difference=False)
    for index, active in ((0, False), (1, True), (1, True), (2, True)):
        values = [dict(game=dict(index=index, active=active)) for _ in range(2)]
        assert compare(values, state, index, 60.) is None
    assert state['observed_games'] == [[1, 2], [1, 2]]


def test_missing_death_positives_cannot_pass() -> None:
    from scripts.train_image_free_fix import fit_task
    rows = [('death_0', np.zeros(16, np.float32), False, None) for _ in range(64)]
    _, receipt = fit_task('death_0', rows)
    assert not receipt['coverage_passed']
    assert receipt['positives'] == 0
    assert receipt['used_features'] == 0


def test_full_comparison_requires_short_gate(tmp_path: Path) -> None:
    from scripts.prepare_image_free_fix import prepare
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(output=str(tmp_path), videos=[])))
    with pytest.raises(ValueError, match='段3'):
        prepare(config, tmp_path/'full', False, None)
    assert not (tmp_path/'full').exists()


def test_full_comparison_rejects_model_changed_after_gate(tmp_path: Path) -> None:
    from scripts.prepare_image_free_fix import prepare
    config, gate = tmp_path/'config.json', tmp_path/'gate.json'
    config.write_text(json.dumps(dict(output=str(tmp_path), videos=[])))
    (tmp_path/'ui.joblib').write_bytes(b'changed')
    gate.write_text(json.dumps(dict(gate_passed=True, model_sha256={'ui.joblib': 'old'})))
    with pytest.raises(ValueError, match='モデルが変わって'):
        prepare(config, tmp_path/'full', False, gate)
    assert not (tmp_path/'full').exists()


def test_each_teacher_ojama_class_requires_coverage() -> None:
    from scripts.train_image_free_fix import fit_task
    data = np.random.default_rng(17).normal(size=(128, 16)).astype(np.float32)
    rows = [('ojama', row, 'rock' if index % 2 else None, None) for index, row in enumerate(data)]
    _, receipt = fit_task('ojama', rows, ['rock', 'moon'])
    assert receipt['positives'] == 64
    assert receipt['underrepresented_classes'] == ['moon']
    assert not receipt['coverage_passed']


def test_baseline_reuse_rejects_different_failure_window(tmp_path: Path) -> None:
    from scripts.verify_image_free_fix import baseline_rows
    (tmp_path/'c68.json').write_text(json.dumps(dict(video='c68', centers=[14554], rows=[])))
    with pytest.raises(ValueError, match='対象区間'):
        baseline_rows(tmp_path, 'c68', [19761])
