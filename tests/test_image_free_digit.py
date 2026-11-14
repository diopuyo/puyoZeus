"""画像非同梱候補の既定 OFF・失敗閉鎖・画像読込禁止を検証する。"""
from pathlib import Path

import cv2
import joblib
import numpy as np
import pytest

from src.image_free_digit import ImageFreeScoreOcr, digit_features, FEATURE_SIZE, MODEL_SCHEMA
from src.score_ocr import ScoreOcr


class ConstantEstimator:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.value)


def model_file(tmp_path: Path, label: int = 3) -> Path:
    path = tmp_path / 'student.joblib'
    joblib.dump(dict(schema=MODEL_SCHEMA, feature_size=FEATURE_SIZE,
                    labels=ConstantEstimator(label), confidence=ConstantEstimator(0.9),
                    multiply=ConstantEstimator(0.8)), path)
    return path


def test_default_remains_legacy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv('PUYO_IMAGE_FREE_SCORE_OCR', raising=False)
    assert type(ScoreOcr.load_default(tmp_path)) is ScoreOcr


def test_missing_model_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('PUYO_IMAGE_FREE_SCORE_OCR', '1')
    monkeypatch.setenv('PUYO_IMAGE_FREE_DIGIT_MODEL', str(tmp_path / 'missing.joblib'))
    with pytest.raises(FileNotFoundError):
        ScoreOcr.load_default(tmp_path)


def test_no_image_reads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('PUYO_IMAGE_FREE_SCORE_OCR', '1')
    monkeypatch.setenv('PUYO_IMAGE_FREE_DIGIT_MODEL', str(model_file(tmp_path)))
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('画像を読んではならない')
    monkeypatch.setattr(cv2, 'imread', forbidden)
    reader = ScoreOcr.load_default(tmp_path)
    result = reader.read(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert result.score_1p == result.score_2p == 33333333
    reader.read_formula_side(np.zeros((1080, 1920, 3), dtype=np.uint8), '1P')


def test_unknown_is_not_a_digit(tmp_path: Path) -> None:
    reader = ImageFreeScoreOcr(model_file(tmp_path, 10))
    assert reader._classify_digit(np.zeros((50, 40, 3), dtype=np.uint8)) == (None, 0.9)


def test_features_are_finite() -> None:
    values = digit_features(np.zeros((50, 40, 3), dtype=np.uint8))
    assert values.shape == (FEATURE_SIZE[0] * FEATURE_SIZE[1] * 3,)
    assert np.isfinite(values).all()


def test_untrained_threshold_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='教師閾値'):
        ImageFreeScoreOcr(model_file(tmp_path), min_confidence=0.01)


def test_wide_r1_match_is_not_silently_disabled(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match='R1'):
        _ = ImageFreeScoreOcr(model_file(tmp_path))._mult_template_gray
    with pytest.raises(RuntimeError, match='R1'):
        ImageFreeScoreOcr(model_file(tmp_path)).placement_multiply_reader()


def test_wide_r1_uses_dedicated_estimator(tmp_path: Path) -> None:
    reader = ImageFreeScoreOcr(model_file(tmp_path))
    reader._student['wide_multiply'] = ConstantEstimator(0.73)
    callback = reader.placement_multiply_reader()
    assert callback(np.zeros((1080, 1920, 3), dtype=np.uint8)) == [0.73, 0.73]


@pytest.mark.parametrize('label,confidence', [(3, 0.9), (3, 0.6), (10, 0.8)])
def test_batched_read_preserves_legacy_aggregation(
    tmp_path: Path, label: int, confidence: float,
) -> None:
    reader = ImageFreeScoreOcr(model_file(tmp_path, label))
    reader._student['confidence'] = ConstantEstimator(confidence)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for side in ('1P', '2P'):
        for offset in (None, (2, 1)):
            expected = ScoreOcr._read_one_side_detail(reader, frame, side, offset)
            assert reader._read_one_side_detail(frame, side, offset) == expected
    sequence = [frame, frame // 2]
    result = reader.read_batch(sequence)
    for i, image in enumerate(sequence):
        for j, side in enumerate(('1P', '2P')):
            detail = reader.read_side_detail(image, side)
            formula = reader.read_formula_side(image, side, detail[2], detail[3])
            assert result[i][j] == (detail, formula)
    assert reader.read_batch([]) == []


def test_recovery_rejects_conflicting_digits() -> None:
    """色補正と背景補正のどちらも通るが、数字が食い違う場合は欠測を保つ。"""
    labels = np.array([0, 0, 7, 7, 10, 10, 10])
    confidence = np.full(len(labels), 0.9)
    assert ImageFreeScoreOcr._agree_recovery(labels, confidence) is None


def test_single_alignment_vote_does_not_invent_digit() -> None:
    """位置をずらした一例だけ読めても、明暗補正で再確認できなければ採らない。"""
    labels = np.array([10, 10, 10, 10, 10, 2, 10])
    confidence = np.full(len(labels), 0.9)
    assert ImageFreeScoreOcr._agree_recovery(labels, confidence) is None


def test_digit_cell_memo_returns_same_answer_and_isolated_copies(monkeypatch) -> None:
    """同じ画素の桁セルはセルごとに判定を使い回し、答えは同じ・呼び出し元の書き換えは使い回しに漏れない (11/14)。"""
    import numpy as np
    from src.image_free_digit import ImageFreeScoreOcr
    ocr = ImageFreeScoreOcr.__new__(ImageFreeScoreOcr)
    calls = []

    def fake(cells):
        calls.append(len(cells))
        return (np.array([int(c[0, 0, 0]) for c in cells]),
                np.array([int(c[0, 0, 0]) / 100 for c in cells]))
    monkeypatch.setattr(ocr, '_predict_digit_cells_uncached', fake)
    cells = [np.full((50, 40, 3), 7, np.uint8), np.zeros((50, 40, 3), np.uint8)]
    first = ocr._predict_digit_cells(cells)
    first[0][0] = 99
    second = ocr._predict_digit_cells([c.copy() for c in cells])
    assert calls == [2] and second[0].tolist() == [7, 0] and second[1].tolist() == [0.07, 0.0]
    other = [np.full((50, 40, 3), 8, np.uint8), cells[1], np.full((50, 40, 3), 8, np.uint8)]
    labels, confidences = ocr._predict_digit_cells(other)
    assert calls == [2, 1]                      # 新しいセル 1 つだけ (同じ画素の重複も 1 回) を判定する
    assert labels.tolist() == [8, 0, 8] and confidences.tolist() == [0.08, 0.0, 0.08]
    assert labels.dtype == fake(cells)[0].dtype


def test_instances_share_model_and_cell_memo(tmp_path, monkeypatch) -> None:
    """同じ判定器のファイルを読む部品どうしは、判定器と桁の使い回しの表を共有する (11/14: 0 点の判定と
    B3 の得点の読みが同じマスを 2 回計算していた)。2 つ目の部品は 1 つ目が判定したマスを計算し直さない。"""
    import joblib
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from src import image_free_digit as module
    rng = np.random.default_rng(20261114)
    x = rng.normal(size=(60, module.FEATURE_SIZE[0] * module.FEATURE_SIZE[1] * 3))
    labels = rng.integers(0, 11, size=60)
    model = dict(schema=module.MODEL_SCHEMA, feature_size=module.FEATURE_SIZE,
                 labels=ExtraTreesClassifier(n_estimators=4, random_state=1, n_jobs=1).fit(x, labels),
                 confidence=ExtraTreesRegressor(n_estimators=4, random_state=1, n_jobs=1).fit(x, rng.random(60)))
    path = tmp_path / 'digit_student.joblib'
    joblib.dump(model, path)
    monkeypatch.setattr(module, '_SHARED_STUDENTS', {})
    first, second = module.ImageFreeScoreOcr(path), module.ImageFreeScoreOcr(path)
    assert first._student is second._student and first._digit_memo is second._digit_memo
    calls = []
    original = module.ImageFreeScoreOcr._predict_digit_cells_uncached
    monkeypatch.setattr(module.ImageFreeScoreOcr, '_predict_digit_cells_uncached',
                        lambda self, cells: calls.append(len(cells)) or original(self, cells))
    cells = [rng.integers(0, 255, size=(50, 40, 3), dtype=np.uint8) for _ in range(3)]
    a = first._predict_digit_cells(cells)
    b = second._predict_digit_cells([c.copy() for c in cells])
    assert calls == [3] and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    joblib.dump(model, tmp_path / 'other.joblib')
    assert module.ImageFreeScoreOcr(tmp_path / 'other.joblib')._digit_memo is not first._digit_memo   # 別のファイルは別
