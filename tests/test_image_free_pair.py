"""画像なし方式の試合終了の組判定 (11/14): 切り抜き画像が無くても、学んだ森で組が成立すること。"""
import numpy as np
import pytest

from src.match_end_pair import PAIR_NCC_THRESHOLD, PAIR_ROIS


@pytest.fixture
def image_free(monkeypatch):
    monkeypatch.setenv('PUYO_IMAGE_FREE_ALL', '1')
    calls = []

    def fake_learned_score(key: str, image: np.ndarray) -> float:
        calls.append((key, image.shape))
        return PAIR_NCC_THRESHOLD + (0.01 if int(image[0, 0, 0]) else -0.01)
    import src.image_free_ui as image_free_ui
    monkeypatch.setattr(image_free_ui, 'learned_score', fake_learned_score)
    return calls


def _frame(lit: set[str]) -> np.ndarray:
    """lit の領域の左上だけ 1 にした画面 (偽の森は左上の画素で採否を返す)。"""
    frame = np.zeros((1080, 1920, 3), np.uint8)
    for key in lit:
        x, y, _, _ = PAIR_ROIS[key]
        frame[y, x] = 1
    return frame


def test_pair_uses_learned_forest_per_template_without_images(image_free) -> None:
    from src.match_end_pair import MatchEndPairDetector
    detector = MatchEndPairDetector({})                       # 画像なし方式では切り抜き画像が無い
    result = detector.detect(_frame({'yatta_1p', 'batan_2p'}))
    assert result.detected and result.winner == '1P'
    assert sorted({key for key, _ in image_free}) == ['pair_batan', 'pair_yatta']   # 左右で同じ森
    assert {shape for _, shape in image_free} == {(h, w, 3) for _, _, w, h in PAIR_ROIS.values()}
    assert not detector.detect(_frame({'yatta_1p', 'yatta_2p'})).detected   # 組にならない


def test_async_pair_detector_inherits_learned_scoring(image_free) -> None:
    from src.async_match_end_pair import AsyncMatchEndPairDetector
    detector = AsyncMatchEndPairDetector({})
    crops = {key: _frame({'yatta_2p', 'batan_1p'})[y:y + h, x:x + w] for key, (x, y, w, h) in PAIR_ROIS.items()}
    result = detector._evaluate(crops, 1.0)
    assert result.detected and result.winner == '2P'


def test_template_path_unchanged_when_image_free_off(monkeypatch) -> None:
    monkeypatch.setenv('PUYO_IMAGE_FREE_ALL', '0')
    from src.match_end_pair import MatchEndPairDetector
    detector = MatchEndPairDetector({})
    assert detector._score_crop(np.zeros((500, 700, 3), np.uint8), 'batan_1p') == -1.0   # 画像なしでは照合しない


def test_ui_ensemble_pair_returns_threshold_side() -> None:
    from src.image_free_ui_ensemble import UIEnsemble, task_features

    class Fixed:
        def __init__(self, label: str) -> None:
            self.label = label

        def predict(self, data: np.ndarray) -> np.ndarray:
            assert data.shape == (1, task_features('pair_yatta', np.zeros((600, 800, 3), np.uint8)).size)
            return np.array([self.label])
    image = np.zeros((600, 800, 3), np.uint8)
    assert UIEnsemble('pair_yatta', Fixed('true')).predict_image(image) > PAIR_NCC_THRESHOLD
    assert UIEnsemble('pair_yatta', Fixed('false')).predict_image(image) < PAIR_NCC_THRESHOLD
