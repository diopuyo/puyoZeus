"""MultiCnnEnsemble のテスト (Z-X)。"""
from __future__ import annotations

import numpy as np

from src.multi_cnn_ensemble import MultiCnnEnsemble
from src.patch_classifier import CLASS_INDEX_TO_COLOR, NUM_CLASSES


class _FakeCNN:
    """固定の確率を返すモック CNN。"""

    def __init__(self, probs_array: np.ndarray) -> None:
        self.probs = probs_array

    def predict_proba(self, patch: np.ndarray) -> np.ndarray:
        return self.probs.copy()

    def predict_proba_batch(self, patches: list[np.ndarray]) -> np.ndarray:
        return np.tile(self.probs[None, :], (len(patches), 1))


def test_predict_proba_uniform_weights() -> None:
    """2 CNN を等重み平均。"""
    p1 = np.zeros(NUM_CLASSES, dtype=np.float32)
    p1[0] = 1.0  # CNN1: EM 100%
    p2 = np.zeros(NUM_CLASSES, dtype=np.float32)
    p2[1] = 1.0  # CNN2: RED 100%
    ensemble = MultiCnnEnsemble(cnns=[_FakeCNN(p1), _FakeCNN(p2)])
    patch = np.zeros((8, 8, 3), dtype=np.uint8)
    avg = ensemble.predict_proba(patch)
    assert avg[0] == 0.5
    assert avg[1] == 0.5


def test_predict_proba_weighted() -> None:
    """重み付き平均。"""
    p1 = np.zeros(NUM_CLASSES, dtype=np.float32)
    p1[0] = 1.0
    p2 = np.zeros(NUM_CLASSES, dtype=np.float32)
    p2[1] = 1.0
    ensemble = MultiCnnEnsemble(
        cnns=[_FakeCNN(p1), _FakeCNN(p2)],
        weights=[0.7, 0.3],
    )
    avg = ensemble.predict_proba(np.zeros((8, 8, 3), dtype=np.uint8))
    assert abs(avg[0] - 0.7) < 1e-5
    assert abs(avg[1] - 0.3) < 1e-5


def test_classify_via_argmax() -> None:
    """classify は argmax を返す。"""
    p1 = np.zeros(NUM_CLASSES, dtype=np.float32)
    p1[3] = 1.0  # GRN
    ensemble = MultiCnnEnsemble(cnns=[_FakeCNN(p1)])
    patch = np.zeros((8, 8, 3), dtype=np.uint8)
    color = ensemble.classify(patch)
    assert color == CLASS_INDEX_TO_COLOR[3]


def test_predict_proba_batch() -> None:
    p1 = np.zeros(NUM_CLASSES, dtype=np.float32)
    p1[2] = 1.0
    p2 = np.zeros(NUM_CLASSES, dtype=np.float32)
    p2[3] = 1.0
    ensemble = MultiCnnEnsemble(cnns=[_FakeCNN(p1), _FakeCNN(p2)])
    patches = [np.zeros((8, 8, 3), dtype=np.uint8)] * 3
    probs = ensemble.predict_proba_batch(patches)
    assert probs.shape == (3, NUM_CLASSES)
    assert abs(probs[0, 2] - 0.5) < 1e-5
    assert abs(probs[0, 3] - 0.5) < 1e-5


def test_empty_cnns_returns_zeros() -> None:
    ensemble = MultiCnnEnsemble(cnns=[])
    avg = ensemble.predict_proba(np.zeros((8, 8, 3), dtype=np.uint8))
    assert np.all(avg == 0)


def test_weight_mismatch_raises() -> None:
    p = np.zeros(NUM_CLASSES, dtype=np.float32)
    try:
        MultiCnnEnsemble(
            cnns=[_FakeCNN(p)],
            weights=[0.5, 0.5],
        )
        assert False, "should have raised"
    except ValueError:
        pass
