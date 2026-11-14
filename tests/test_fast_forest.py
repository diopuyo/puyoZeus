"""fast_forest: sklearn の決定木・森と同じ答えを返すこと。"""
import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.fast_forest import ENABLE_ENV, FastTreeModel, accelerate, supported

SEED = 20261114


@pytest.fixture(scope='module')
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(400, 12))
    labels = np.array(['"a"', '"b"', 'null'])[(x[:, 0] + x[:, 1] > 0).astype(int) + (x[:, 2] > 1).astype(int)]
    target = x[:, :2] @ np.array([[1.5, -0.5], [0.3, 2.0]]) + rng.normal(scale=0.1, size=(400, 2))
    return x, labels, target


def _probe(x: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED + 1)
    return np.vstack([x[:50], rng.normal(scale=2.0, size=(200, x.shape[1]))])


@pytest.mark.parametrize('make', [
    lambda: ExtraTreesClassifier(n_estimators=16, random_state=SEED, n_jobs=1),
    lambda: DecisionTreeClassifier(random_state=SEED),
])
def test_classifier_matches_sklearn(data, make) -> None:
    x, labels, _ = data
    model = make().fit(x, labels)
    probe = _probe(x)
    assert np.array_equal(FastTreeModel(model).predict(probe), model.predict(probe))
    assert all(FastTreeModel(model).predict(probe[i:i + 1])[0] == model.predict(probe[i:i + 1])[0] for i in range(20))


@pytest.mark.parametrize('make, outputs', [
    (lambda: ExtraTreesRegressor(n_estimators=16, random_state=SEED, n_jobs=1), 1),
    (lambda: ExtraTreesRegressor(n_estimators=16, random_state=SEED, n_jobs=1), 2),
    (lambda: DecisionTreeRegressor(random_state=SEED), 2),
])
def test_regressor_matches_sklearn_exactly(data, make, outputs) -> None:
    x, _, target = data
    y = target[:, 0] if outputs == 1 else target
    model = make().fit(x, y)
    probe = _probe(x)
    fast = FastTreeModel(model).predict(probe)
    assert fast.shape == model.predict(probe).shape
    assert np.array_equal(fast, model.predict(probe))       # 足す順まで同じなので小数も完全一致


def test_nan_falls_back_to_sklearn(data) -> None:
    x, labels, _ = data
    model = ExtraTreesClassifier(n_estimators=4, random_state=SEED, n_jobs=1).fit(x, labels)
    probe = x[:3].copy()
    probe[0, 0] = np.nan
    assert np.array_equal(FastTreeModel(model).predict(probe), model.predict(probe))


def test_accelerate_swaps_dict_and_holder_and_can_be_disabled(data, monkeypatch) -> None:
    x, labels, target = data
    classifier = ExtraTreesClassifier(n_estimators=4, random_state=SEED, n_jobs=1).fit(x, labels)
    regressor = ExtraTreesRegressor(n_estimators=4, random_state=SEED, n_jobs=1).fit(x, target[:, 0])

    class Holder:
        def __init__(self) -> None:
            self.classifier, self.coordinates = classifier, None

    model = accelerate({'schema': 1, 'labels': classifier, 'tasks': {'x': Holder()}, 'confidence': regressor})
    assert isinstance(model['labels'], FastTreeModel) and isinstance(model['confidence'], FastTreeModel)
    assert isinstance(model['tasks']['x'].classifier, FastTreeModel) and model['schema'] == 1
    assert model['labels'].classes_ is classifier.classes_       # predict 以外は元のまま
    monkeypatch.setenv(ENABLE_ENV, '0')
    assert accelerate({'labels': classifier})['labels'] is classifier


def test_parallel_or_multi_output_classifier_is_not_swapped(data) -> None:
    x, labels, _ = data
    parallel = ExtraTreesClassifier(n_estimators=4, random_state=SEED, n_jobs=2).fit(x, labels)
    multi = DecisionTreeClassifier(random_state=SEED).fit(x, np.stack([labels, labels[::-1]], axis=1))
    assert not supported(parallel) and not supported(multi)


def test_predict_together_matches_separate_predicts(data) -> None:
    """同じ特徴の森 (分類と回帰) をまとめてたどっても、それぞれの predict・sklearn と同じ答え。"""
    from src.fast_forest import predict_together
    x, labels, target = data
    classifier = ExtraTreesClassifier(n_estimators=16, random_state=SEED, n_jobs=1).fit(x, labels)
    regressor = ExtraTreesRegressor(n_estimators=12, random_state=SEED + 2, n_jobs=1).fit(x, target[:, 0])
    single = DecisionTreeRegressor(random_state=SEED).fit(x, target)
    fast = [FastTreeModel(m) for m in (classifier, regressor, single)]
    probe = _probe(x)
    for rows in (probe, probe[:1], probe[3:4]):
        together = predict_together(fast, rows)
        for model, original, answer in zip(fast, (classifier, regressor, single), together):
            assert np.array_equal(answer, model.predict(rows))
            assert np.array_equal(answer, original.predict(rows))
    assert all(np.array_equal(a, b.predict(probe)) for a, b in
               zip(predict_together([classifier, fast[1]], probe), (classifier, regressor)))   # 混在は別々に評価
