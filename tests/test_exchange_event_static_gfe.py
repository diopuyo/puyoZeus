"""静止と仮想着弾の振分け・既定OFF・59列検証を確認する。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from src.board import Board
from src.exchange_event_evaluator import FileExchangeModels, STATIC_GFE_VERSION, evaluate_exchange_event, file_sha256
from src.exchange_event_features import G_COLUMNS
from src.exchange_event_landing import ExchangeLandingProjection
from tests.test_exchange_event_evaluator import inputs, write_models
from tests.test_exchange_event_user_phase import candidate


def r1_candidate(directory: Path, thresholds: tuple) -> Path:
    """現行59列の小さな実モデルを保存する。"""
    directory.mkdir()
    x = np.random.default_rng(0).normal(size=(24, len(G_COLUMNS)))
    model = HistGradientBoostingClassifier(max_iter=2, min_samples_leaf=2).fit(x, np.arange(24) % 2)
    path = directory / 'G_fe.joblib'
    joblib.dump(model, path)
    entry = dict(file=path.name, columns=G_COLUMNS, sha256=file_sha256(path), estimator_type=type(model).__name__)
    (directory / 'manifest.json').write_text(json.dumps(dict(version=STATIC_GFE_VERSION,
        elapsed_thresholds=thresholds, models=dict(G_fe=entry))), encoding='utf-8')
    return directory


@pytest.mark.parametrize('kind', ['V3s', 'R1s'])
def test_purpose_routing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    """実際の着弾経路はLR、通常静止は指定モデルを使う。"""
    write_models(tmp_path)
    base = FileExchangeModels.load(tmp_path)
    path = (candidate(tmp_path/'new', 'HistGradientBoostingClassifier') if kind == 'V3s'
            else r1_candidate(tmp_path/'new', base.elapsed_thresholds))
    models = FileExchangeModels.load(tmp_path, gfe_directory=path, gfe_static_only=True)
    assert models.landing_models is not None
    assert models.user_phase == (kind == 'V3s')
    calls = []
    original = FileExchangeModels.predict_source_probability

    def capture(self: FileExchangeModels, name: str, values: np.ndarray) -> float:
        calls.append((self, name, len(values)))
        return original(self, name, values)

    monkeypatch.setattr(FileExchangeModels, 'predict_source_probability', capture)
    event = inputs()[0]
    evaluate_exchange_event(event, models)
    assert calls[-1] == (models, 'G_fe', 54 if kind == 'V3s' else 59)
    overlay = SimpleNamespace(tracker=SimpleNamespace(models=models), _start=0,
        _m0=lambda *args: .5, _build_static=lambda *args: event)
    latest = tuple(SimpleNamespace(board=Board(), queue=np.zeros(4)) for _ in range(2))
    result = ExchangeLandingProjection()._landing_gfe(overlay, SimpleNamespace(), latest, [0, 0], 10.)
    assert calls[-1] == (models.landing_models, 'G_fe', 59)
    assert result == evaluate_exchange_event(event, base)
    for event in inputs()[1:]:
        assert evaluate_exchange_event(event, models) == evaluate_exchange_event(event, base)


def test_off_equal(tmp_path: Path) -> None:
    """既定・明示OFFとも候補なしの出力をバイトで保持する。"""
    write_models(tmp_path)
    base = FileExchangeModels.load(tmp_path)
    off = FileExchangeModels.load(tmp_path, gfe_static_only=False)
    assert off.landing_models is None
    for event in inputs():
        assert np.float64(evaluate_exchange_event(event, base)).tobytes() == np.float64(
            evaluate_exchange_event(event, off)).tobytes()
    with pytest.raises(ValueError, match='候補ディレクトリ'):
        FileExchangeModels.load(tmp_path, gfe_static_only=True)


@pytest.mark.parametrize('key', ['version', 'columns', 'elapsed_thresholds', 'sha256', 'estimator_type'])
def test_r1_rejects_invalid_manifest(tmp_path: Path, key: str) -> None:
    """列・閾値・推定器・ハッシュの不整合を拒否する。"""
    base = FileExchangeModels((20., 40.), {})
    directory = r1_candidate(tmp_path/'new', base.elapsed_thresholds)
    path = directory/'manifest.json'
    meta = json.loads(path.read_text())
    target = meta if key in ('version', 'elapsed_thresholds') else meta['models']['G_fe']
    target[key] = [] if key in ('columns', 'elapsed_thresholds') else 'invalid'
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        base.with_static_gfe(directory)


@pytest.mark.parametrize('key', ['version', 'models', 'G_fe', 'columns',
                                'elapsed_thresholds', 'file', 'sha256', 'estimator_type'])
def test_missing_manifest_key(tmp_path: Path, key: str) -> None:
    """必須キー欠落はKeyErrorを漏らさずValueErrorにする。"""
    base = FileExchangeModels((20., 40.), {})
    directory = r1_candidate(tmp_path/'new', base.elapsed_thresholds)
    path = directory/'manifest.json'
    meta = json.loads(path.read_text())
    target = (meta if key in ('version', 'models', 'elapsed_thresholds') else
              meta['models'] if key == 'G_fe' else meta['models']['G_fe'])
    del target[key]
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        base.with_static_gfe(directory)


def test_rejects_double_static(tmp_path: Path) -> None:
    """同一候補でも二重適用を拒否し、着弾用LRの境界を維持する。"""
    base = FileExchangeModels((20., 40.), {})
    directory = r1_candidate(tmp_path/'new', base.elapsed_thresholds)
    models = base.with_static_gfe(directory)
    with pytest.raises(ValueError, match='二重適用'):
        models.with_static_gfe(directory)
    assert models.landing_models is base


@pytest.mark.parametrize('index', range(len(G_COLUMNS)))
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_all_gfe_columns(index: int, value: float) -> None:
    """M0由来の列を含む全列を推定器呼出前に拒否する。"""
    model = FileExchangeModels((20., 40.), {})
    values = np.zeros(len(G_COLUMNS))
    values[index] = value
    with pytest.raises(ValueError, match='missing_input'):
        model.predict_source_probability('G_fe', values)


def test_nonfinite_reaches_missing_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """HistGBCがNaNを扱えてもtrackerではLR同様の欠測にする。"""
    from src.exchange_event_tracker import ExchangeEventTracker
    import src.exchange_event_evaluator as evaluator
    base = FileExchangeModels((20., 40.), {})
    directory = r1_candidate(tmp_path/'new', base.elapsed_thresholds)
    tracker = ExchangeEventTracker(base.with_static_gfe(directory))
    values = np.zeros(len(G_COLUMNS))
    values[len(evaluator.D_COLUMNS)] = np.nan
    monkeypatch.setattr(evaluator, 'build_features', lambda *args: ('G_fe', values, 0))
    assert not tracker._evaluate(inputs()[0], 'G_fe', 0.)
    assert tracker.probability is None
    assert 'missing_input' in tracker.diagnostics[-1]['reason']
