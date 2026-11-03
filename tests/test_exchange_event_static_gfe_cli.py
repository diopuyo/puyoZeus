"""描画の実起動経路から静止・仮想着弾のモデル境界までを検証する。"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src import production_config
from src.board import Board
from src.exchange_event_evaluator import FileExchangeModels, evaluate_exchange_event
from src.exchange_event_tracker import ExchangeChainRecord, ExchangeRecord
from src.exchange_hidden_row_probability import weighted_landing, weighted_s3
from tests.test_exchange_event_evaluator import inputs, write_models
from tests.test_exchange_event_static_gfe import r1_candidate


class OverlayReady(Exception):
    """実overlayとtrackerの構築後、映像デバイスを開く前に止める。"""


def start_render(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> Any:
    """main・argparse・generateは実関数を通し、映像I/Oだけを省く。"""
    from scripts import visualize_advantage_overlay as render
    captured = []
    original = render.ExchangeEventOverlay

    def capture(*args: Any, **kwargs: Any) -> Any:
        captured.append(original(*args, **kwargs))
        raise OverlayReady

    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", ["render", *arguments])
        patch.setattr("src.exchange_event_m0.FileM0Predictor", lambda path: lambda *args: .5)
        patch.setattr(render, "ExchangeEventOverlay", capture)
        with pytest.raises(OverlayReady):
            render.main()
    return captured[0]


@pytest.fixture
def model_paths(tmp_path: Path) -> tuple[Path, Path]:
    """小さな実LRとHistGBCを別資産として作る。"""
    write_models(tmp_path)
    return tmp_path, r1_candidate(tmp_path / "static", (20., 40.))


@pytest.mark.parametrize("registry", [False, True])
def test_render_static_and_landing(model_paths: tuple[Path, Path],
                                   monkeypatch: pytest.MonkeyPatch, registry: bool) -> None:
    """直接指定と将来の本番1行登録の双方で同じtrackerへ届く。"""
    base, candidate = model_paths
    flags = ["--exchange-event-update", "--exchange-event-model-dir", str(base)]
    flags += ["--exchange-event-static-gfe-dir", str(candidate)]
    if registry:
        import shlex
        registry_flags = shlex.join(flags)
        monkeypatch.setattr(production_config, "exchange_event_flags", lambda: registry_flags)
        flags = ["--production-exchange-event"]
    overlay = start_render(monkeypatch, flags)
    models = overlay.tracker.models
    assert type(models.estimators["G_fe"]) is HistGradientBoostingClassifier
    assert type(models.landing_models.estimators["G_fe"]) is LogisticRegression
    calls = []
    original = FileExchangeModels.predict_source_probability

    def predict(self: FileExchangeModels, name: str, values: np.ndarray) -> float:
        calls.append((name, type(self.estimators[name])))
        return original(self, name, values)

    monkeypatch.setattr(FileExchangeModels, "predict_source_probability", predict)
    assert overlay.tracker._evaluate(inputs()[0], "G_fe", 0.)
    assert calls == [("G_fe", HistGradientBoostingClassifier)]
    check_weighted_routes(overlay, monkeypatch, calls)


def check_weighted_routes(overlay: Any, monkeypatch: pytest.MonkeyPatch, calls: list) -> None:
    """加重S3・加重着弾・実G_feキャッシュを同じtrackerで通す。"""
    static, firing, end = inputs()
    tracker, projection = overlay.tracker, overlay._landing_projection
    chain = ExchangeChainRecord("1P", 1, 0., 0.)
    entry = dict(options=[dict(score=700., weight=.5), dict(score=700., weight=.5)],
                 stats=dict(mean_send=10.), elapsed=0.)
    tracker.hidden_row_belief = SimpleNamespace(active=lambda c: entry)
    tracker.current, tracker.firing = ExchangeRecord(1, 0, 0., [chain]), firing
    overlay._start, overlay._build_static = 0., lambda *args: static
    overlay._m0 = lambda *args: .5
    latest = tuple(SimpleNamespace(board=Board(), queue=np.zeros(4)) for _ in range(2))
    snapshot = SimpleNamespace(net_balance_capped=0., forecast_p1=0.,
                               total_dropped_to_p1=0., total_dropped_to_p2=0.)
    monkeypatch.setattr(projection, "_incoming", lambda *args: [1, 0])
    monkeypatch.setattr(projection, "_probability_inputs", lambda ov, snap, sides, inc, hands, t:
                        (projection._landing_gfe(ov, snap, sides, inc, t), {}))
    calls.clear()
    assert np.isfinite(weighted_s3(tracker.hidden_row_belief, tracker, end))
    assert weighted_landing(projection, overlay, snapshot, latest, (), {}, 10.) is not None
    assert ("G_fe", LogisticRegression) in calls
    assert all(kind is LogisticRegression for _, kind in calls)
    assert projection.gfe_cache
    before = list(calls)
    projection._landing_gfe(overlay, snapshot, latest, [1, 0], 10.)
    assert calls == before


def test_render_default_bytes(model_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """未指定時は実CLIのtracker出力も従来モデルとバイト一致する。"""
    base, _ = model_paths
    overlay = start_render(monkeypatch, ["--exchange-event-update", "--exchange-event-model-dir", str(base)])
    models = FileExchangeModels.load(base)
    assert overlay.tracker.models.landing_models is None
    for name, event in zip(("G_fe", "S1", "S3"), inputs()):
        assert overlay.tracker._evaluate(event, name, 0.)
        assert np.float64(overlay.tracker.probability).tobytes() == np.float64(
            evaluate_exchange_event(event, models)).tobytes()
