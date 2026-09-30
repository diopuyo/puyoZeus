"""仮想着弾評価・S3 勝率の再利用 (出力同一の高速化) が、計算し直した場合と同じ値を返すことを確かめる。"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import src.exchange_event_landing as landing
import src.exchange_hidden_row_probability as hidden
from src.board import Board
from src.exchange_event_evaluator import D_COLUMNS, StaticInput

PROBABILITY = 0.3
ELAPSED_THRESHOLDS = (30.0, 90.0)


class FakeModels:
    """モデル境界の代役。入力特徴の和から決まる値を返す (呼び出し回数も数える)。"""
    elapsed_thresholds = ELAPSED_THRESHOLDS
    count_features = False

    def __init__(self) -> None:
        self.calls = 0

    def predict_source_probability(self, model_name: str, features: np.ndarray) -> float:
        self.calls += 1
        return float(np.clip(PROBABILITY + 1e-3 * float(np.nansum(features)) % 0.2, 0.01, 0.99))


def make_overlay(models: FakeModels, counter: dict) -> SimpleNamespace:
    def m0(boards: np.ndarray, queues: np.ndarray) -> float:
        counter['m0'] += 1
        return 0.4 + 1e-3 * float(boards.sum() % 7)

    def build(virtual: tuple, snapshot: SimpleNamespace, elapsed: float, m0_value: float) -> StaticInput:
        features = np.zeros(len(D_COLUMNS))
        features[0] = float(virtual[0]._grid.sum() % 11)
        return StaticInput(features, m0_value, elapsed, source_side=0)
    return SimpleNamespace(_start=0.0, _m0=m0, _build_static=build,
                           tracker=SimpleNamespace(models=models, latest_chain=lambda side: None))


def make_latest(queue: tuple) -> tuple:
    board = Board()
    board._grid[12, :3] = 1
    side = SimpleNamespace(board=board, queue=np.array(queue, dtype=np.int64), t_sec=0.0)
    return (side, SimpleNamespace(board=board.copy(), queue=np.array(queue, dtype=np.int64), t_sec=0.0))


@pytest.mark.parametrize('cache_on', (True, False))
def test_landing_gfe_cache_returns_same_value(monkeypatch: pytest.MonkeyPatch, cache_on: bool) -> None:
    monkeypatch.setattr(landing, 'EXACT_GFE_CACHE', cache_on)
    counter = {'m0': 0}
    models = FakeModels()
    overlay = make_overlay(models, counter)
    projection = landing.ExchangeLandingProjection()
    snapshot = SimpleNamespace(net_balance_capped=1.0, forecast_p1=0.5)
    latest = make_latest((1, 2, 3, 4))
    first = projection._landing_gfe(overlay, snapshot, latest, [10, 0], 20.0)
    second = projection._landing_gfe(overlay, snapshot, latest, [10, 0], 20.0)
    assert first == second
    assert counter['m0'] == (1 if cache_on else 2)
    other = projection._landing_gfe(overlay, snapshot, make_latest((2, 2, 3, 4)), [10, 0], 20.0)
    assert counter['m0'] == (2 if cache_on else 3)
    assert isinstance(other, float)


@dataclass
class FakeEnd:
    """`dataclasses.replace` できる S3 入力の代役 (scores_after だけを使う)。"""
    scores_after: np.ndarray


def test_s3_cache_evaluates_each_total_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_evaluate(event: FakeEnd, models: object) -> float:
        calls.append(tuple(event.scores_after))
        return 0.1 * float(event.scores_after[0] % 7) + 0.01
    monkeypatch.setattr(hidden, 'evaluate_exchange_event', fake_evaluate)
    chains = [SimpleNamespace(side='1P'), SimpleNamespace(side='1P'), SimpleNamespace(side='2P')]
    event = FakeEnd(np.zeros(2))
    cache: dict = {}
    scenarios = [((100.0, 200.0, 50.0), 0.5), ((200.0, 100.0, 50.0), 0.25), ((150.0, 150.0, 50.0), 0.25)]
    with_cache = [hidden.s3_probability(cache, event, chains, scores, None) for scores, _ in scenarios]
    assert len(calls) == 1  # 3 通りとも側別合計は (300, 50)
    assert len(set(with_cache)) == 1
    monkeypatch.setattr(hidden, 'EXACT_S3_CACHE', False)
    calls.clear()
    without = [hidden.s3_probability({}, event, chains, scores, None) for scores, _ in scenarios]
    assert len(calls) == 3 and without == with_cache


def test_capped_scenarios_keep_top_weights_and_total_mass() -> None:
    rows = [((float(i),), w) for i, w in enumerate([0.1, 0.4, 0.05, 0.3, 0.15])]
    assert hidden.capped(rows, None) is rows            # 既定は従来どおり全候補
    assert hidden.capped(rows, 5) is rows               # 上限以下は素通し
    kept = hidden.capped(rows, 3)                       # 重み上位 3 (0.4, 0.3, 0.15) を元の順で
    assert [scores for scores, _ in kept] == [(1.0,), (3.0,), (4.0,)]
    assert sum(w for _, w in kept) == pytest.approx(sum(w for _, w in rows))
    assert kept == hidden.capped(rows, 3)               # 決定的
    tie = [((0.0,), 0.5), ((1.0,), 0.25), ((2.0,), 0.25)]
    assert [s for s, _ in hidden.capped(tie, 2)] == [(0.0,), (1.0,)]  # 同点は元の順


def test_multilanding_node_limit_is_read_from_projection() -> None:
    import src.exchange_event_multilanding as multilanding
    projection = landing.ExchangeLandingProjection()
    assert projection.multilanding_node_limit is None
    seen = []
    original = multilanding.prove_multilanding

    def spy(*args: object, **kwargs: object) -> dict:
        seen.append(args[-1])
        return dict(dead=False, reason='spy', rounds=[], nodes=0)
    multilanding.prove_multilanding = spy
    try:
        board = Board()
        multilanding.cached_proof(projection, board, (1, 2, 3, 4), 10, 1, 0.0, 0)
        projection.multilanding_node_limit = 1234
        multilanding.cached_proof(projection, board, (1, 2, 3, 4), 11, 1, 0.0, 0)
    finally:
        multilanding.prove_multilanding = original
    assert seen == [multilanding.MAX_SEARCH_NODES, 1234]
