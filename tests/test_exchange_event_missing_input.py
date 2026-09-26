"""E3b: 不完全な観測で本番ONを止めず、下位評価と記録を保つ。"""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from src.chain_id_resolver import ChainObservation, ObservationKind
from src.exchange_event_features import SIDE_COLUMNS
from src.exchange_event_overlay import ExchangeEventOverlay
from src.exchange_event_tracker import ExchangeEventTracker
from tests.test_exchange_event_tracker import Models, static, fire
from tests.test_exchange_event_overlay import build_static, Signals, result


def tracker() -> ExchangeEventTracker:
    """静止評価が既にある本番相当の初期状態。"""
    value = ExchangeEventTracker(Models())
    value.boundary(1, 0)
    value.static(static(), 1, (1, 1))
    return value


@pytest.mark.parametrize("triggers", [(2.0, None), (2.0, 3.0), (None, None)])
def test_unknown_firing_keeps_static_and_dumps_count(triggers: tuple, tmp_path: Path) -> None:
    """既読の古い発火が最小時刻となり、左右Falseになる実クラッシュ経路。"""
    value = tracker()
    value._seen.add((1, "1P", 2.0))
    observation = ChainObservation("2P", 3, ObservationKind.FORMULA_STEP, 1, 40)
    for _ in range(2):
        value.fire(t_sec=3, triggers=triggers, static=static(),
                   prefire_sides=np.zeros((2, len(SIDE_COLUMNS))), score_elapsed_sec=3,
                   observations=(observation,))
    assert value.current is None and value.records == []
    assert value.source == "G_fe" and value.probability == .4
    assert value.static(static(), 4, (4, 4))
    value.save(tmp_path / "events.jsonl")
    saved = json.loads((tmp_path / "events.diagnostics.json").read_text())
    assert saved["counts"]["unknown_firing_side"] == 1


@pytest.mark.parametrize("bad", [None, np.nan, np.inf, -1])
def test_missing_trigger_at_overlay_keeps_static(bad: float | None) -> None:
    value = ExchangeEventOverlay(Models(), build_static, Signals, lambda b, q: .5)
    counters = SimpleNamespace(finalized_count_p1=0, finalized_count_p2=0,
                               chain_total_score_p1=0, chain_total_score_p2=0)
    value.update(result(0), object(), counters, 0, 1)
    observed = result(1)
    observed.p1.chain_event.trigger_sec = bad
    value.update(observed, object(), counters, 1, 1)
    assert value.tracker.current is None
    assert value.tracker.source == "G_fe" and value.tracker.probability == .4
    assert value.tracker.diagnostics[-1]["reason"] == "unknown_firing_side"


@pytest.mark.parametrize("prefire", [np.zeros((1, 1)), np.full((2, len(SIDE_COLUMNS)), np.inf)])
def test_bad_prefire_never_leaves_partial_record(prefire: np.ndarray) -> None:
    value = tracker()
    value.fire(t_sec=2, triggers=(2, None), static=static(),
               prefire_sides=prefire, score_elapsed_sec=2)
    assert not value.records and value.current is None
    assert value.source == "G_fe" and value.probability == .4
    assert value.diagnostics


def test_missing_snapshot_does_not_raise_stopiteration() -> None:
    value = ExchangeEventOverlay(Models(), build_static, Signals, lambda b, q: .5)
    counters = SimpleNamespace(finalized_count_p1=0, finalized_count_p2=0,
                               chain_total_score_p1=0, chain_total_score_p2=0)
    value.update(result(0), object(), counters, 0, 1)
    value._snapshots.clear()
    value.update(result(1), object(), counters, 1, 1)
    assert value.tracker.source == "G_fe"
    assert value.tracker.diagnostics[-1]["reason"] == "missing_prefire_snapshot"


def test_invalid_m0_preserves_last_static_probability() -> None:
    value = ExchangeEventOverlay(Models(), build_static, Signals, lambda b, q: .5)
    counters = SimpleNamespace(finalized_count_p1=0, finalized_count_p2=0,
                               chain_total_score_p1=0, chain_total_score_p2=0)
    value.update(result(0), object(), counters, 0, 1)
    value._m0 = lambda b, q: np.nan
    value.update(result(.5), object(), counters, .5, 1)
    assert value.tracker.source == "G_fe" and value.tracker.probability == .4
    assert value.tracker.diagnostics[-1]["requested_stage"] == "G_fe"


def test_missing_s3_delta_retains_s1() -> None:
    value = tracker()
    fire(value)
    value.end("1P", 3, "next")
    value.latest_chain("1P").score_ready_sec = 3
    value.finish_frame(3 + 10 / 30)
    assert value.source == "S1" and value.probability == .6
    assert value.diagnostics[-1]["reason"] == "missing_chain_score"


def test_invalid_model_output_never_starts_s1() -> None:
    value = tracker()
    value.models.predict_source_probability = lambda name, x: np.nan
    fire(value)
    assert value.source == "G_fe" and value.probability == .4
    assert value.records == []
