"""E13実観測を使い、F1b学習とv2評価器の全追加列を照合する。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import train_exchange_event_models_v2_20260927 as train
from scripts import audit_retrain_f1b_20260927 as f1b
from src.exchange_event_count_features import CountObservation, NEW_COLUMNS
from src.exchange_event_evaluator import (FileExchangeModels, StaticInput, FiringInput,
    ExchangeEndInput, count_sides, build_features, evaluate_exchange_event)
from src.exchange_event_features import D_COLUMNS, SIDE_COLUMNS
from src.exchange_event_tracker import ExchangeEventTracker

ROWS = json.loads((Path(__file__).parent / "fixtures/e14_prefire_rows.json").read_text())


def inputs(row: dict, side: int = 0) -> tuple[dict, FiringInput, ExchangeEndInput]:
    """学習形式と実行形式へ同じ保存行を独立に組み立てる。"""
    pre = row["pre"]
    grids = np.array([v["grid"] for v in pre], dtype=np.int8)
    queues = np.array([v["queue"] for v in pre])
    before = np.array([v["score"] for v in pre], dtype=float)
    after = np.array(row["after"], dtype=float)
    raw = dict(grids=np.concatenate((grids, np.zeros_like(grids))),
        score=np.r_[before, after], game_idx=np.full(4, row["game"]),
        t_sec=np.array([*(v["sec"] for v in pre), row["post_sec"], row["post_sec"]]))
    raw.update({key: np.r_[queues[:, i], [0, 0]] for i, key in enumerate(train.QUEUE)})
    elapsed = row["trigger"]-row["start"]
    static = StaticInput(np.zeros(len(D_COLUMNS)), .5, elapsed, side)
    event = FiringInput(static, np.zeros((2, len(SIDE_COLUMNS))), tuple(row["firing"]),
                        CountObservation(grids, queues, elapsed))
    return raw, event, ExchangeEndInput(event, before, after, elapsed)


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"g{r['game']}_{r['trigger']:.2f}")
@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("stage", ["S1", "S3"])
def test_e13_training_runtime_columns(row: dict, side: int, stage: str) -> None:
    """17列の値・視点・列順を学習パイプラインと独立に照合する。"""
    raw, firing, ended = inputs(row, side)
    pre, post = np.array([0, 1]), np.array([2, 3])
    args = (pre, post, row["trigger"], {row["game"]: row["start"]}, np.array(row["firing"]))
    expected = (train.event_features(raw, *args)[0] if stage == "S1"
                else f1b.corrected_event(raw, *args))
    event = firing if stage == "S1" else ended
    columns = []
    models = SimpleNamespace(count_features=True, elapsed_thresholds=(30., 60.),
        predict_source_probability=lambda name, x: (columns.append(x), .5)[1])
    evaluate_exchange_event(event, models)
    _, base, _ = build_features(event, models.elapsed_thresholds)
    np.testing.assert_allclose(columns[0], np.r_[base, train.orient(expected, 1-2*side)], atol=1e-12)
    assert NEW_COLUMNS == train.NEW_COLUMNS


@pytest.mark.parametrize("field", ["grids", *train.QUEUE, "t_sec"])
def test_s3_future_rows_cannot_change_features(field: str) -> None:
    """未来盤面・NEXT・時刻が混入しない。"""
    raw, firing, event = inputs(ROWS[0])
    raw[field][2:] = 5
    expected = f1b.corrected_event(raw, np.array([0, 1]), np.array([2, 3]),
        ROWS[0]["trigger"], {ROWS[0]["game"]: ROWS[0]["start"]}, np.array(ROWS[0]["firing"]))
    np.testing.assert_allclose(count_sides(event), expected)


def test_runtime_keeps_original_observation() -> None:
    """更新・追加参加でも発火前入力と換算時刻を維持する。"""
    _, event, _ = inputs(ROWS[0])
    tracker = ExchangeEventTracker(SimpleNamespace(count_features=True))
    tracker.firing, tracker.current = event, SimpleNamespace(chains=[])
    tracker.refresh_features(StaticInput(np.ones(len(D_COLUMNS)), .8, 100.),
                             np.ones((2, len(SIDE_COLUMNS))))
    tracker._refresh_firing(event.static, np.ones((2, len(SIDE_COLUMNS))), 100.)
    assert tracker.firing is event
    with pytest.raises(ValueError):
        event.count_observation.grids[:] = 0


def test_real_model_dimensions_and_common_gfe() -> None:
    """RTがS1′/S3′軽量版を選び、G_feはv1と同じ成果物を使う。"""
    models = FileExchangeModels.load("models/exchange_event_v2")
    old = FileExchangeModels.load("models/exchange_event_v1")
    assert models.count_features and not old.count_features
    for name in ("S1", "S3"):
        assert models.estimators[name].n_features_in_ == old.estimators[name].n_features_in_+len(NEW_COLUMNS)
    event = StaticInput(np.zeros(len(D_COLUMNS)), .5, 0.)
    assert evaluate_exchange_event(event, models) == evaluate_exchange_event(event, old)


def test_panel_shows_unsaturated_raw_counts() -> None:
    """パネルはlog値や0〜1値でなく、送れる個数と符号付き余地を表示する。"""
    from tests.test_e11_review_data_panel import fixture_overlay, review_row
    from scripts.review_data_panel import panel_lines, FONT_SIZE, PADDING
    from scripts.visualize_advantage_overlay import _font

    overlay = fixture_overlay()
    values = np.array([[72, 144, 216, 288, 360, -150], [1190, 1190, 1190, 1190, 1190, 20]], dtype=float)
    overlay.tracker.count_sides = np.sign(values) * np.log1p(abs(values))
    row = review_row(overlay)
    assert row["1P_NF_ojama_k5"] == pytest.approx(360.)
    assert row["1P_counter_margin"] == pytest.approx(-150.)
    assert row["2P_counter_margin"] == pytest.approx(20.)
    _, columns = panel_lines(row, {})
    assert any("k5:360" in line and "-150個" in line for line in columns[0])
    font = _font(FONT_SIZE)
    assert max(font.getlength(line) for line in columns[1][7:9]) <= 640 - 2 * PADDING
    assert "k3:1190" in columns[1][7]


def test_v2_rejects_missing_observation() -> None:
    """必要な追加特徴をゼロで埋めて誤ってv2へ渡さない。"""
    event = FiringInput(StaticInput(np.zeros(len(D_COLUMNS)), .5, 0.),
                        np.zeros((2, len(SIDE_COLUMNS))), (True, False))
    models = SimpleNamespace(count_features=True, elapsed_thresholds=(30., 60.))
    with pytest.raises(ValueError, match="発火前盤面"):
        evaluate_exchange_event(event, models)


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"g{r['game']}_{r['trigger']:.2f}")
def test_overlay_selects_e13_prefire_rows(row: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """検知遅延で未来STABLEがあっても、配線が発火前の保存行を選ぶ。"""
    from src import exchange_event_overlay as adapter
    from src.board import Board

    _, expected, _ = inputs(row)
    captured = []
    model = SimpleNamespace(count_features=True)
    overlay = adapter.ExchangeEventOverlay(model, lambda b, s, t, p: expected.static, lambda *a: None)
    overlay.tracker = SimpleNamespace(models=model, current=None, fire=lambda **kw: captured.append(kw))
    overlay._start = row["start"]
    overlay._snapshots = [(min(v["sec"] for v in row["pre"]), object())]
    for idx, value in enumerate(row["pre"]):
        overlay._history[idx] = [adapter.ConfirmedSide(value["sec"], Board.from_list(value["grid"]),
            np.array(value["queue"])), adapter.ConfirmedSide(row["trigger"]+.1, Board(), np.zeros(4))]
    monkeypatch.setattr(adapter, "prefire_side_features", lambda *a: np.zeros(len(SIDE_COLUMNS)))
    trigger = row["trigger"]
    event = SimpleNamespace(trigger_sec=trigger, mechanism="formula", chain_count=1, total_score=40)
    sides = [SimpleNamespace(chain_event=event if active else None) for active in row["firing"]]
    fresh = [(idx, trigger) for idx, active in enumerate(row["firing"]) if active]
    triggers = tuple(trigger if active else None for active in row["firing"])
    overlay._fire(SimpleNamespace(p1=sides[0], p2=sides[1]), object(), trigger+.2, triggers, fresh)
    actual = captured[0]["count_observation"]
    np.testing.assert_array_equal(actual.grids, expected.count_observation.grids)
    np.testing.assert_array_equal(actual.queues, expected.count_observation.queues)
    assert actual.elapsed_sec == expected.count_observation.elapsed_sec
