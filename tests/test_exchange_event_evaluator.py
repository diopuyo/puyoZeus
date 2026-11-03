"""撃ち合い特徴と評価器の入力・視点・保存形式を検証する。"""
from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace
import json
from pathlib import Path

from src.exchange_event_evaluator import (
    StaticInput, FiringInput, ExchangeEndInput, FileExchangeModels,
    MODEL_COLUMNS, MODEL_VERSION, build_features, evaluate_exchange_event, file_sha256,
)

from src.exchange_event_features import (
    D_COLUMNS, G_COLUMNS, SIDE_COLUMNS, arrival_features, fill_phase,
    g_features, prefire_side_features, score_features,
)


@pytest.mark.parametrize("own,diff,want", [(0, 0, 0), (1/3, 0, 0),
    (2/3, 0, 1), (1, 0, 2), (1, 1, 1), (np.nan, 0, 0), (-1, 0, 0)])
def test_fill_phase(own: float, diff: float, want: int) -> None:
    assert fill_phase(own, diff) == want


def test_g_column_order_and_phase_invariance() -> None:
    design = np.zeros((2, len(D_COLUMNS)))
    x = g_features(design, np.array([.8, .8]), np.array([1, -1]), np.array([[0, 2], [0, 2]]))
    assert x.shape == (2, len(G_COLUMNS))
    np.testing.assert_allclose(x[0, 47:53], x[1, 47:53])
    np.testing.assert_allclose(x[0, 53:], -x[1, 53:])
    assert x[0, 46] == pytest.approx(np.log(4))


def test_arrival_orientation() -> None:
    sides = np.arange(2 * len(SIDE_COLUMNS)).reshape(2, -1) / 30
    a = arrival_features(sides, [1, 0])
    b = arrival_features(sides, [1, 0], 1)
    n = len(SIDE_COLUMNS)
    np.testing.assert_equal(a[:n], b[n:2*n])
    np.testing.assert_equal(a[2*n:3*n], -b[2*n:3*n])
    np.testing.assert_equal(a[-2:], b[-2:][::-1])


@pytest.mark.parametrize("elapsed,rate", [(0, 70), (96, 70), (97, 52), (112, 39)])
def test_margin_score(elapsed: float, rate: int) -> None:
    x = score_features([100, 300], [800, 650], elapsed)
    assert x[0] == pytest.approx((700/rate) / (1+700/rate))
    assert x[2] == pytest.approx((350/rate) / (1+350/rate))
    np.testing.assert_equal(x[3:], [1, 0, 0, 0])


@pytest.mark.parametrize("missing", [np.nan, np.inf, -1])
def test_score_missing(missing: float) -> None:
    x = score_features([missing, 0], [700, 70], 0)
    assert np.isnan(x[[0, 2, 3, 4]]).all()
    np.testing.assert_equal(x[-2:], [1, 0])
    assert x[1] == .5


def test_score_negative_delta_and_tie() -> None:
    np.testing.assert_equal(score_features([100, 10], [50, 10], 0), np.zeros(7))


def test_score_swap() -> None:
    a = score_features([0, 0], [700, 140], 0)
    b = score_features([0, 0], [700, 140], 0, 1)
    np.testing.assert_equal(a[:2], b[:2][::-1])
    assert a[2] == -b[2]
    np.testing.assert_equal(a[3:5], b[3:5][::-1])


@pytest.mark.parametrize("elapsed", [-1, np.nan, np.inf])
def test_invalid_elapsed(elapsed: float) -> None:
    with pytest.raises(ValueError):
        score_features([0, 0], [0, 0], elapsed)


def test_first_move_rate() -> None:
    assert score_features([0, 0], [70, 0], 96, from_first_move=True)[0] > .5


def test_prefire_dead_board() -> None:
    grid = np.zeros((13, 6), dtype=int)
    grid[1:, 2] = 1
    result = prefire_side_features(grid, np.zeros(4), 0)
    assert result[1] == pytest.approx(12/78)
    np.testing.assert_equal(result[-5:], np.zeros(5))


@pytest.mark.parametrize("shape", [(12, 6), (13, 7)])
def test_invalid_board(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        prefire_side_features(np.zeros(shape), np.zeros(4), 0)


class FakeModels:
    """非対称な自側モデルで視点変換を検証する。"""

    elapsed_thresholds = (20., 40.)

    def predict_source_probability(self, model_name: str, features: np.ndarray) -> float:
        weights = np.linspace(-.1, .2, len(features))
        return float(1 / (1 + np.exp(-np.nan_to_num(features) @ weights)))


def inputs() -> tuple[StaticInput, FiringInput, ExchangeEndInput]:
    static = StaticInput(np.linspace(0, 1, len(D_COLUMNS)), .7, 30)
    firing = FiringInput(static, np.linspace(0, 1, 2*len(SIDE_COLUMNS)).reshape(2, -1), (True, False))
    end = ExchangeEndInput(firing, [100, 300], [800, 1000], 110)
    return static, firing, end


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("missing", [False, True])
def test_probability_antisymmetry(index: int, missing: bool) -> None:
    static, firing, end = inputs()
    if missing:
        firing = replace(firing, prefire_sides=np.full((2, len(SIDE_COLUMNS)), np.nan))
        end = replace(end, firing=firing, scores_after=[np.nan, 1000])
    swapped_static = replace(static, source_side=1, m0_probability_1p=1-static.m0_probability_1p)
    swapped_firing = replace(firing, static=swapped_static,
                            prefire_sides=firing.prefire_sides[::-1], firing=firing.firing[::-1])
    swapped_end = replace(end, firing=swapped_firing, scores_before=end.scores_before[::-1],
                          scores_after=end.scores_after[::-1])
    p = evaluate_exchange_event((static, firing, end)[index], FakeModels())
    q = evaluate_exchange_event((swapped_static, swapped_firing, swapped_end)[index], FakeModels())
    assert p == pytest.approx(1-q, abs=1e-14)


def test_unconfirmed_scores_keep_s1() -> None:
    _, firing, end = inputs()
    assert evaluate_exchange_event(replace(end, scores_confirmed=False), FakeModels()) == evaluate_exchange_event(firing, FakeModels())
    assert build_features(end, (20., 40.))[0] == "S3"


@pytest.mark.parametrize("probability", [-.1, 1.1, np.nan, np.inf])
def test_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError):
        StaticInput(np.zeros(len(D_COLUMNS)), probability, 0)


@pytest.mark.parametrize("side", [-1, 2])
def test_invalid_source_side(side: int) -> None:
    with pytest.raises(ValueError):
        StaticInput(np.zeros(len(D_COLUMNS)), .5, 0, side)


def test_d_missing_and_immutable() -> None:
    raw = np.full(len(D_COLUMNS), np.nan)
    event = StaticInput(raw, .5, 0)
    raw[:] = 1
    assert np.isnan(event.d_features).all()
    np.testing.assert_equal(build_features(event, (20., 40.))[1][:len(D_COLUMNS)], np.zeros(len(D_COLUMNS)))
    with pytest.raises(ValueError):
        event.d_features[0] = 1


@pytest.mark.parametrize("firing", [(False, False), (True,), (1, 0)])
def test_invalid_trigger(firing: tuple) -> None:
    with pytest.raises(ValueError):
        replace(inputs()[1], firing=firing)


def write_models(directory: Path) -> None:
    """実際の保存形式を小さいLRで作る。"""
    import joblib
    from sklearn.linear_model import LogisticRegression

    manifest = dict(version=MODEL_VERSION, elapsed_thresholds=[20, 40], models={})
    for key in ("G_fe", "S1", "S3", "S1_light", "S3_light"):
        name = key.removesuffix("_light")
        columns = MODEL_COLUMNS[name]
        x = np.random.default_rng(0).normal(size=(20, len(columns)))
        model = LogisticRegression().fit(x, np.arange(len(x)) % 2)
        path = directory / (key + ".joblib")
        joblib.dump(model, path)
        manifest["models"][key] = dict(file=path.name, columns=columns, sha256=file_sha256(path))
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("lightweight", [True, False])
def test_file_models(tmp_path: Path, lightweight: bool) -> None:
    write_models(tmp_path)
    models = FileExchangeModels.load(tmp_path, lightweight)
    for event in inputs():
        assert 0 <= evaluate_exchange_event(event, models) <= 1


@pytest.mark.parametrize("corruption", ["version", "columns", "hash", "threshold"])
def test_file_validation(tmp_path: Path, corruption: str) -> None:
    write_models(tmp_path)
    path = tmp_path / "manifest.json"
    meta = json.loads(path.read_text())
    if corruption == "version":
        meta["version"] = "unknown"
    elif corruption == "columns":
        meta["models"]["S1_light"]["columns"].reverse()
    elif corruption == "hash":
        meta["models"]["G_fe"]["sha256"] = "wrong"
    else:
        meta["elapsed_thresholds"] = [40, 20]
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError):
        FileExchangeModels.load(tmp_path)


def test_unknown_input() -> None:
    with pytest.raises(TypeError):
        evaluate_exchange_event(None, FakeModels())


def test_missing_fill_difference_uses_zero_phase() -> None:
    d = np.zeros(len(D_COLUMNS))
    d[D_COLUMNS.index("board_puyo_total")] = 1
    d[D_COLUMNS.index("diff_board_puyo_total")] = np.nan
    _, features, _ = build_features(StaticInput(d, .5, 0), (20., 40.))
    np.testing.assert_equal(features[47:50], [1, 0, 0])
