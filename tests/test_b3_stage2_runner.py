"""段2実行器の仮データ検証。実データ・本物モデル・封印には触れない。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.b3_stage2_score import confirm_changes, metric_intervals, paired_interval, raw_game_scores
from src.b3_stage2_trace import make_display, predict_rows, review_rows, smooth_display, to_adv
from src.b3_train_model import HgbConfig, feature_columns, fit_two_state, load_model, save_model
from src.b3_train_synth import make_rows


@pytest.fixture
def dummy() -> tuple:
    rows, _ = make_rows(n_videos=2, games_per_video=2, rows_per_game=12)
    model = fit_two_state(rows, feature_columns(rows.columns), HgbConfig("dummy", 2, .2, 3, 2), 1003)
    rows["zenchi_set"], rows["zenchi_game"] = 1, rows.game_idx
    rows["t_source_sec"] = rows.t_sec + 100
    return rows, model


def test_model_roundtrip_and_no_overwrite(tmp_path: Path, dummy: tuple) -> None:
    rows, model = dummy
    path = tmp_path / "dummy.joblib"
    save_model(model, path)
    np.testing.assert_array_equal(model.predict(rows), load_model(path).predict(rows))
    with pytest.raises(FileExistsError):
        save_model(model, path)


def test_prediction_original_orientation_and_selected_branch(dummy: tuple) -> None:
    rows, model = dummy
    predicted = predict_rows(rows, model)
    assert len(predicted) == len(rows) // 2
    assert predicted.orientation.eq(0).all()
    assert predicted.loc[predicted.mode_exchange == 1, "p_static"].isna().all()
    assert predicted.loc[predicted.mode_exchange == 0, "p_exchange"].isna().all()
    np.testing.assert_array_equal(predicted.p_selected, model.predict(predicted))


def test_invalid_mode_rejected(dummy: tuple) -> None:
    rows, model = dummy
    rows.loc[rows.orientation == 0, "mode_exchange"] = 2
    with pytest.raises(ValueError, match="状態"):
        predict_rows(rows, model)


def test_held_cap_and_game_boundary_and_independent_death() -> None:
    df = pd.DataFrame(dict(zenchi_game=[1, 2], t_source_sec=[0., 4.],
                           p_selected=[.8, .2], mode=["static", "exchange"]))
    times = np.array([0., 2., 2.1, 3., 4., 7.])
    ref = dict(t_sec=times, source=np.array(["prob"] * 5 + ["confirmed_death"]),
               display_p1=np.array([.5] * 5 + [.99]))
    games = [dict(game=1, start=0, end=3), dict(game=2, start=3, end=8)]
    result = make_display(df, ref, games, [], None)
    np.testing.assert_array_equal(result["p_selected"], [.8, .8, .5, .5, .2, .5])
    assert result["source"][2] == result["source"][3] == "null"
    assert result["display_p1"][-1] == result["p_display"][-1] == .99


def test_smoothing_adv_and_probability_separate() -> None:
    t = np.array([0., 1., 2.])
    adv, p = smooth_display(np.array([.5, .8, .2]), np.array(["static"] * 3),
                            np.array([1, 1, 2]), t, np.array([]), .1)
    assert p[1] == pytest.approx(.575)
    assert adv[1] == pytest.approx(to_adv(.8, .1) * .25)
    assert p[2] == .2


def test_paired_agree_is_ratio_not_mean_of_ratios() -> None:
    c = np.array([[1, 1], [0, 9]])
    r = np.array([[0, 1], [0, 9]])
    result = paired_interval(c, r, ratio=True)
    assert result["point"] == .1
    assert result["bootstrap_n"] == 2000 and result["bootstrap_seed"] == 1003
    assert result == paired_interval(c, r, ratio=True)


def test_metric_game_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        metric_intervals([dict(game=1)], [dict(game=2)])


def test_zero_delays_remain_in_population() -> None:
    before = dict(rows=[dict(game=g, side="1P", first_sec=1., false=False) for g in range(10)])
    after = dict(rows=[dict(r) for r in before["rows"]])
    after["rows"][-1]["first_sec"] += 6
    change = confirm_changes(before, after)
    assert change["common"] == 10 and change["new_delay_over_5s"] == 1
    assert change["delay_p95_diff"] == pytest.approx(3.3)
    after["rows"].pop(0)
    assert confirm_changes(before, after)["lost"] == 1


def test_raw_scoring_does_not_use_display_or_death_override() -> None:
    display = dict(t_sec=np.array([0., 1.]), p_selected=np.array([.5, .5]),
                   display_p1=np.array([1., 1.]), source=np.array(["null", "confirmed_death"]))
    ref = dict(t_sec=display["t_sec"], display_p1=np.array([.5, .5]))
    rows = raw_game_scores(display, ref, [dict(game=1, start=0, end=2, winner="1P")])
    assert rows[0]["ll_c"] == pytest.approx(np.log(2))
    assert rows[0]["null_frames"] == 1 and rows[0]["frames"] == 2


def test_review_uses_keys_and_source_time(dummy: tuple) -> None:
    rows, model = dummy
    predicted = predict_rows(rows, model)
    original = rows[rows.orientation == 0]
    review = review_rows(original.iloc[::-1], predicted)
    assert review[0]["t_sec"] == predicted.t_source_sec.min()
    assert review[0]["p_selected"] is not None
    with pytest.raises(ValueError):
        review_rows(original.iloc[1:], predicted)
