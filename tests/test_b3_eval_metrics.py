"""B3 採点 (LL/Brier/ECE/重み/null/bootstrap/オラクル) のテスト。"""
import numpy as np
import pandas as pd
import pytest

from src.b3_eval_metrics import (STATUS_HOLD, STATUS_NULL, STATUS_OK, apply_null, average_over_seeds, clip_p, ece,
                                 game_values, match_equal_mean, null_report, oracle_exceeded, paired_bootstrap,
                                 row_brier, row_log_loss, score_predictions, video_equal_mean)


def test_clip_and_logloss_values():
    p = np.array([0.0, 1.0, 0.5])
    y = np.array([1.0, 0.0, 1.0])
    c = clip_p(p)
    assert c[0] == pytest.approx(1e-6) and c[1] == pytest.approx(1 - 1e-6)
    ll = row_log_loss(y, p)
    assert ll[0] == pytest.approx(-np.log(1e-6)) and ll[2] == pytest.approx(np.log(2))
    assert row_brier(np.array([1.0]), np.array([0.25]))[0] == pytest.approx(0.5625)


def test_ece_known_cases():
    assert ece(np.array([1.0, 0.0]), np.array([0.55, 0.55]), np.ones(2)) == pytest.approx(0.05)
    assert ece(np.array([1.0, 1.0]), np.array([0.05, 0.95]), np.ones(2)) == pytest.approx((0.95 + 0.05) / 2)
    perfect = ece(np.array([1.0, 0.0, 1.0, 1.0]), np.array([1.0, 0.0, 1.0, 1.0]), np.ones(4))
    assert perfect == pytest.approx(0.0, abs=2e-6)


def test_null_scored_at_half_and_reported():
    p = np.array([0.9, 0.1, 0.8])
    st = np.array([STATUS_OK, STATUS_NULL, STATUS_HOLD])
    assert apply_null(p, st).tolist() == [0.9, 0.5, 0.8]
    r = null_report(st, np.array([1.0, 2.0, 1.0]), np.ones(3))
    assert r["null_rows"] == 1 and r["hold_rows"] == 1 and r["null_rate_sec"] == pytest.approx(0.5)
    assert apply_null(p, None).tolist() == p.tolist()


def _frame(n_games=4, rows=3):
    recs = []
    for g in range(n_games):
        for _ in range(rows):
            recs.append(dict(video_id=f"v{g // 2}", game_idx=g, weight=1.0 / rows, held_sec=1.0, won_p1=float(g % 2)))
    return pd.DataFrame(recs)


def test_game_values_match_equal_and_video_equal():
    df = _frame()
    gmap = {"v0": "A", "v1": "B"}
    vals = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9])
    gv = game_values(df, vals, gmap)
    assert len(gv) == 4 and match_equal_mean(gv) == pytest.approx(0.45)
    assert video_equal_mean(gv) == pytest.approx((0.2 + 0.7) / 2)
    # 動画 v0 に試合を足すと、試合等重みは試合数に比例、動画等重みは動画間で等しい
    extra = df[df.video_id == "v0"].copy()
    extra["game_idx"] += 10
    df2 = pd.concat([df, extra], ignore_index=True)
    gv2 = game_values(df2, np.concatenate([vals, vals[:6]]), gmap)
    assert len(gv2) == 6
    assert video_equal_mean(gv2) == pytest.approx((np.mean([0.1, 0.3, 0.1, 0.3]) + 0.7) / 2)


def test_game_values_stratum_renormalizes_within_game():
    df = _frame(n_games=2, rows=2)
    df["weight"] = [0.8, 0.2, 0.5, 0.5]
    vals = np.array([1.0, 0.0, 0.0, 0.0])
    gv = game_values(df, vals, {"v0": "A"}, mask=np.array([True, False, True, True]))
    assert sorted(gv["value"].tolist()) == [0.0, 1.0]


def test_group_missing_raises():
    with pytest.raises(KeyError):
        game_values(_frame(), np.zeros(12), {"v0": "A"})


def _gv(vals, groups):
    return pd.DataFrame({"video_id": [f"v{i}" for i in range(len(vals))], "game_idx": 0, "group": groups,
                         "value": vals, "n_rows": 1, "weight_sum": 1.0, "held_sec": 1.0})


def test_bootstrap_deterministic_and_zero_width_on_constant_diff():
    groups = [f"g{i}" for i in range(8)]
    c, r = _gv([0.5] * 8, groups), _gv([0.6] * 8, groups)
    b1, b2 = paired_bootstrap(c, r, 500, 1003), paired_bootstrap(c, r, 500, 1003)
    assert b1 == b2
    assert b1["point"] == pytest.approx(-0.1)
    assert b1["lo"] == pytest.approx(-0.1) and b1["hi"] == pytest.approx(-0.1)


def test_bootstrap_resamples_groups_not_games():
    groups = ["big"] * 10 + ["s1", "s2", "s3"]
    diffs = [0.1] * 10 + [-0.1] * 3
    c = _gv(diffs, groups)
    r = _gv([0.0] * 13, groups)
    b = paired_bootstrap(c, r, 2000, 1003)
    assert b["n_groups"] == 4 and b["n_games"] == 13
    assert b["lo"] < b["point"] < b["hi"] and b["hi"] - b["lo"] > 0.1


def test_bootstrap_reports_unpaired_games():
    c = _gv([0.1, 0.2, 0.3], ["a", "b", "c"])
    r = _gv([0.1, 0.2], ["a", "b"])
    b = paired_bootstrap(c, r, 100, 1)
    assert b["n_games"] == 2 and b["games_only_cand"] == 1 and b["games_only_ref"] == 0


def test_average_over_seeds():
    a = _gv([0.2, 0.4], ["a", "b"])
    b = _gv([0.4, 0.6], ["a", "b"])
    assert average_over_seeds([a, b])["value"].tolist() == pytest.approx([0.3, 0.5])


def test_score_predictions_perfect_vs_null():
    df = _frame()
    gmap = {"v0": "A", "v1": "B"}
    y = df["won_p1"].to_numpy()
    good = score_predictions(df, np.where(y == 1, 0.99, 0.01), gmap)
    assert good["ll"] < 0.02 and good["population"]["games"] == 4
    st = np.full(len(df), STATUS_NULL)
    nul = score_predictions(df, np.where(y == 1, 0.99, 0.01), gmap, status=st)
    assert nul["ll"] == pytest.approx(np.log(2)) and nul["null"]["null_rows"] == len(df)


def test_oracle_exceeded():
    assert oracle_exceeded(0.70, 0.72) is True
    assert oracle_exceeded(0.73, 0.72) is False
    assert oracle_exceeded(0.70, None) is None
