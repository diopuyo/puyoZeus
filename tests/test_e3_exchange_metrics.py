"""事前登録指標の分母・打切り・反転を小さな手計算例で確認する。"""
import numpy as np
import pytest

from scripts.aggregate_e3_exchange_eval_20260926 import (
    finite_json, m1_events, m2_events, m2_summary, m3_scores, m4_stability, response_delay,
)


def display(probabilities: list[float], times: list[float] | None = None) -> dict:
    """1P確率の小さな表示系列。"""
    count = len(probabilities)
    return dict(display_p1=np.asarray(probabilities), t_sec=np.asarray(times or list(range(count))),
                display_adv=np.asarray(probabilities) * 200 - 100, game_idx=np.zeros(count, dtype=int))


@pytest.mark.parametrize("probabilities,target,expected", [
    ([.2, .4, .5, .8], .8, 1), ([.8, .6, .5, .2], .2, 1),
    ([.2, .7, .7, .7], .8, 0), ([.2, .3, .4, .4], .8, np.inf),
    ([.2, .2, .2, .2], .2, 0),
])
def test_response_delay_halfway(probabilities: list[float], target: float, expected: float) -> None:
    assert response_delay(display(probabilities), 1, 4, target) == expected


def test_response_delay_next_exchange_is_excluded() -> None:
    assert np.isinf(response_delay(display([.2, .3, .8, .8]), 1, 2, .8))


def test_m1_s3_missing_and_repeated_revocations_keep_denominators() -> None:
    events = [dict(exchange_id=1, game_idx=0, values=[dict(source="S3", t_sec=2)],
                   landings=[dict(t_sec=3)], chains=[dict(observed_sec=1,
                       end_signals=[dict(revoked_sec=4), dict(revoked_sec=5)])]),
              dict(exchange_id=2, game_idx=0, values=[], landings=[], chains=[])]
    summary, rows = m1_events(events)
    assert summary["exchanges"] == 2
    assert summary["s3_exchanges"] == summary["paired"] == summary["early"] == 1
    assert summary["before_landing_fraction"] == 1
    assert summary["early_fraction"] == 1
    assert rows[0]["delta_sec"] == -1


def test_m2_uses_finalize_not_provisional_ready_and_counts_censoring() -> None:
    events = [dict(exchange_id=1, game_idx=0, trigger_sec=0,
                   chains=[dict(score_finalize_sec=1, score_ready_sec=0)],
                   values=[dict(source="G_fe", p1=.8, t_sec=2)]),
              dict(exchange_id=2, game_idx=0, trigger_sec=3,
                   chains=[dict(score_finalize_sec=None, score_ready_sec=3)], values=[])]
    series = dict(off=display([.2, .3, .3, .8]), on=display([.2, .5, .8, .8]))
    summary, rows = m2_events(events, series)
    assert summary["eligible"] == 1 and summary["missing_finalize"] == 1
    assert summary["unreached"] == dict(off=1, on=0)
    assert summary["median_seconds"] == dict(off=np.inf, on=0)
    assert rows[0]["end_sec"] == 1


def test_m3_scores_dense_labeled_frames_and_phase_counts() -> None:
    series = display([.5] * 8, [frame / 30 for frame in range(8)])
    result = m3_scores(series, [dict(start=1, end=7, winner="1P")])
    assert result["all_frames"] == 8 and result["unlabeled_frames"] == 2
    assert result["groups"]["all"]["frames"] == 6
    assert result["groups"]["all"]["matches"] == 1
    assert result["groups"]["all"]["log_loss"] == pytest.approx(np.log(2))
    assert result["groups"]["all"]["auc"] is None
    assert [result["groups"][f"P_time_{p}"]["frames"] for p in (1, 2, 3)] == [2, 2, 2]


def test_m4_zero_crossing_counts_but_match_boundary_does_not() -> None:
    series = display([0, .5, 1, 0, .5, 0])
    series["game_idx"] = np.array([0, 0, 0, 1, 1, 1])
    result = m4_stability(series)
    assert result["frames"] == 6 and result["matches"] == 2
    assert result["saturated_frames"] == 4
    assert result["flips"] == 1
    assert result["flips_per_minute"] == 300


def test_json_preserves_infinite_delay_and_numpy_counts() -> None:
    assert finite_json(dict(n=np.int64(3), delay=np.inf)) == dict(n=3, delay="Infinity")


def test_pooled_median_uses_events_including_infinity() -> None:
    rows = [dict(delays=dict(off=0., on=1.)), dict(delays=dict(off=np.inf, on=2.)),
            dict(delays=dict(off=np.inf, on=np.inf)), dict(missing="score_finalize")]
    result = m2_summary(rows)
    assert result["exchanges"] == 4 and result["eligible"] == 3
    assert result["median_seconds"] == dict(off=np.inf, on=2.)
    assert result["pass_threshold"] is True
