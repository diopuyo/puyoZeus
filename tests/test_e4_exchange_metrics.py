"""E4の同値区間・ゲート・撤回帰属を手計算例で検証する。"""
import numpy as np
import pytest

from scripts.aggregate_e4_exchange_eval_20260926 import freshness_gate, m1_events, m2_targets
from src.display_freshness import display_freshness


@pytest.mark.parametrize("values,same,longest,updates", [
    ([1, 1, 2, 2, 2], .75, .1, 1), ([1, 1, 1], 1., .1, 0),
    ([1, 2, 3], 0., 1 / 30, 2), ([1], 0., 1 / 30, 0),
])
def test_freshness_counts(values: list, same: float, longest: float, updates: int) -> None:
    result = display_freshness(np.asarray(values), 30)
    assert result["equal_fraction"] == same
    assert result["longest_equal_seconds"] == pytest.approx(longest)
    assert result["updates"] == updates
    assert result["updates_per_minute"] == pytest.approx(updates / (len(values) / 1800))


@pytest.mark.parametrize("values", [[], [np.nan], [[1, 2]], [np.inf]])
def test_invalid_freshness_rejected(values: list) -> None:
    with pytest.raises(ValueError):
        display_freshness(np.asarray(values), 30)


@pytest.mark.parametrize("same,seconds,expected", [(.55, 3., True), (.5501, 3., False), (.5, 3.1, False)])
def test_both_freshness_conditions_required(same: float, seconds: float, expected: bool) -> None:
    result = dict(off=dict(equal_fraction=.5, longest_equal_seconds=3),
                  on=dict(equal_fraction=same, longest_equal_seconds=seconds))
    assert freshness_gate(result) == expected


def test_continuation_revoke_before_its_s3_is_not_false_s3() -> None:
    event = dict(exchange_id=1, game_idx=0, values=[dict(source="S3", t_sec=2)],
        landings=[], chains=[dict(observed_sec=1, end_signals=[dict(t_sec=1.5)]),
            dict(observed_sec=3, end_signals=[dict(t_sec=4, revoked_sec=5)])])
    summary, _ = m1_events([event])
    assert summary["revoked"] == 0 and summary["continued"] == 1
    assert summary["pass_threshold"] is True
    event["values"].append(dict(source="S3", t_sec=4.5))
    summary, _ = m1_events([event])
    assert summary["revoked"] == 1 and summary["continued"] == 1
    assert summary["pass_threshold"] is False


def test_m2_target_uses_first_g_after_physical_landing() -> None:
    event = dict(exchange_id=1, game_idx=0, trigger_sec=0, closed_sec=2,
                 landings=[dict(t_sec=3)], values=[dict(source="G_fe", t_sec=2, p1=.1)])
    display = dict(t_sec=np.arange(5), source=np.array(["S1", "S3", "G_fe", "G_fe", "G_fe"]),
                   game_idx=np.zeros(5), display_p1=np.array([.5, .6, .1, .8, .9]))
    result = m2_targets([event], display)
    assert result[0]["values"] == [dict(source="G_fe", t_sec=3, p1=.8)]
    assert event["values"][0]["t_sec"] == 2
