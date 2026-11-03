"""G_fe段1の境界・特徴列・再現停止条件を検証する。"""
import numpy as np
import pytest
import pandas as pd

from scripts.gfe_phase_stage1_20261002 import (
    column_names, features, reproduction, user_phase,
)
from src.exchange_event_features import g_features


@pytest.mark.parametrize("own,opp,expected", [
    (18, 18, 0), (18, 19, 1), (19, 18, 1), (47, 18, 1),
    (18, 47, 1), (48, 18, 2), (18, 48, 2), (47, 48, 2),
    (18, np.nan, 0), (19, np.nan, 1), (47, np.nan, 1), (48, np.nan, 2),
])
def test_user_phase(own: float, opp: float, expected: int) -> None:
    """両側の境界と欠測時の自側への復帰を固定する。"""
    phase, missing = user_phase(np.array([own / 72]), np.array([(own-opp) / 72]))
    assert phase[0] == expected
    assert missing[0] == float(np.isnan(opp))


@pytest.mark.parametrize("variant,count", [("V0", 59), ("V1", 60), ("V1b", 54), ("V2", 192), ("V3", 54)])
def test_features(variant: str, count: int) -> None:
    """既存変換の再現と全3段階の交互作用を照合する。"""
    design = np.arange(138).reshape(3, 46) / 138
    p, sign = np.array([.2, .5, .8]), np.array([1, -1, 1])
    phase = np.array([[0, 2], [1, 1], [2, 0]])
    actual = features(variant, design, p, sign, phase, np.array([0, 1, 0]))
    assert actual.shape == (3, count)
    assert len(column_names(variant)) == count
    if variant == "V0":
        np.testing.assert_array_equal(actual, g_features(design, p, sign, phase))
    if variant == "V2":
        for stage in range(3):
            np.testing.assert_array_equal(actual[:, 54+46*stage:54+46*(stage+1)],
                                          design * (phase[:, 0] == stage)[:, None])


def test_reproduction_rejects_difference() -> None:
    """不一致を許容幅拡大で通さない。"""
    expected = dict(log_loss=.63, auc=.65)
    assert reproduction(expected, expected)["passed"]
    assert reproduction(dict(log_loss=.630001, auc=.65), expected)["passed"]
    assert not reproduction(dict(log_loss=.6302, auc=.65), expected)["passed"]


def test_missing_own_rejected() -> None:
    """未登録の自側欠測は処理を止める。"""
    with pytest.raises(ValueError):
        user_phase(np.array([np.nan]), np.array([0.]))


def test_bootstrap_preserves_row_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """長短動画を等重みの動画平均へ誤変換しない。"""
    from scripts import gfe_phase_stage1_report_20261002 as report

    monkeypatch.setattr(report, "BOOTSTRAPS", 100)
    y = np.array([1., 1., 1., 0.])
    baseline, candidate = np.full(4, .5), np.array([.8, .8, .8, .8])
    actual = report.paired_ll(y, baseline, candidate, np.array([0, 0, 0, 1]), 2)
    expected = -np.log(np.array([.8, .8, .8, .2])).mean() + np.log(.5)
    assert actual["delta"] == pytest.approx(expected)
    assert actual == report.paired_ll(y, baseline, candidate, np.array([0, 0, 0, 1]), 2)


def test_v0_mismatch_stops_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """V0不一致時には候補データ読み込みも学習も起動しない。"""
    from scripts import gfe_phase_stage1_20261002 as experiment
    from scripts import gfe_phase_stage1_report_20261002 as report

    written = []
    monkeypatch.setattr(report, "write_summary", written.append)
    def reject_read(*args: object) -> None:
        raise AssertionError("不一致後に候補の入力を読んだ")
    monkeypatch.setattr(experiment, "datasets", reject_read)
    stopped = dict(status="STOP_V0_MISMATCH")
    experiment.finish(stopped, pd.DataFrame(), 0.)
    assert written == [stopped]
