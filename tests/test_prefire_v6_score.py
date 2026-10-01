"""採点の試合等重み・信頼区間・試合内対照を検証する。"""
import numpy as np
import pytest
import json
from pathlib import Path

from scripts.prefire_v6_score import paired, placebo, game_metrics, false_deaths, join_displays


def test_identical_interval_zero() -> None:
    x = np.asarray([.1, .7, .9])
    assert paired(x, x)['ci95'] == [0., 0.]


def test_equal_game_weight() -> None:
    assert paired(np.asarray([.2, .8]), np.asarray([.1, .7]))['difference'] == pytest.approx(.1)


def test_placebo_stays_inside_game() -> None:
    base = dict(t_sec=np.arange(6.), display_p1=np.full(6, .5))
    actual = dict(base, display_p1=np.asarray([.8, .5, .5, .5, .5, .5]))
    games = [dict(start=0, end=3), dict(start=3, end=6)]
    result = placebo(base, actual, games)
    np.testing.assert_allclose(result['display_p1'][3:], .5)
    assert result['display_p1'][1] == .8


def test_zero_placebo_preserves_exact_probability() -> None:
    """増分0ではlogit往復の丸め誤差を改善・悪化として作り出さない。"""
    base = dict(t_sec=np.arange(100.), display_p1=np.random.default_rng(0).uniform(size=100))
    result = placebo(base, base, [dict(start=0, end=100)])
    np.testing.assert_array_equal(result['display_p1'], base['display_p1'])


def test_placebo_uses_prediction_increment_only() -> None:
    """通常評価器の版差は予測の増分ではない。"""
    base = dict(t_sec=np.arange(3.), display_p1=np.full(3,.5))
    actual = dict(base, display_p1=np.full(3,.6))
    result = placebo(base, actual, [dict(start=0,end=3)], np.zeros(3))
    np.testing.assert_array_equal(result['display_p1'], base['display_p1'])


def test_join_keeps_fractional_boundary_frame() -> None:
    """境界1.01秒の試合には隣の収集区間にある1.00秒表示も含める。"""
    parts = [dict(t_sec=np.asarray([t]), display_p1=np.asarray([p]),
                  display_adv=np.asarray([10.]), game_idx=np.asarray([0])) for t,p in ((0.,.5),(1.,.8))]
    joined = join_displays(parts)
    row = game_metrics(joined, dict(start=0,end=1.01,winner='1P'), np.ones(2,dtype=bool))
    assert row['m_time'] == pytest.approx((-np.log(.5)-np.log(.8))/2)
    assert len(np.unique(joined['game_idx'])) == 2


def test_join_rejects_overlapping_times() -> None:
    with pytest.raises(ValueError):
        join_displays([dict(t_sec=np.asarray([0.])), dict(t_sec=np.asarray([0.]))])


def test_flips_ignore_even_band() -> None:
    display = dict(t_sec=np.arange(6.), display_p1=np.full(6, .5),
                   display_adv=np.asarray([10., 2., -10., -2., 10., 0.]))
    result = game_metrics(display, dict(start=0, end=6, winner='1P'), np.ones(6, dtype=bool))
    assert result['flips'] == 2


def test_false_death_counts_game_side_once() -> None:
    value = dict(t_sec=1, source='unavoidable_death', dead_sides=['1P'])
    events = [dict(values=[value, value])]
    assert false_deaths(events, [dict(game=1, start=0, end=2, winner='1P')]) == {(1, '1P')}


def test_regression_uses_original_record_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """分割セット1の相対時刻を、元の全長zenchi記録の勝敗へ混ぜない。"""
    from scripts import prefire_v6_score as score
    monkeypatch.setattr(score, 'BASELINE_DIRS', {'zenchi': None})
    monkeypatch.setattr(score, 'OUT', tmp_path/'out')
    monkeypatch.setattr(score, 'EXEV', tmp_path/'original')
    monkeypatch.setattr(score, 'EVALSET', tmp_path/'unused_split_labels')
    label = score.EXEV/'logs/review_zenchi_part3/official_games.json'
    label.parent.mkdir(parents=True)
    label.write_text(json.dumps([dict(start=100, end=200, winner='1P')]))
    replay = score.OUT/'replay/zenchi'
    replay.mkdir(parents=True)
    np.savez(replay/'prefire_v6_features.npz',
             columns=['t_sec', 'game_idx', 'proof_1p', 'proof_2p'],
             values=[[10, 1, 0, 1], [150, 4, 0, 1], [170, 5, 0, 1]])
    np.savez(replay/'display.npz', t_sec=[10, 170])
    assert score.regression_predictions() == {('zenchi', 5, '1P')}
