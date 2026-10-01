"""係数学習の識別不能とlogit補正の向きを検証する。"""
import numpy as np
from pathlib import Path

from scripts.prefire_v6_train import fit, displayed_samples


def test_zero_delta_not_claimed_as_learned() -> None:
    table = np.tile([0., 1., .5, .5, 0., 1., 0., 0., 0.], (20, 1))
    result = fit(table, np.ones(20), np.full(20, 1/20))
    assert result['nonzero_delta'] == 0
    assert not result['identifiable']
    assert result['coefficients'] == [0.]*5


def test_helpful_prediction_strength_increases() -> None:
    table = np.tile([0., 1., .5, .8, 1., 1., 0., 0., 0.], (20, 1))
    result = fit(table, np.ones(20), np.full(20, 1/20))
    assert sum(result['coefficients'][:2]) > 0


def test_harmful_prediction_strength_decreases() -> None:
    table = np.tile([0., 1., .5, .8, 1., 1., 0., 0., 0.], (20, 1))
    result = fit(table, np.zeros(20), np.full(20, 1/20))
    assert sum(result['coefficients'][:2]) < 0


def test_warmup_rows_do_not_enter_training(tmp_path: Path) -> None:
    """前区間のウォームアップと表示されなかった行を重複して学習しない。"""
    rows = np.asarray([[0, 1, .5, .9], [1, 1, .5, .6], [2, 1, .5, .8]])
    np.savez(tmp_path/'prefire_v6_features.npz', values=rows)
    np.savez(tmp_path/'display.npz', t_sec=[1., 3.])
    selected, displayed = displayed_samples(tmp_path)
    np.testing.assert_array_equal(selected, rows[1:2])
    np.testing.assert_array_equal(displayed, [1., 3.])
