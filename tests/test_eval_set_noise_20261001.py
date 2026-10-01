"""評価セットのゆらぎ集計 (scripts/eval_set_noise_20261001) の単体テスト。"""
from __future__ import annotations

import numpy as np
import pytest

from scripts import eval_set_noise_20261001 as noise


def test_aggregate_mean_and_ratio() -> None:
    assert noise.aggregate(np.array([1., 2., 3.])) == pytest.approx(2.)
    ratio = np.array([[1., 2.], [3., 6.], [0., 2.]])
    assert noise.aggregate(ratio) == pytest.approx(4 / 10)
    boot = ratio[np.array([[0, 0, 0], [1, 2, 2]])]
    assert noise.aggregate(boot) == pytest.approx([.5, 3 / 10])


def test_paired_identical_has_zero_diff_and_se() -> None:
    values = np.linspace(.3, .9, 57)
    result = noise.paired(values, values.copy(), noise.bootstrap_index(57))
    assert result['diff'] == 0 and result['se'] == 0 and result['mdd'] == 0


def test_paired_constant_shift_is_exact() -> None:
    values = np.linspace(.3, .9, 57)
    result = noise.paired(values + .01, values, noise.bootstrap_index(57))
    assert result['diff'] == pytest.approx(.01)
    assert result['ci95'] == pytest.approx([.01, .01])


def test_seed_spread_and_mdd_formula() -> None:
    result = noise.seed_spread([.50, .51, .52, .53, .54])
    sd = np.std([.50, .51, .52, .53, .54], ddof=1)
    assert result['sd'] == pytest.approx(sd)
    assert result['range'] == pytest.approx(.04)
    assert result['mdd_single'] == pytest.approx(noise.Z_SUM * np.sqrt(2) * sd)
    assert result['mdd_mean5'] == pytest.approx(noise.Z_SUM * np.sqrt(2 / 5) * sd)


def test_bootstrap_index_is_reproducible() -> None:
    assert np.array_equal(noise.bootstrap_index(57), noise.bootstrap_index(57))


def test_per_game_cp_is_mean_of_three_checkpoints() -> None:
    rows = [dict(m_cp25=.1, m_cp50=.2, m_cp75=.6, late_hits=1, late_frames=2, loss_sum=3., frames=4, m_time=.5)]
    assert noise.per_game(rows, 'm_cp') == pytest.approx([.3])
    assert noise.per_game(rows, 'm_agree').tolist() == [[1., 2.]]
    assert noise.per_game(rows, 'm_time') == pytest.approx([.5])
