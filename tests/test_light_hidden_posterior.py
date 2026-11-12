"""因果事前、尤度、同点集約と95%・K制限を独立に確認する。"""
from types import SimpleNamespace
import numpy as np
import pytest

from src.light_hidden_posterior import HiddenPrior, posterior, total_groups, scenarios
from src.light_refind import Candidate, SIMULATOR
from src.probabilistic_board import ProbabilisticCell


def physical(weights: list[float], scores: tuple = (40, 80, 120)) -> SimpleNamespace:
    board = np.zeros((13, 6), dtype=np.int8)
    candidates = [Candidate(board, (score,), board, weight) for weight, score in zip(weights, scores)]
    return SimpleNamespace(candidates=candidates, evidence=SimpleNamespace(
        retained=list(range(len(candidates))), scores=np.zeros(len(candidates))),
        rejected=False, active=True, observed=(), chain_id=1.)


def test_prior_times_likelihood_and_no_resurrection() -> None:
    value = physical([.8, .2])
    value.evidence.scores[:] = np.log([.25, 1.])
    assert [r['weight'] for r in posterior(value)] == pytest.approx([.5, .5])
    value.evidence.retained = [1]
    assert [r['index'] for r in posterior(value)] == [1]


def test_same_total_is_aggregated_before_selection() -> None:
    value = physical([.3, .3, .4], (40, 40, 80))
    groups = total_groups(value)
    assert [r['total'] for r in groups] == [40, 80]
    assert [r['weight'] for r in groups] == pytest.approx([.6, .4])
    selected = scenarios([SimpleNamespace(_light_physics=value), SimpleNamespace()])
    assert len(selected) == 2 and sum(r['weight'] for r in selected) == pytest.approx(1.)


def test_cutoff_and_limit_keep_omitted_mass_visible() -> None:
    value = physical([.96, .03, .01])
    owner = SimpleNamespace(_light_physics=value)
    selected = scenarios([owner, SimpleNamespace()])
    assert len(selected) == 1 and selected[0]['retained_mass'] == pytest.approx(.96)
    value = physical([.4, .35, .25])
    selected = scenarios([SimpleNamespace(_light_physics=value), SimpleNamespace()], limit=2)
    assert len(selected) == 2 and selected[0]['retained_mass'] == pytest.approx(.75)


def test_single_total_has_no_extra_evaluation() -> None:
    value = physical([.5, .5], (40, 40))
    assert scenarios([SimpleNamespace(_light_physics=value), SimpleNamespace()]) == []


def test_freeze_does_not_see_later_hidden_state() -> None:
    value = HiddenPrior(SIMULATOR)
    board = np.zeros((13, 6), dtype=np.int8)
    board[1:, 0] = 1
    value.belief.advance(board, (1, 2, 3, 4), 1.)
    value.freeze()
    value.belief.cells[0] = ProbabilisticCell.certain(2)
    weight, uniform = value.weights(board, (0, 0, 0, 0, 0, 0), (1, 2, 3, 4))
    assert weight == uniform == .5


def test_impossible_prior_is_not_replaced_by_uniform() -> None:
    value = physical([0., 1.])
    assert [r['index'] for r in posterior(value)] == [1]
