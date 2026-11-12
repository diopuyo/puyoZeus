"""式と矛盾する確定盤を変えず、反復観測した別色を光の候補だけに使う。"""
from types import SimpleNamespace

import numpy as np
import pytest

from src.fire_board import NativeTrace
from src.light_base_history import ObservedBases
from src.light_refind import PhysicalRefind


def board(color: int, tail: int = 3) -> np.ndarray:
    result = np.zeros((13, 6), dtype=np.int8)
    result[12, :4], result[12, 5] = color, tail
    return result


def predictor(normal: bool = False) -> PhysicalRefind:
    result = PhysicalRefind(NativeTrace())
    result.base = board(1)
    if not normal:
        result.base[12, 2] = 2
    result.colors.update((1, 2, 3, 4))
    for start, raw in ((0., board(1)), (.4, board(2, 4))):
        for t in np.linspace(start, start + .3, 6):
            result.base_history.remember(float(t), raw)
    return result


def test_zero_normal_candidates_admits_all_supported_conflicting_boards() -> None:
    owner = predictor()
    before = owner.base.copy()
    owner.start(.7, 40, True)
    assert len(owner.base_versions) == 3
    assert len(owner.candidates) == 2
    assert {int(c.board[12, 5]) for c in owner.candidates} == {3, 4}
    assert all(c.steps == (40,) for c in owner.candidates)
    assert np.array_equal(owner.base, before)
    assert owner.evidence is not None


def test_existing_normal_candidate_does_not_admit_conflicting_board() -> None:
    owner = predictor(normal=True)
    owner.start(.7, 40, True)
    assert len(owner.base_versions) == len(owner.candidates) == 1
    assert owner.candidates[0].board[12, 5] == 3


def test_fallback_still_requires_first_formula_match() -> None:
    owner = predictor()
    owner.start(.7, 500, True)
    assert owner.candidates == [] and owner.evidence is None


def test_fallback_uses_visible_likelihood_and_does_not_revive_candidates() -> None:
    owner = predictor()
    owner.start(.7, 40, True)
    forecast = SimpleNamespace(chain_id=.7, observed=(40,))
    raw = owner.candidates[0].board
    probs = np.eye(7)[raw[1:]] * .999 + .001 / 7
    for t in np.linspace(1.1, 1.4, 6):
        owner.observe(float(t), {'light_cell_probs': probs}, forecast, True)
    assert owner.evidence.retained == [0]
    forecast.observed = (40, 500)
    owner.observe(1.5, {}, forecast, True)
    assert owner.rejected and owner.evidence is None
    forecast.observed = (40,)
    owner.observe(1.6, {}, forecast, True)
    assert owner.rejected and owner.evidence is None


@pytest.mark.parametrize('count,span,query,floating', [
    (5, .3, .3, False), (6, .2, .2, False), (6, .3, 5., False),
    (6, .3, -1., False), (6, .3, .3, True),
])
def test_color_conflict_permission_preserves_observation_gates(
    count: int, span: float, query: float, floating: bool,
) -> None:
    history = ObservedBases()
    raw, base = board(2), board(1)
    if floating:
        raw[10, 0] = 2
    for t in np.linspace(0., span, count):
        history.remember(float(t), raw)
    assert len(history.variants(query, base, allow_color_conflicts=True)) == 1


def test_color_to_empty_is_a_conflict_and_never_changes_confirmed_base() -> None:
    history = ObservedBases()
    base, raw = board(1), board(1)
    raw[12, 0] = 0
    for t in np.linspace(0., .3, 6):
        history.remember(float(t), raw)
    assert len(history.variants(.3, base)) == 1
    variants = history.variants(.3, base, allow_color_conflicts=True)
    assert len(variants) == 2 and variants[1][12, 0] == 0
    variants[1][:] = 0
    assert base[12, 0] == 1 and history.records[0][1][12, 1] == 1
