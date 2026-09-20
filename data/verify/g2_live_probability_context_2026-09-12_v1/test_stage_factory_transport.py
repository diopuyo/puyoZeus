"""明示factory輸送は実接続同一性を要求し、constructor資格を生成しない。"""
from __future__ import annotations
from types import SimpleNamespace as N
from typing import Any
import pytest
import probability_stage as S


def data() -> tuple[Any, dict[str, Any]]:
    factory = object()
    connection = N(recovery=N(factory=factory))
    return factory, dict(probabilistic_basis_connection=connection,
                         probabilistic_tracking_mode=N(connection=connection))


def test_explicit_actual_without_fake_live_state() -> None:
    factory, state = data()
    before = dict(state)
    assert S.actual_factory(state, factory) is factory
    assert state == before and all(state[k] is v for k, v in before.items())
    assert 'private_suffix_factory' not in state
    with pytest.raises(KeyError, match='private_suffix_factory'):
        S.actual_factory(state)


def test_original_live_lookup_unchanged() -> None:
    factory = object()
    assert S.actual_factory(dict(private_suffix_factory=factory)) is factory


@pytest.mark.parametrize('defect', ['factory', 'connection', 'live_conflict'])
def test_explicit_binding_mismatch_rejected(defect: str) -> None:
    factory, state = data()
    if defect == 'factory':
        factory = object()
    elif defect == 'connection':
        state['probabilistic_tracking_mode'].connection = object()
    else:
        state['private_suffix_factory'] = object()
    with pytest.raises(AssertionError, match='stage_'):
        S.actual_factory(state, factory)
