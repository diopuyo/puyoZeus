"""元確率worldを潰さず未来盤面へ伝播。人工起点であり実origin認証ではない。"""
from dataclasses import replace
import pytest
import belief as B
from test_belief import chain_prior
import projected_view as P


def test_hidden_worlds_reach_different_visible_futures_without_conditioning() -> None:
    value, origin = chain_prior()
    before = P.digest(value)
    projected = P.project(value, value.scope, origin, origin_frame=10, cutoff_frame=12,
        origin_token='artificial-active-origin')
    assert len(projected.outcomes) == 2
    assert sorted(row.weight for row in projected.outcomes) == pytest.approx([0.3, 0.7])
    assert len({row.grid[B.HIDDEN_ROWS:] for row in projected.outcomes}) == 2
    assert all(row.chain_count == 1 and row.raw_chain_score == 40 for row in projected.outcomes)
    assert P.digest(value) == before and value.tokens == ()
    assert projected.provenance == 'physics_projected' and projected.source_digest == before
    assert not projected.native_consumption_applied and not projected.accounting_connected
    assert not projected.source_producer_authorized and not projected.quality_gate_clear


@pytest.mark.parametrize('fault', ['scope', 'clock', 'token', 'origin'])
def test_no_scope_or_origin_fabrication(fault: str) -> None:
    value, origin = chain_prior()
    scope, cutoff, token = value.scope, 12, 'origin'
    if fault == 'scope': scope = ('foreign', *scope[1:])
    elif fault == 'clock': cutoff = 31
    elif fault == 'token': value = replace(value, tokens=(token,))
    else: origin = B.Board()
    before = P.digest(value)
    with pytest.raises(ValueError, match='projected_'):
        P.project(value, scope, origin, origin_frame=10, cutoff_frame=cutoff, origin_token=token)
    assert P.digest(value) == before


def test_repeated_forecast_does_not_consume_origin() -> None:
    value, origin = chain_prior()
    first = P.project(value, value.scope, origin, origin_frame=10, cutoff_frame=12,
        origin_token='basis-cascade:original', origin_trigger_sec=10 / 60)
    again = P.project(value, value.scope, origin, origin_frame=10, cutoff_frame=14,
        origin_token='basis-cascade:original', origin_trigger_sec=10 / 60)
    assert first.outcomes == again.outcomes and first.source_digest == again.source_digest
    assert first.cutoff_frame == 12 and again.cutoff_frame == 14 and value.tokens == ()
    assert again.origin_trigger_sec == 10 / 60


def test_future_trigger_is_not_rounded_into_available_clock() -> None:
    value, origin = chain_prior()
    with pytest.raises(ValueError, match='projected_trigger_clock'):
        P.project(value, value.scope, origin, origin_frame=10, cutoff_frame=12,
            origin_token='origin', origin_trigger_sec=11 / 60)
