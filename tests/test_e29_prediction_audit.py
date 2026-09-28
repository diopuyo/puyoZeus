"""隠し段上限の採用後不一致を、撤回・試合境界で見失わない。"""
from types import SimpleNamespace as NS
import pytest
from scripts.audit_e29_predictions import PredictionTrace, counts


@pytest.mark.parametrize('score,ready,expected,unresolved', [
    (360., True, 0, 0), (400., True, 1, 0), (320., True, 1, 0),
    (None, True, 0, 1), (360., False, 0, 1)])
def test_final_score_and_unresolved_are_separate(score: float | None, ready: bool,
                                                expected: int, unresolved: int) -> None:
    trace = PredictionTrace('review')
    row = dict(game=1, side='1P', chain_id=1, outcome='accepted', maximum_score=360., revoked_sec=2.)
    trace.hidden_engine = NS(summary=lambda: dict(rows=[row], accepted=1, next_mismatch=0))
    trace.hidden_chains[(1, '1P', 1)] = NS(score_ready_sec=2., end_signal_sec=2.,
        end_confirmed=ready, score_delta=score)
    result = counts(trace.hidden_summary())
    assert result['accepted'] == 1 and result['accepted_chains'] == 1
    assert result['final_mismatch'] == expected and result['final_unresolved'] == unresolved


def test_next_mismatch_is_not_an_accepted_prediction() -> None:
    trace = PredictionTrace('review')
    trace.hidden_engine = NS(summary=lambda: dict(rows=[dict(outcome='next_mismatch')],
        accepted=0, next_mismatch=1))
    result = counts(trace.hidden_summary())
    assert result == dict(accepted=0, accepted_chains=0, next_mismatch=1,
        final_mismatch=0, final_unresolved=0)
