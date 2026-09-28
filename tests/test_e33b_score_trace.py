"""候補下限・撤回・表示保持を採点器で取り違えないことを確認する。"""
from types import SimpleNamespace as NS
from scripts.e33b_score_trace import effective_scores, ScoreTrace, value_key


def test_observed_floor_is_applied_before_averaging() -> None:
    chain = NS(side='1P', chain_id=1, formula_total=300, score_delta=200, provisional_score=999)
    engine = NS(active=lambda c: dict(options=[dict(weight=.5, score=100), dict(weight=.5, score=500)]))
    assert effective_scores(engine, [chain])[0]['score'] == 400
    engine.active = lambda c: None
    assert effective_scores(engine, [chain])[0]['score'] == 999


def test_display_hold_uses_evaluation_score_instead_of_current_chain() -> None:
    trace = ScoreTrace.__new__(ScoreTrace)
    value = dict(source='S3_landing', p1=.7, t_sec=10)
    record = NS(game_idx=1, exchange_id=2, values=[value])
    saved = [dict(side='1P', chain_id=1, score=500)]
    trace.values = {value_key(record, value): saved}
    overlay = NS(tracker=NS(current=record),
        _landing_projection=NS(death=None, last=value, death_record=None))
    row = NS(source='S3_landing', display_p1=.7, t_sec=12)
    assert trace.selected_value(overlay, row) == (saved, 10)
    row.display_p1 = .8
    assert trace.selected_value(overlay, row) == ([], None)
