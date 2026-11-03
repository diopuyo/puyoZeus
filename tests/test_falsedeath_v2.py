"""D-v2の独立証拠・候補検査・未知ツモと既定OFF互換性を検証する。"""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from src.board import Board
from src.exchange_event_tracker import ExchangeEventTracker, ExchangeChainRecord, ExchangeRecord
from src.exchange_landing_safety import LandingStateSafety
from src.exchange_post_counter_bound import PostCounterDeathBound

# diag_remaining_2026-10-02/input_evidence.json のall死亡用2P盤面をそのまま固定。
CASES = (
    ('c5', '000000/000000/000000/000000/000000/000000/000000/000000/000000/000000/000000/000000/053000',
     (3, 2, 4, 3), 77),
    ('mia9', '000000/040000/040000/420000/250000/530350/444320/335320/535450/522430/235224/443552/335244',
     (2, 2, 2, 2), 48),
)
NEW_OPTIONS = ('score_attribution_v2', 'post_counter_input_integrity_v2',
               'post_counter_current_puyo_integrity', 'post_counter_current_puyo_integrity_v2c')


def score_tracker(v2: bool = True) -> tuple[ExchangeEventTracker, ExchangeChainRecord]:
    """式欠測・数値差だけで確定した連鎖を構成する。"""
    chain = ExchangeChainRecord('1P', 1, 1., 1., score_before=100., score_delta=1000.,
        score_ready_sec=2., score_ready_reason='display_stable', end_signal_sec=2., end_confirmed=True)
    tracker = ExchangeEventTracker(None, score_attribution=True, score_attribution_v2=v2)
    tracker.current = ExchangeRecord(1, 0, 1., chains=[chain])
    tracker.records = [tracker.current]
    tracker._game_idx = 0
    tracker.observe_score('1P', 3., 1100., formula_total=None)
    return tracker, chain


def test_display_difference_is_not_independent_evidence() -> None:
    _, chain = score_tracker()
    assert chain.score_attribution and chain.independent_score_required
    assert chain.formula_total is None
    assert not LandingStateSafety._verified(chain)


def test_old_d3_keeps_existing_behavior() -> None:
    _, chain = score_tracker(False)
    assert LandingStateSafety._verified(chain)


@pytest.mark.parametrize('proof', ['formula', 'finalize'])
def test_independent_evidence_verifies(proof: str) -> None:
    tracker, chain = score_tracker()
    if proof == 'formula':
        tracker.observe_score('1P', 3., 1100., formula_total=1000.)
    else:
        tracker.finalize('1P', 3., 1000.)
    assert LandingStateSafety._verified(chain)


def test_insufficient_finalize_does_not_verify_larger_display() -> None:
    tracker, chain = score_tracker()
    tracker.finalize('1P', 3., 40.)
    assert not LandingStateSafety._verified(chain)


def test_missing_formula_zero_is_not_evidence() -> None:
    from src.exchange_death_integrity_v2 import independently_verified
    chain = NS(display_score=100.,score_before=100.,formula_total=None,provisional_score=0.)
    assert not independently_verified(chain)


@pytest.mark.parametrize('withdrawn', [True,False])
def test_candidate_withdrawal_or_change_blocks(monkeypatch: pytest.MonkeyPatch, withdrawn: bool) -> None:
    from src import exchange_death_inputs
    from src.exchange_death_integrity_v2 import completion_blocker
    selected = dict(score=40,options=[dict(board=[[0]])])
    fresh = dict(score=80,options=[dict(board=[[0]])],audit=dict(withdrawn=withdrawn))
    monkeypatch.setattr(exchange_death_inputs,'maximum_response',lambda *args: fresh)
    tracker = NS(latest_chain=lambda side: NS())
    expected = 'unverified_post_counter_candidates' if withdrawn else 'stale_post_counter_candidates'
    assert completion_blocker(NS(),tracker,0,dict(hidden=[selected,None])) == expected


@pytest.mark.parametrize('time_upper,agree_lower,expected', [(.019,-.029,True),(.020,-.030,False)])
def test_v2_gate_strict_ci_boundaries(time_upper: float, agree_lower: float, expected: bool) -> None:
    from scripts.falsedeath_v2_score_20261002 import metric_gates
    value = dict(metrics=dict(m_time=dict(paired=dict(diff=.002,ci95=[0.,time_upper])),
                              m_agree=dict(paired=dict(diff=-.005,ci95=[agree_lower,0.]))))
    assert metric_gates(value) == dict(m_time=expected,m_agree=expected)


@pytest.mark.parametrize('name,encoded,queue,incoming', CASES)
def test_unknown_current_pair_prevents_false_bound(name: str, encoded: str, queue: tuple, incoming: int) -> None:
    board = Board.from_list([[int(c) for c in row] for row in encoded.split('/')])
    chain = NS(predicted_final_score=0., predicted_final_board=None, chain_id=1)
    overlay = NS(tracker=NS(latest_chain=lambda side: chain, _score_elapsed=0.), _game=0)
    projection = NS(_known_budget=lambda *args: True, _chaining=lambda *args: False)
    context = dict(hidden=[None,None], certain=[True,True], credit=[0,0],
                   verified=[True,True], replies=[board,board])
    results = []
    for enabled in (False,True):
        engine = PostCounterDeathBound(current_puyo_integrity=enabled)
        engine.queues = (queue,queue)
        engine.colors = [NS(active=lambda: (2,3,4,5))]*2
        results.append(engine.prove(projection,overlay,1,(),incoming,1,context,1.))
    assert results[0]['dead'], name
    assert not results[1]['dead'], name


@pytest.mark.parametrize('reason', ['board_unsettled','next_transition','color_score_mismatch'])
def test_v2_keeps_input_blockers(reason: str) -> None:
    engine = PostCounterDeathBound(input_integrity_v2=True)
    engine.next_motion = (reason == 'next_transition',False)
    projection = NS(safety=NS(blocker=lambda *args: reason))
    assert engine.input_blocker(projection,NS(),0,1.) == reason


@pytest.mark.parametrize('available', [True,False])
def test_v2_checks_selected_candidates(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    from src import exchange_death_inputs
    bound = dict(score=40,options=[dict(board=Board()._grid.tolist())],audit=dict(accepted=True))
    monkeypatch.setattr(exchange_death_inputs,'maximum_response',lambda *args: bound if available else None)
    chain = NS(end_signal_sec=1.,post_end_drop_sec=1.1,end_confirmed=True)
    tracker = NS(latest_chain=lambda side: chain)
    projection = NS(safety=NS(blocker=lambda *args: 'completion_prefix_unverified'))
    context = dict(hidden=[bound,None])
    engine = PostCounterDeathBound(input_integrity_v2=True)
    result = engine.input_blocker(projection,tracker,0,2.,context)
    assert result == (None if available else 'unverified_post_counter_candidates')
    old = PostCounterDeathBound(input_integrity=True)
    assert old.input_blocker(projection,tracker,0,2.,context) == 'completion_prefix_unverified'


def test_v2_keeps_observed_budget_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    chain = NS(end_signal_sec=1.,post_end_drop_sec=None,end_confirmed=True)
    projection = NS(safety=NS(blocker=lambda *args: None))
    engine = PostCounterDeathBound(input_integrity_v2=True)
    assert engine.input_blocker(projection,NS(latest_chain=lambda side: chain),0,2.) == 'unverified_response_upper_bound'


def test_new_flags_default_off() -> None:
    from src.exchange_event_overlay import ExchangeEventOverlay
    from scripts.replay_exchange_event_20260926 import replay
    from scripts.visualize_advantage_overlay import generate
    for function in (ExchangeEventOverlay,replay,generate):
        for name in NEW_OPTIONS:
            assert inspect.signature(function).parameters[name].default is False


@pytest.mark.parametrize('source', ['review', 'q_7gc4TgFig', 'fcXG83vInDY', 'mia8KCjr52g', 'zenchi'])
def test_production_off_is_byte_identical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                         source: str) -> None:
    """既存本番review原票そのものと全表示・イベントバイトを比較する。"""
    import sys
    from scripts.falsedeath_replay_20261001 import input_record, OUT, enabled_flags
    from src import production_config
    from scripts.replay_exchange_event_20260926 import main, compare
    record = input_record('prod',source)
    expected = OUT/'replay/prod'/source
    if not record.exists() or not expected.exists():
        pytest.skip('本番原票がある評価環境で実行する統合テスト')
    # 採用後もこのテストは採用前の既定OFF互換性を検査する。
    excluded = {*enabled_flags('v2h2b'), '--a2-safety-review', '--completion-score-max-chain', '--exchange-event-static-gfe-dir'}
    legacy = ' '.join(e.flag for e in production_config.EXCHANGE_EVENT_ADOPTED
                      if e.flag.split()[0] not in excluded)
    monkeypatch.setattr(production_config, 'exchange_event_flags', lambda: legacy)
    monkeypatch.setattr(sys,'argv',['replay',str(record),'--out',str(tmp_path),'--production-exchange-event'])
    main()
    result = compare(expected,tmp_path)
    assert result['display'] == result['events'] == 'byte_identical'
