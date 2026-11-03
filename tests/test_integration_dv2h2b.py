"""D-v2h2b本番登録から実描画経路への配線と監査保持を検証する。"""
from __future__ import annotations

import inspect

import pytest

from scripts.falsedeath_replay_20261001 import enabled_flags
from src import production_config
from src.exchange_history_integrity import MatchHistoryIntegrity
from src.exchange_post_counter_bound import PostCounterDeathBound
from tests.test_exchange_event_static_gfe_cli import start_render
from tests.test_falsedeath_v2d import sides


def test_production_v2h2b_reaches_render_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    """フラグをテストで補わず、本番CLIだけで全8項目が実体へ届く。"""
    monkeypatch.setattr('src.exchange_event_terminal.ObservedDeathDetector', lambda: None)
    overlay = start_render(monkeypatch, ['--production-exchange-event'])
    guard = overlay._history_integrity
    projection = overlay._landing_projection
    engine = projection.post_counter_bound
    actual = {
        '--death-input-integrity': projection.death_input_integrity,
        '--score-attribution-v2': overlay.tracker.score_attribution_v2,
        '--post-counter-current-puyo-integrity-v2c': engine.current_puyo_integrity_v2c,
        '--death-history-integrity': guard is not None,
        '--death-history-boundary-start': guard.boundary_score_start,
        '--post-counter-completion-recent-integrity': engine.completion_recent_integrity,
        '--death-integrity-v2h2': guard.strict_integrity and engine.strict_integrity,
        '--death-freshness-cap': engine.cap_completed,
    }
    assert set(actual) == set(enabled_flags('v2h2b'))
    assert all(actual.values())
    assert not guard.retain_audit and not engine.retain_audit
    registered = {entry.flag: entry for entry in production_config.EXCHANGE_EVENT_ADOPTED}
    for flag in set(enabled_flags('v2h2b')) - set(enabled_flags('v2d')):
        assert registered[flag].adopted == '2026-10-03'
        assert registered[flag].reason == production_config.DV2H2B_ADOPTION_REASON


@pytest.mark.parametrize('retain', [False, True])
def test_boundary_evidence_survives_audit_clear(retain: bool) -> None:
    """前試合の得点を先に読むため、監査辞書の破棄後もH1′が受理できる。"""
    guard = MatchHistoryIntegrity(retain_audit=retain, boundary_score_start=True,
                                  strict_integrity=True)
    previous = sides(2, 2)
    previous[0].score, previous[1].score = 100, 200
    guard.observe(previous, 0, 0.)
    guard.audit.append({'game': 0})
    current = sides(0, 0)
    current[0].score = current[1].score = 0
    guard.observe(current, 1, 1.)
    evidence = guard.evidence()
    assert evidence['history_complete']
    assert evidence['boundary_score_evidence']['accepted']
    assert set(guard.games) == ({0, 1} if retain else {1})
    assert len(guard.audit) == int(retain)


@pytest.mark.parametrize('cls', [MatchHistoryIntegrity, PostCounterDeathBound])
def test_new_constructor_arguments_are_keyword_only(cls: type) -> None:
    """rtのretain_auditとH1′・E35追加フラグを位置で混同させない。"""
    for name, parameter in inspect.signature(cls).parameters.items():
        if name != 'early_exit':
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
