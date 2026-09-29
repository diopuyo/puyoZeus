"""E25包括条件を分離してもOFF経路と既存包括条件を維持する。"""
from __future__ import annotations

from types import SimpleNamespace
import pytest

from src.exchange_event_landing import ExchangeLandingProjection
from src.exchange_completion_recovery import CompletionRecovery
from src.board import Board


@pytest.mark.parametrize('flag', ['pending_ledger', 'color_score_safety', 'completion_recovery'])
def test_single_component_does_not_enable_other_components(flag: str) -> None:
    safety = ExchangeLandingProjection(**{flag: True}).safety
    assert safety.ledger_enabled == (flag == 'pending_ledger')
    assert safety.guard_enabled == (flag == 'color_score_safety')
    assert safety.recovery_enabled == (flag == 'completion_recovery')


def test_default_keeps_legacy_path() -> None:
    assert ExchangeLandingProjection().safety is None


def test_legacy_bundle_keeps_all_components() -> None:
    safety = ExchangeLandingProjection(landing_state_safety=True).safety
    assert safety.ledger_enabled and safety.guard_enabled and safety.recovery_enabled


@pytest.mark.parametrize('flag', ['pending_ledger', 'completion_recovery'])
def test_without_guard_never_blocks(flag: str) -> None:
    safety = ExchangeLandingProjection(**{flag: True}).safety
    assert safety.blocker(None, None, 0, 1.) is None


def test_color_guard_cannot_recover_missing_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('src.exchange_completion_recovery.enumerate_completions',
        lambda *args: pytest.fail('整合確認だけで起点補完を起動した'))
    recovery, board = CompletionRecovery(), Board()
    chain = SimpleNamespace(trigger_sec=1., predicted_final_board=None, predicted_chain_count=0)
    recovery.seed(chain, SimpleNamespace(before_board=board), [],
                  ExchangeLandingProjection().simulator, allow_recovery=False)
    assert not recovery.entries
