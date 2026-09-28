"""終了信号を無視する版と既存E33の独立性を検証する。"""
from __future__ import annotations
import inspect
import pytest
from tests.test_e33_stage_timeout import fixture, STAGE_SEC
from src.exchange_prefire_stage_timeout import STAGE_TIMEOUT_SEC


@pytest.mark.parametrize('reason', ['next', 'slide', 'ojama'])
def test_only_ignores_end_and_still_times_out(reason: str) -> None:
    engine, entry, overlay, result = fixture()
    engine.timeout_only = True
    entry['chain'].end_signal_sec = STAGE_SEC+.1
    entry['chain'].end_reason = reason
    engine.observe(overlay, result, STAGE_SEC+.1)
    assert len(entry['options']) == 2
    engine.observe(overlay, result, STAGE_SEC+STAGE_TIMEOUT_SEC+.1)
    assert len(entry['options']) == 1
    assert entry['audit']['stage_exclusions'][0]['reason'] == 'stage_timeout'


def test_revoked_end_does_not_permanently_withdraw() -> None:
    engine, entry, overlay, result = fixture()
    engine.timeout_only = True
    entry['options'] = entry['options'][1:]
    entry['chain'].end_signal_sec, entry['chain'].end_reason = STAGE_SEC+.1, 'ojama'
    engine.observe(overlay, result, STAGE_SEC+.1)
    assert not entry['audit']['withdrawn']
    entry['chain'].end_signal_sec, entry['chain'].end_reason = None, None
    engine._filter(entry, (2, 360), STAGE_SEC+1.3)
    assert len(entry['options']) == 1 and not entry['audit']['withdrawn']


def test_overlay_wiring_and_mutual_exclusion() -> None:
    from src.exchange_event_overlay import ExchangeEventOverlay
    with pytest.raises(ValueError, match='hidden-row-belief'):
        ExchangeEventOverlay(None, None, None, prefire_stage_timeout_only=True)
    with pytest.raises(ValueError, match='同時指定'):
        ExchangeEventOverlay(None, None, None, prefire_stage_timeout=True, prefire_stage_timeout_only=True)
    overlay = ExchangeEventOverlay(None, None, None, prefire_snapshot=True,
        hidden_row_belief=True, prefire_stage_timeout_only=True)
    assert overlay._prefire.timeout_only


@pytest.mark.parametrize('name', ['overlay', 'replay', 'render'])
def test_new_flag_defaults_off(name: str) -> None:
    from src.exchange_event_overlay import ExchangeEventOverlay
    from scripts.replay_exchange_event_20260926 import replay
    from scripts.visualize_advantage_overlay import generate
    call = dict(overlay=ExchangeEventOverlay, replay=replay, render=generate)[name]
    assert inspect.signature(call).parameters['prefire_stage_timeout_only'].default is False
