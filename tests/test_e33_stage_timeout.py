"""段不在・絶対終了・撤回と同フレームの式優先を検証する。"""
from __future__ import annotations
from types import SimpleNamespace as NS
import pytest
from src.board import Board
from src.chain import ChainSimulator
from src.chain_detector import CHAIN_MECHANISM_FORMULA_READ
from src.exchange_event_tracker import ExchangeChainRecord
from src.exchange_prefire_stage_timeout import StageTimeoutPrefire, STAGE_TIMEOUT_SEC

STAGE_SEC = 10.0


def fixture() -> tuple:
    """同じ1段接頭辞を持つ1段・2段候補を用意する。"""
    engine = StageTimeoutPrefire(ChainSimulator())
    chain = ExchangeChainRecord('1P', 1, STAGE_SEC, STAGE_SEC)
    chain.predicted_final_score = 77.
    options = [dict(prefix=p, score=p[-1], weight=.5, board=Board()._grid.tolist())
               for p in ((40,), (40, 360))]
    row = dict(observations=[], predicted_score=200., withdrawn=False)
    entry = dict(chain=chain, options=options, last=None, audit=row, elapsed=0.,
                 original=(77., None, None), published=None)
    engine.entries[1] = entry
    engine._filter(entry, (1, 40), STAGE_SEC)
    engine._publish(entry)
    overlay = NS(tracker=NS(latest_chain=lambda s: chain if s == '1P' else None),
                 _last_formula=[None, None])
    result = NS(p1=NS(chain_event=None), p2=NS(chain_event=None))
    return engine, entry, overlay, result


@pytest.mark.parametrize('offset,remaining', [(0., 2), (STAGE_TIMEOUT_SEC, 2),
                                            (STAGE_TIMEOUT_SEC+.01, 1)])
def test_timeout_boundary(offset: float, remaining: int) -> None:
    engine, entry, overlay, result = fixture()
    engine.observe(overlay, result, STAGE_SEC+offset)
    assert len(entry['options']) == remaining
    assert entry['audit']['predicted_score'] == 200.
    if remaining == 1:
        assert entry['chain'].predicted_final_score == 40.
        assert entry['options'][0]['weight'] == 1.


@pytest.mark.parametrize('reason', ['next', 'slide', 'ojama'])
def test_absolute_end_precedes_timeout(reason: str) -> None:
    engine, entry, overlay, result = fixture()
    entry['chain'].end_signal_sec = STAGE_SEC+.1
    entry['chain'].end_reason = reason
    engine.observe(overlay, result, STAGE_SEC+.1)
    assert len(entry['options']) == 1
    assert entry['audit']['stage_exclusions'][0]['reason'] == reason


@pytest.mark.parametrize('reason', ['settled', 'score_finalize', None])
def test_nonabsolute_end_does_not_exclude(reason: str | None) -> None:
    engine, entry, overlay, result = fixture()
    entry['chain'].end_signal_sec, entry['chain'].end_reason = STAGE_SEC+.1, reason
    engine.observe(overlay, result, STAGE_SEC+.1)
    assert len(entry['options']) == 2


def test_formula_visible_waits_for_confirmation() -> None:
    engine, entry, overlay, result = fixture()
    stamp = STAGE_SEC+STAGE_TIMEOUT_SEC+.1
    overlay._last_formula[0] = stamp
    engine.observe(overlay, result, stamp)
    assert len(entry['options']) == 2


def test_new_stage_wins_on_expired_frame() -> None:
    engine, entry, overlay, result = fixture()
    stamp = STAGE_SEC+STAGE_TIMEOUT_SEC+.1
    result.p1.chain_event = NS(mechanism=CHAIN_MECHANISM_FORMULA_READ,
        trigger_sec=STAGE_SEC, chain_count=2, total_score=360)
    engine.observe(overlay, result, stamp)
    assert entry['stage_sec'] == stamp
    assert entry['chain'].predicted_final_score == 360.
    assert 'stage_exclusions' not in entry['audit']


def test_repeated_stage_does_not_extend_deadline() -> None:
    engine, entry, _, _ = fixture()
    engine._filter(entry, (1, 40), STAGE_SEC+1.)
    assert entry['stage_sec'] == STAGE_SEC


def test_all_removed_restores_current_fallback_permanently() -> None:
    engine, entry, overlay, result = fixture()
    entry['options'] = entry['options'][1:]
    entry['chain'].predicted_final_score = 99.
    stamp = STAGE_SEC+STAGE_TIMEOUT_SEC+.1
    engine.observe(overlay, result, stamp)
    assert entry['audit']['withdrawn']
    assert entry['chain'].predicted_final_score == 99.
    engine.observe(overlay, result, stamp+1.)
    assert len(entry['audit']['stage_exclusions']) == 1


def test_exclusion_is_not_repeated_and_other_side_does_not_end() -> None:
    engine, entry, overlay, result = fixture()
    overlay._last_formula[1] = STAGE_SEC
    stamp = STAGE_SEC+STAGE_TIMEOUT_SEC+.1
    engine.observe(overlay, result, stamp)
    engine.observe(overlay, result, stamp+1.)
    assert len(entry['audit']['stage_exclusions']) == 1


def test_reset_discards_deadlines() -> None:
    engine, _, _, _ = fixture()
    engine.reset()
    assert not engine.entries


@pytest.mark.parametrize('name', ['overlay', 'replay', 'render'])
def test_flag_defaults_off(name: str) -> None:
    import inspect
    from src.exchange_event_overlay import ExchangeEventOverlay
    from scripts.replay_exchange_event_20260926 import replay
    from scripts.visualize_advantage_overlay import generate
    call = dict(overlay=ExchangeEventOverlay, replay=replay, render=generate)[name]
    assert inspect.signature(call).parameters['prefire_stage_timeout'].default is False


def test_flag_requires_e32() -> None:
    from src.exchange_event_overlay import ExchangeEventOverlay
    with pytest.raises(ValueError, match='hidden-row-belief'):
        ExchangeEventOverlay(None, None, None, prefire_stage_timeout=True)
    engine = ExchangeEventOverlay(None, None, None, prefire_snapshot=True,
        hidden_row_belief=True, prefire_stage_timeout=True)
    assert isinstance(engine._prefire, StageTimeoutPrefire)
