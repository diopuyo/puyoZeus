"""D3: 途中確定の訂正と前連鎖の式残響の帰属制限。"""
from src.exchange_event_tracker import ExchangeEventTracker, ExchangeChainRecord, ExchangeRecord


def tracker(enabled: bool = True) -> tuple[ExchangeEventTracker, ExchangeChainRecord]:
    """第41試合の早期確定状態を構成する。"""
    item = ExchangeChainRecord('1P', 6, 6038.583333, 6038.583333,
        score_before=439., score_delta=880., score_ready_sec=6041.883333,
        score_ready_reason='score_finalize', score_finalize_sec=6041.883333,
        end_signal_sec=6051.15, end_confirmed=True)
    result = ExchangeEventTracker(None, score_attribution=enabled)
    result.current = ExchangeRecord(1, 1, item.trigger_sec, chains=[item])
    return result, item


def test_early_finalize_updated() -> None:
    result, item = tracker()
    result.observe_score('1P', 6059., 40139., 439., 39700.)
    assert item.score_delta == item.provisional_score == 39700.
    assert item.score_ready_reason == 'formula_display_match'


def test_default_preserves_legacy() -> None:
    result, item = tracker(False)
    result.observe_score('1P', 6059., 40139., 439., 39700.)
    assert item.provisional_score == 880.


def test_previous_formula_not_attributed() -> None:
    result, item = tracker()
    item.score_before, item.score_delta = 44444., None
    item.score_ready_sec, item.end_signal_sec = None, None
    result.observe_score('1P', 4620.7, 44444., 44444., 42800.)
    assert item.formula_total == 0.
    assert item.score_delta is None


def test_partial_numeric_increase_caps_formula() -> None:
    result, item = tracker()
    result.observe_score('1P', 6042., 2000., 439., 39700.)
    assert item.formula_total == 1561.
    assert item.score_delta == 880.


def test_ledger_does_not_resurrect_cancelled_attack() -> None:
    """567→12への巻戻しによる相殺済み予告の復活を防ぐ。"""
    from src.exchange_pending_ledger import PendingLedger
    from src.exchange_landing_safety import LandingStateSafety
    from src.scoring import score_to_ojama
    result, item = tracker()
    ledger = PendingLedger()
    ledger.observe(1, {(1, 8): (289, True)}, (0, 0))
    result.observe_score('1P', 6059., 40139., 439., 39700.)
    amount = int(score_to_ojama(item.provisional_score).ojama_count)
    ledger.observe(1, {(1, 8): (289, True), (0, 6): (amount, LandingStateSafety._verified(item))}, (0, 0))
    assert amount == 567
    assert ledger.pending[0] == 0


def test_prediction_is_not_numeric_evidence() -> None:
    from src.exchange_landing_safety import LandingStateSafety
    result, item = tracker()
    item.end_signal_sec, item.score_ready_sec = None, None
    item.score_delta, item.predicted_final_score = None, 42800.
    item.score_before = 44444.
    result.observe_score('1P', 4620.7, 44444., 44444., 42800.)
    assert not LandingStateSafety._verified(item)


def test_capped_formula_is_not_full_formula_match() -> None:
    result, item = tracker()
    item.score_ready_sec, item.score_delta = None, None
    result.observe_score('1P', 6059., 2000., 439., 39700.)
    assert item.formula_total == 1561.
    assert item.score_ready_sec is None
