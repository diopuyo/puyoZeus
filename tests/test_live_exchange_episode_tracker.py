"""リアルタイム交換episodeアダプタの回帰テスト。"""
from __future__ import annotations

import pytest

from src.chain_detector import CHAIN_MECHANISM_BASELINE, CHAIN_MECHANISM_FORMULA
from src.exchange_episode_tracker import (
    ChainEventObservation,
    GenerationObservation,
    SettlementObservation,
)
from src.exchange_ledger import EpisodeStatus, PhysicalContext, Side
from src.exchange_ledger import (
    FINALIZE_SOURCE_SCORE_OCR_DIFF, EventKind, ExchangeEvent, ExchangeLedger,
)
from src.live_exchange_episode_tracker import LiveExchangeEpisodeTracker


def _chain(
    t_sec: float, score: int, *, side: str = "1P", mechanism: str = CHAIN_MECHANISM_FORMULA,
) -> ChainEventObservation:
    return ChainEventObservation(
        side=side, t_sec=t_sec, mechanism=mechanism, chain_count=1,
        total_score=score, ojama_sent=0, game_idx=0, elapsed_sec=t_sec)


def _generation(t_sec: float, amount: int, side: str = "1P") -> GenerationObservation:
    return GenerationObservation(
        side=side, t_sec=t_sec, game_idx=0, generated_delta=amount)


def _ctx(**kwargs: object) -> PhysicalContext:
    return PhysicalContext(**kwargs)


def test_live_formula_opens_and_finalize_replaces_without_double_count() -> None:
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    first = tracker.observe_frame(
        t_sec=1.0, context=_ctx(p1_chaining=True),
        chain_observations=(_chain(1.0, 700),))
    assert first.ledger.net_raw == pytest.approx(10.0)
    assert first.ledger.is_unresolved is True
    assert first.ledger.allows_hard_override is False
    assert first.latest_chain_id_p1 == 1

    second = tracker.observe_frame(
        t_sec=2.0, context=_ctx(p1_chaining=False),
        chain_observations=(_chain(2.0, 1400),),
        generation_observations=(_generation(2.0, 18),))
    assert second.ledger.net_raw == pytest.approx(18.0)
    assert second.ledger.total_generated == pytest.approx(18.0)
    assert second.resolved_chain_count == 1


def test_live_formula_ignores_impossible_cumulative_score_decrease() -> None:
    """同一物理連鎖の累積点OCRが下振れしても台帳を減算しない。"""
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    first = tracker.observe_frame(
        t_sec=1.0, context=_ctx(p1_chaining=True),
        chain_observations=(_chain(1.0, 70),))

    second = tracker.observe_frame(
        t_sec=1.1, context=_ctx(p1_chaining=True),
        chain_observations=(_chain(1.1, 0),))

    assert first.ledger.net_raw == pytest.approx(1.0)
    assert second.ledger.net_raw == pytest.approx(1.0)
    assert second.active_chains[0].provisional_score == 70
    assert second.formula_step_observation_count == 2
    assert second.provisional_score_decrease_ignored_count == 1


def test_live_settlement_closes_episode_with_conservation() -> None:
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    tracker.observe_frame(
        t_sec=1.0, context=_ctx(p1_chaining=True),
        chain_observations=(_chain(1.0, 700),))
    tracker.observe_frame(
        t_sec=2.0, context=_ctx(),
        generation_observations=(_generation(2.0, 10),))
    settled = tracker.observe_frame(
        t_sec=3.0, context=_ctx(),
        settlement=SettlementObservation(
            t_sec=3.0, game_idx=0, canceled_by_1p=0.0,
            canceled_by_2p=0.0, landed_on_1p=0.0, landed_on_2p=10.0))
    assert settled.ledger.is_unresolved is False
    closed = tracker.closed_episodes_live()
    assert len(closed) == 1
    assert closed[0].status is EpisodeStatus.CLOSED
    assert closed[0].total_generated == pytest.approx(closed[0].total_landed)


def test_live_match_boundary_forces_close_and_clears_active() -> None:
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    tracker.observe_frame(
        t_sec=1.0, context=_ctx(p1_chaining=True),
        chain_observations=(_chain(1.0, 700),))
    boundary = tracker.observe_frame(t_sec=2.0, context=_ctx(game_idx=1))
    assert boundary.active_chains == ()
    assert boundary.ledger.is_unresolved is False
    assert boundary.closed_episode_count == 1
    assert boundary.last_close_reason == "match_boundary"


def test_match_boundary_does_not_reapply_disappeared_pending_as_settlement() -> None:
    """境界で未照合計上した残量を、同frameの決済として二重処理しない。"""
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    tracker.observe_frame(
        t_sec=1.0, context=_ctx(game_idx=0, p1_chaining=True),
        chain_observations=(_chain(1.0, 700),))

    boundary = tracker.observe_frame(
        t_sec=2.0, context=_ctx(game_idx=1),
        settlement=SettlementObservation(
            t_sec=2.0, game_idx=1, canceled_by_1p=0.0,
            canceled_by_2p=10.0, landed_on_1p=0.0, landed_on_2p=0.0))

    assert boundary.unattributed_settlement_total == 0.0
    assert boundary.boundary_settlement_excluded_count == 1
    assert boundary.boundary_settlement_excluded_amount == pytest.approx(10.0)


def test_simulate_only_chain_never_enters_live_ledger() -> None:
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    pending = tracker.observe_frame(
        t_sec=1.0, context=_ctx(),
        chain_observations=(_chain(
            1.0, 52150, mechanism=CHAIN_MECHANISM_BASELINE),))
    assert pending.ledger.net_raw == 0.0
    # 生成量がまだ台帳へ入っていなくても、物理連鎖が生きている間は
    # 「交換なし」と断定して ±100 を許してはならない。
    assert pending.active_chains
    assert pending.ledger.is_unresolved is True
    assert pending.ledger.allows_hard_override is False
    closed = tracker.observe_frame(t_sec=2.0, context=_ctx(game_idx=1))
    assert closed.ledger.total_generated == 0.0
    assert closed.closed_episode_count == 0
    assert closed.simulate_excluded_chain_count == 1
    assert closed.simulate_excluded_amount > 0.0


def test_active_defender_survives_state_flicker_for_hard_override_gate() -> None:
    """state が一瞬 STABLE でも resolver 上の反撃連鎖を終了扱いしない。"""
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    tracker.observe_frame(
        t_sec=1.0, context=_ctx(p1_chaining=True, p2_chaining=True),
        chain_observations=(
            _chain(1.0, 700, side="1P"),
            _chain(1.0, 1400, side="2P"),
        ))

    flicker = tracker.observe_frame(
        t_sec=2.0,
        context=_ctx(p1_chaining=False, p2_chaining=True, p1_room=1),
        generation_observations=(_generation(2.0, 20, side="2P"),))

    assert any(item.side == "1P" for item in flicker.active_chains)
    assert flicker.ledger.net_raw < 0.0
    assert flicker.ledger.is_unresolved is True
    assert flicker.ledger.hard_override_target is None
    assert flicker.ledger.allows_hard_override is False


def test_active_only_snapshot_still_allows_confirmed_physical_death() -> None:
    """生成量未到着でも、片側だけの死亡確定は90%方向へ通してよい。"""
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    dead = tracker.observe_frame(
        t_sec=1.0, context=_ctx(p2_dead=True),
        chain_observations=(_chain(
            1.0, 52150, mechanism=CHAIN_MECHANISM_BASELINE),))

    assert dead.ledger.net_raw == 0.0
    assert dead.ledger.is_unresolved is True
    assert dead.ledger.allows_hard_override is True
    assert dead.ledger.hard_override_target == 100.0


def test_physical_death_releases_hard_override_gate() -> None:
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    tracker.observe_frame(
        t_sec=1.0, context=_ctx(p1_chaining=True),
        chain_observations=(_chain(1.0, 700),))
    dead = tracker.observe_frame(t_sec=1.1, context=_ctx(p2_dead=True))
    assert dead.ledger.is_unresolved is True
    assert dead.ledger.allows_hard_override is True
    assert dead.ledger.hard_override_target == 100.0


def test_wipe_is_applied_before_settlement() -> None:
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    tracker.observe_frame(
        t_sec=1.0, context=_ctx(p1_chaining=True),
        chain_observations=(_chain(1.0, 700),))
    tracker.observe_frame(
        t_sec=2.0, context=_ctx(),
        generation_observations=(_generation(2.0, 10),))
    wiped = tracker.observe_frame(
        t_sec=3.0, context=_ctx(), wiped_sides=(Side.P2,))
    assert wiped.ledger.retired_chain_count == 1
    assert wiped.ledger.is_unresolved is False


def test_last_closed_summary_is_exposed_for_sidecar_verification() -> None:
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    tracker.observe_frame(
        t_sec=1.0, context=_ctx(p1_chaining=True),
        chain_observations=(_chain(1.0, 700),))
    tracker.observe_frame(
        t_sec=2.0, context=_ctx(),
        generation_observations=(_generation(2.0, 10),))
    snap = tracker.observe_frame(
        t_sec=3.0, context=_ctx(),
        settlement=SettlementObservation(
            t_sec=3.0, game_idx=0, canceled_by_1p=0.0,
            canceled_by_2p=0.0, landed_on_1p=0.0, landed_on_2p=10.0))
    assert snap.last_closed_status == "CLOSED"
    assert snap.last_closed_generated == pytest.approx(10.0)
    assert snap.last_closed_landed == pytest.approx(10.0)
    assert snap.last_closed_unreconciled == 0.0
    assert snap.last_closed_has_settlement is True
    assert snap.ledger_residual_all == 0.0


def test_finish_does_not_replay_incremental_events() -> None:
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    tracker.observe_frame(
        t_sec=1.0, context=_ctx(p1_chaining=True),
        chain_observations=(_chain(1.0, 700),))
    before = tracker.observe_frame(
        t_sec=2.0, context=_ctx(),
        chain_observations=(_chain(2.0, 1400),),
        generation_observations=(_generation(2.0, 18),))
    tracker.finish()
    after = tracker.finish_live()
    assert after.ledger.total_generated == pytest.approx(
        before.ledger.total_generated)
    assert after.resolved_chain_count == before.resolved_chain_count


def test_same_frame_settlement_wins_over_max_sec_timeout() -> None:
    """60秒超の最初の観測が決済なら、max_secで先に閉じてはならない。"""
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    tracker.observe_frame(
        t_sec=0.0, context=_ctx(p1_chaining=True),
        chain_observations=(_chain(0.0, 700),))
    tracker.observe_frame(
        t_sec=1.0, context=_ctx(),
        generation_observations=(_generation(1.0, 10),))
    snap = tracker.observe_frame(
        t_sec=61.0, context=_ctx(),
        settlement=SettlementObservation(
            t_sec=61.0, game_idx=0, canceled_by_1p=0.0,
            canceled_by_2p=10.0, landed_on_1p=0.0, landed_on_2p=0.0))
    closed = tracker.closed_episodes_live()
    assert closed[-1].status is EpisodeStatus.CLOSED
    assert closed[-1].close_reason == "normal_close"
    assert snap.ledger.post_close_settlement_dropped_count == 0


def test_late_settlement_backfills_forced_summary_without_reopening() -> None:
    """max_sec後の決済は旧要約へ回収し、同じchainでepisodeを再開しない。"""
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    tracker.observe_frame(
        t_sec=0.0, context=_ctx(p1_chaining=True),
        chain_observations=(_chain(0.0, 700),))
    tracker.observe_frame(
        t_sec=1.0, context=_ctx(),
        generation_observations=(_generation(1.0, 10),))
    tracker.observe_frame(
        t_sec=61.0, context=_ctx(),
        settlement=SettlementObservation(
            t_sec=61.0, game_idx=0, canceled_by_1p=0.0,
            canceled_by_2p=3.0, landed_on_1p=0.0, landed_on_2p=0.0))
    snap = tracker.observe_frame(
        t_sec=62.0, context=_ctx(),
        settlement=SettlementObservation(
            t_sec=62.0, game_idx=0, canceled_by_1p=0.0,
            canceled_by_2p=7.0, landed_on_1p=0.0, landed_on_2p=0.0))
    closed = tracker.closed_episodes_live()
    assert len(closed) == 1
    assert closed[0].total_canceled == pytest.approx(10.0)
    assert closed[0].unreconciled == 0.0
    assert snap.ledger.is_unresolved is False
    assert snap.ledger.duplicate_generated_suppressed_count == 0
    assert snap.ledger.post_close_settlement_dropped_count == 0
    assert snap.ledger.post_close_settlement_backfilled_count == 1
    assert snap.ledger.unreconciled == 0.0


def test_late_finalize_and_settlement_keep_global_and_episode_in_sync() -> None:
    """max_sec後の確定量増加も、強制要約とglobal未照合量へ反映する。"""
    ledger = ExchangeLedger(defer_max_sec_close=True)
    ctx = PhysicalContext(p1_chaining=True)
    ledger.push(ExchangeEvent(
        kind=EventKind.FIRE, side=Side.P1, t_sec=0.0,
        amount=10.0, chain_id=1, source="test"), ctx)
    ledger.push(ExchangeEvent(
        kind=EventKind.FINALIZE, side=Side.P1, t_sec=1.0,
        amount=10.0, chain_id=1, source=FINALIZE_SOURCE_SCORE_OCR_DIFF), ctx)
    ledger.push(ExchangeEvent(
        kind=EventKind.FIRE, side=Side.P1, t_sec=59.0,
        amount=10.0, chain_id=2, source="test"), ctx)
    ledger.close_expired(61.0)
    ledger.push(ExchangeEvent(
        kind=EventKind.FINALIZE, side=Side.P1, t_sec=62.0,
        amount=20.0, chain_id=2, source=FINALIZE_SOURCE_SCORE_OCR_DIFF), ctx)
    ledger.push(ExchangeEvent(
        kind=EventKind.CANCEL, side=Side.P1, t_sec=63.0,
        amount=5.0, chain_id=2, source="test"), ctx)

    closed = ledger.closed_episodes()[0]
    snap = ledger.snapshot(ctx)
    assert closed.total_generated == pytest.approx(30.0)
    assert closed.total_canceled == pytest.approx(5.0)
    assert closed.unreconciled == pytest.approx(25.0)
    assert snap.unreconciled == pytest.approx(25.0)
    assert snap.total_generated - snap.total_canceled - snap.total_landed == pytest.approx(
        ledger.total_outstanding_all_chains())
    assert snap.post_close_finalize_backfilled_count == 1
    assert snap.post_close_settlement_backfilled_count == 1


def test_normal_close_accepts_late_equal_finalize_without_false_drop() -> None:
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    tracker.observe_frame(
        t_sec=0.0, context=_ctx(p1_chaining=True),
        chain_observations=(_chain(0.0, 700),))
    tracker.observe_frame(
        t_sec=1.0, context=_ctx(),
        settlement=SettlementObservation(
            t_sec=1.0, game_idx=0, canceled_by_1p=0.0,
            canceled_by_2p=10.0, landed_on_1p=0.0, landed_on_2p=0.0))
    snap = tracker.observe_frame(
        t_sec=2.0, context=_ctx(),
        generation_observations=(_generation(2.0, 10),))
    closed = tracker.closed_episodes_live()
    assert len(closed) == 1
    assert closed[0].unreconciled == 0.0
    assert snap.ledger.is_unresolved is False
    assert snap.ledger.post_close_finalize_backfilled_count == 1
    assert snap.ledger.post_close_finalize_dropped_count == 0
    assert snap.ledger.post_close_settlement_dropped_count == 0


def test_normal_close_late_larger_finalize_reclassifies_without_second_episode() -> None:
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    tracker.observe_frame(
        t_sec=0.0, context=_ctx(p1_chaining=True),
        chain_observations=(_chain(0.0, 700),))
    tracker.observe_frame(
        t_sec=1.0, context=_ctx(),
        settlement=SettlementObservation(
            t_sec=1.0, game_idx=0, canceled_by_1p=0.0,
            canceled_by_2p=10.0, landed_on_1p=0.0, landed_on_2p=0.0))
    snap = tracker.observe_frame(
        t_sec=2.0, context=_ctx(p2_room=72),
        generation_observations=(_generation(2.0, 20),))
    closed = tracker.closed_episodes_live()
    assert len(closed) == 1
    assert closed[0].unreconciled == pytest.approx(10.0)
    assert closed[0].status is EpisodeStatus.CLOSED_FORCED
    assert closed[0].close_reason == "late_finalize_after_normal_close"
    assert snap.ledger.unreconciled == pytest.approx(10.0)
    assert snap.ledger.is_unresolved is True
    assert snap.ledger.allows_hard_override is False
    assert snap.closed_normal_unreconciled_count == 0
    assert snap.ledger.post_close_finalize_dropped_count == 0


def test_reclassified_late_finalize_settles_without_second_episode() -> None:
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    tracker.observe_frame(
        t_sec=0.0, context=_ctx(p1_chaining=True),
        chain_observations=(_chain(0.0, 700),))
    tracker.observe_frame(
        t_sec=1.0, context=_ctx(),
        settlement=SettlementObservation(
            t_sec=1.0, game_idx=0, canceled_by_1p=0.0,
            canceled_by_2p=10.0, landed_on_1p=0.0, landed_on_2p=0.0))
    tracker.observe_frame(
        t_sec=2.0, context=_ctx(p2_room=72),
        generation_observations=(_generation(2.0, 20),))
    snap = tracker.observe_frame(
        t_sec=3.0, context=_ctx(p2_room=72),
        settlement=SettlementObservation(
            t_sec=3.0, game_idx=0, canceled_by_1p=0.0,
            canceled_by_2p=0.0, landed_on_1p=0.0, landed_on_2p=10.0))
    closed = tracker.closed_episodes_live()
    assert len(closed) == 1
    assert closed[0].status is EpisodeStatus.CLOSED_FORCED
    assert closed[0].unreconciled == 0.0
    assert snap.ledger.is_unresolved is False


def test_finalize_only_backfill_does_not_fake_settlement_input() -> None:
    ledger = ExchangeLedger(defer_max_sec_close=True)
    ctx = PhysicalContext(p1_chaining=True)
    ledger.push(ExchangeEvent(
        kind=EventKind.FIRE, side=Side.P1, t_sec=0.0,
        amount=10.0, chain_id=1, source="test"), ctx)
    ledger.close_expired(61.0)
    ledger.push(ExchangeEvent(
        kind=EventKind.FINALIZE, side=Side.P1, t_sec=62.0,
        amount=20.0, chain_id=1, source=FINALIZE_SOURCE_SCORE_OCR_DIFF), ctx)
    assert ledger.closed_episodes()[0].has_settlement_input is False


def test_side_wipe_late_finalize_updates_retired_ledger_without_new_episode() -> None:
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    tracker.observe_frame(
        t_sec=0.0, context=_ctx(p1_chaining=True),
        chain_observations=(_chain(0.0, 700),))
    wiped = tracker.observe_frame(
        t_sec=1.0, context=_ctx(), wiped_sides=(Side.P2,))
    assert wiped.last_close_reason == "side_wipe"
    assert wiped.ledger.retired_generated == pytest.approx(10.0)

    finalized = tracker.observe_frame(
        t_sec=2.0, context=_ctx(),
        generation_observations=(_generation(2.0, 20),))
    assert len(tracker.closed_episodes_live()) == 1
    assert finalized.ledger.is_unresolved is False
    assert finalized.ledger.total_generated == 0.0
    assert finalized.ledger.retired_generated == pytest.approx(20.0)
    assert finalized.ledger.post_retire_backfilled_count == 1

    settled = tracker.observe_frame(
        t_sec=3.0, context=_ctx(),
        settlement=SettlementObservation(
            t_sec=3.0, game_idx=0, canceled_by_1p=0.0,
            canceled_by_2p=20.0, landed_on_1p=0.0, landed_on_2p=0.0))
    assert len(tracker.closed_episodes_live()) == 1
    assert settled.ledger.retired_canceled == pytest.approx(20.0)
    assert settled.ledger.retired_unreconciled == 0.0
    assert settled.unattributed_settlement_total == 0.0


def test_old_normal_summary_is_reclassified_after_newer_close() -> None:
    ledger = ExchangeLedger(defer_max_sec_close=True)
    ctx = PhysicalContext()
    for chain_id, start in ((1, 0.0), (2, 2.0)):
        ledger.push(ExchangeEvent(
            kind=EventKind.FIRE, side=Side.P1, t_sec=start,
            amount=10.0, chain_id=chain_id, source="test"), ctx)
        ledger.push(ExchangeEvent(
            kind=EventKind.CANCEL, side=Side.P1, t_sec=start + 1.0,
            amount=10.0, chain_id=chain_id, source="test"), ctx)
    ledger.push(ExchangeEvent(
        kind=EventKind.FINALIZE, side=Side.P1, t_sec=4.0,
        amount=20.0, chain_id=1, source=FINALIZE_SOURCE_SCORE_OCR_DIFF), ctx)
    closed = ledger.closed_episodes()
    assert len(closed) == 2
    assert closed[0].unreconciled == pytest.approx(10.0)
    assert closed[0].status is EpisodeStatus.CLOSED_FORCED
    assert closed[0].close_reason == "late_finalize_after_normal_close"
    assert closed[1].unreconciled == 0.0
    assert ledger.closed_normal_unreconciled_count() == 0


def test_mapped_chain_side_wipe_does_not_duplicate_retired_unreconciled() -> None:
    ledger = ExchangeLedger(defer_max_sec_close=True)
    ctx = PhysicalContext()
    ledger.push(ExchangeEvent(
        kind=EventKind.FIRE, side=Side.P1, t_sec=0.0,
        amount=10.0, chain_id=1, source="test"), ctx)
    ledger.close_expired(61.0)
    ledger.retire_side_chains(Side.P2, 62.0, ctx)
    wiped = ledger.snapshot(ctx)
    assert wiped.unreconciled == pytest.approx(10.0)
    assert wiped.retired_unreconciled == 0.0
    ledger.push(ExchangeEvent(
        kind=EventKind.CANCEL, side=Side.P1, t_sec=63.0,
        amount=10.0, chain_id=1, source="test"), ctx)
    settled = ledger.snapshot(ctx)
    assert settled.unreconciled == 0.0
    assert settled.retired_unreconciled == 0.0
    assert ledger.closed_episodes()[0].unreconciled == 0.0
