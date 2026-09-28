"""E34の起点保持・現在層分離・入力不足の検出を確認する。"""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pytest

from src.board import Board
from src.board_state_machine import BoardState
from src.chain_detector import ChainEvent
from src.exchange_event_overlay import ExchangeEventOverlay
from src.exchange_event_record import ExchangeEventRecorder, read_records
from src.exchange_event_tracker import ExchangeChainRecord
from src.exchange_prefire_origin import PrefireOriginGuard, origin_match_gate
from src.match_range_gate import MatchRangeGate
from tests.test_exchange_event_overlay import Models, Signals, build_static, result

START, END, FIRE = 1., 10., 4.


def side(color: int = 1, hold: bool = False) -> NS:
    """公開STABLE盤面と収集側の保持印を独立に設定する。"""
    board = Board()
    board._grid[-1, 0] = color
    return NS(state=BoardState.STABLE, confirmed_board=board,
              next_pair=(1, 2), dnext_pair=(3, 4), prefire_origin_hold=hold)


def guard() -> PrefireOriginGuard:
    return PrefireOriginGuard(MatchRangeGate(((START, END),)))


@pytest.mark.parametrize('stamp,hold', [(0., False), (2., True), (END+1, False)])
def test_invalid_origin_holds_last_valid(stamp: float, hold: bool) -> None:
    value = guard()
    value.observe((side(), side()), START)
    value.observe((side(2, hold), side(2, hold)), stamp)
    assert [len(h) for h in value.history] == [1, 1]
    assert value.history[0][-1].board._grid[-1, 0] == 1


def test_chain_end_resumes_and_side_holds_are_independent() -> None:
    value = guard()
    value.observe((side(), side()), START)
    value.observe((side(2, True), side(2)), 2.)
    assert [len(h) for h in value.history] == [1, 2]
    value.observe((side(3), side(3)), 3.)
    assert value.history[0][-1].board._grid[-1, 0] == 3


@pytest.mark.parametrize('state', [BoardState.CHAIN, BoardState.TSUMO_FALL])
def test_nonstable_is_never_an_origin(state: BoardState) -> None:
    value, sample = guard(), side()
    sample.state = state
    value.observe((sample, sample), START)
    assert value.history == [[], []]


def test_notification_board_and_same_frame_are_not_saved_origins() -> None:
    value = guard()
    value.observe((side(), side()), START)
    value.observe((side(3), side(3)), FIRE)
    chain = ExchangeChainRecord('1P', 1, FIRE, FIRE)
    event = NS(before_board=side(2).confirmed_board, chain_count=6)
    selected = value.select(chain, event, 0, 0)
    assert selected.before_board._grid[-1, 0] == 1
    assert event.before_board._grid[-1, 0] == 2
    selected.before_board._grid[-1, 0] = 4
    assert value.history[0][0].board._grid[-1, 0] == 1


def test_reset_does_not_reuse_previous_match() -> None:
    value = guard()
    value.observe((side(), side()), START)
    value.reset()
    selected = value.select(ExchangeChainRecord('1P', 1, FIRE, FIRE),
                            NS(before_board=Board()), 0, 1)
    assert selected.before_board is None
    assert value.history == [[], []]


def test_missing_hold_is_not_silently_treated_as_false() -> None:
    sample = side()
    del sample.prefire_origin_hold
    with pytest.raises(ValueError, match='保持印が未取得'):
        guard().observe((sample, sample), START)


def test_missing_match_ranges_are_not_silently_accepted(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='試合範囲が未取得'):
        origin_match_gate('missing', tmp_path)


def test_current_layer_still_remembers_held_board() -> None:
    overlay = ExchangeEventOverlay(Models(), build_static, Signals,
        prefire_snapshot=True, prefire_origin_guard=True,
        prefire_match_gate=MatchRangeGate(((START, END),)))
    sides = (side(2, True), side(3, True))
    overlay._origin_guard.observe(sides, START)
    overlay._remember(sides, NS(), START)
    assert overlay._origin_guard.history == [[], []]
    np.testing.assert_array_equal(overlay._history[0][-1].board._grid, sides[0].confirmed_board._grid)
    assert sides[0].state == BoardState.STABLE


def test_hold_marker_round_trip(tmp_path: Path) -> None:
    path, sample = tmp_path/'record.jsonl.gz', result(0.)
    sample.p1.prefire_origin_hold = True
    sample.p2.prefire_origin_hold = False
    snapshot = NS(net_balance_capped=0, forecast_p1=0, total_dropped_to_p1=0, total_dropped_to_p2=0)
    finalization = NS(finalized_count_p1=0, finalized_count_p2=0,
                      chain_total_score_p1=0, chain_total_score_p2=0)
    writer = ExchangeEventRecorder(path, 'sample', False, tmp_path)
    writer.update(sample, snapshot, finalization, START, 0, (None, None), (0, 0), (False, False))
    writer.close()
    restored = list(read_records(path))[1]['args'][0]
    assert restored.p1.prefire_origin_hold is True
    assert restored.p2.prefire_origin_hold is False


def test_guard_is_default_off() -> None:
    overlay = ExchangeEventOverlay(Models(), build_static, Signals)
    assert overlay._origin_guard is None


def test_frozen_live_notification_is_supported() -> None:
    value = guard()
    value.observe((side(), side()), START)
    event = ChainEvent(FIRE, END, side(2).confirmed_board, 6, 0, 0, 0, 0, 0, 0, False)
    selected = value.select(ExchangeChainRecord('1P', 1, FIRE, FIRE), event, 0, 0)
    assert selected.before_board._grid[-1, 0] == 1
    assert event.before_board._grid[-1, 0] == 2
    assert replace(selected, before_board=event.before_board) == event


def test_enrichment_requires_exact_original_recognition() -> None:
    from scripts.enrich_e34_origins import attach
    original, replayed = result(0.), result(0.)
    replayed.p1.landing_chain_started = True
    replayed.p2.landing_chain_started = False
    attach(original, replayed, START)
    assert original.p1.prefire_origin_hold is True
    assert original.p2.prefire_origin_hold is False
    replayed.p1.score += 1
    with pytest.raises(ValueError, match='認識再生不一致'):
        attach(original, replayed, START)


def test_live_render_uses_collect_flags_and_frozen_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import scripts.visualize_advantage_overlay as viz
    import src.exchange_prefire_origin as origin
    import src.prefire_snapshot_reader as reader
    from src.recognition_pipeline import SideResult, PipelineResult
    from tests.test_exchange_event_overlay import stub, FPS
    stub(monkeypatch)
    config: dict = {}
    class LivePipeline:
        def update(self, index: int, stamp: float, frame: np.ndarray) -> PipelineResult:
            raw = result(stamp)
            sides = [SideResult(f'{i+1}P', s.state, s.confirmed_board, None,
                s.confirmed_board, None, s.score, 0, s.chain_event,
                next_pair=s.next_pair, dnext_pair=s.dnext_pair, next_slide_motion=False,
                landing_chain_started=True) for i, s in enumerate((raw.p1, raw.p2))]
            return PipelineResult(index, stamp, True, *sides)
        def tsumo_count(self, side: str) -> int:
            return 1
    def load(**kwargs: Any) -> LivePipeline:
        config.update(kwargs)
        return LivePipeline()
    monkeypatch.setattr(viz.RecognitionPipeline, 'load_default', load)
    monkeypatch.setattr(origin, 'origin_match_gate', lambda *a: MatchRangeGate(((0., END),)))
    monkeypatch.setattr(reader, 'PrefireSnapshotReader', lambda: NS(update=lambda *a: (None, None)))
    dest = tmp_path/'inputs.jsonl.gz'
    viz.generate(tmp_path/'sample.mp4', tmp_path/'out.mp4', FIRE, 1/FPS,
        render=False, enable_exchange_event_update=True, exchange_event_m0_predictor=lambda b, q: .5,
        prefire_snapshot=True, prefire_origin_guard=True, exchange_event_record_path=dest)
    assert config['enable_landing_chain_record_hold'] is True
    assert config['enable_chain_active_record_hold'] is True
    updates = [r for r in read_records(dest) if r['kind'] == 'update']
    assert updates and all(r['args'][0].p1.prefire_origin_hold for r in updates)
