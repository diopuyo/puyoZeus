"""A統合の本番配線・死亡証拠・試合境界の保持を検証する。"""
from __future__ import annotations

from dataclasses import asdict
import inspect
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.board import Board
from src.board_state_machine import BoardState
from src.exchange_history_integrity import MatchHistoryIntegrity
from src.exchange_post_counter_bound import PostCounterDeathBound
from tests.test_falsedeath_v2 import CASES
from tests.test_falsedeath_v2d import sides


def test_production_render_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """登録を差し替えず、main→generate→実overlayの全機能を確認する。"""
    from scripts import visualize_advantage_overlay as render
    from tests.test_exchange_event_static_gfe_cli import start_render
    original, captured = render.generate, {}

    def generate(*args: object, **kwargs: object) -> int:
        captured.update(inspect.signature(original).bind(*args, **kwargs).arguments)
        return original(*args, **kwargs)

    monkeypatch.setattr(render, 'generate', generate)
    monkeypatch.setattr('src.exchange_event_terminal.ObservedDeathDetector', lambda: None)
    overlay = start_render(monkeypatch, ['--production-exchange-event'])
    assert captured['falling_puyo_integrity'] is True
    assert captured['exchange_event_switch_smoothing'] is True
    models, projection = overlay.tracker.models, overlay._landing_projection
    assert type(models.estimators['G_fe']) is HistGradientBoostingClassifier
    assert type(models.landing_models.estimators['G_fe']) is LogisticRegression
    assert projection.death_input_integrity
    assert overlay.tracker.score_attribution_v2
    assert overlay._history_integrity is not None
    assert projection.post_counter_bound.current_puyo_integrity_v2c
    assert not projection.post_counter_bound.current_puyo_integrity
    assert not projection.post_counter_bound.input_integrity_v2


def test_production_d4_reaches_recognition(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """同じmainからRecognitionPipeline.load_defaultのD4引数まで到達する。"""
    from scripts import visualize_advantage_overlay as render
    helpers = importlib.import_module('tests.test_advantage_overlay_production_recognition_2026-08-13')
    helpers._stub(monkeypatch, n_frames=0)
    captured = {}

    class PipelineReady(Exception):
        """認識器への実引数を受け取った時点で映像処理を止める。"""

    def capture(**kwargs: object) -> None:
        captured.update(kwargs)
        raise PipelineReady

    monkeypatch.setattr(render.RecognitionPipeline, 'load_default', capture)
    monkeypatch.setattr('src.exchange_event_terminal.ObservedDeathDetector', lambda: None)
    monkeypatch.setattr('src.exchange_event_m0.FileM0Predictor', lambda path: lambda *args: .5)
    monkeypatch.setattr('src.prefire_snapshot_reader.PrefireSnapshotReader', lambda: NS())
    monkeypatch.setattr('src.midchain_board_reader.MidchainBoardReader', lambda: NS())
    monkeypatch.setattr(sys, 'argv', ['render', '--production-exchange-event',
                                    '--out', str(tmp_path/'out.mp4')])
    with pytest.raises(PipelineReady):
        render.main()
    assert captured['enable_falling_puyo_integrity'] is True


@pytest.mark.parametrize('retain', [False, True])
def test_audits_cross_many_match_boundaries(retain: bool) -> None:
    """通常は試合内、保存再生は全原票を保持する。recordの証拠は消さない。"""
    from src.exchange_event_tracker import ExchangeRecord
    from src.exchange_integrity_memory import integrity_memory_roots
    guard, engine = MatchHistoryIntegrity(retain_audit=retain), PostCounterDeathBound(retain_audit=retain)
    saved = None
    for game in range(100):
        pair = sides(0, 0)
        guard.observe(pair, game, float(game))
        engine.observe(NS(p1=pair[0], p2=pair[1]), game)
        guard.audit.append(dict(game=game))
        engine.audit.append(dict(game=game))
        if game == 0:
            saved = ExchangeRecord(1, game, 0., history_integrity=guard.games[game])
    assert len(guard.games) == len(guard.audit) == len(engine.audit) == (100 if retain else 1)
    assert asdict(saved)['history_integrity']['game'] == 0
    roots = integrity_memory_roots(NS(_history_integrity=guard,
        _landing_projection=NS(post_counter_bound=engine)))
    assert roots == dict(history_integrity=guard, post_counter_bound=engine)


def test_history_revokes_public_proofs_without_mutating_cached_evidence() -> None:
    """公開deadを統一し、計算証明・キャッシュ・封印済み記録を保つ。"""
    guard = MatchHistoryIntegrity()
    guard.observe(sides(20, 10), 0, 0.)
    original = dict(dead=True, proofs=[dict(dead=True)])
    value = dict(source='unavoidable_death', p1=1., t_sec=1., dead_sides=['2P'],
        single_death_proof=[None, original], post_counter_bound=[original])
    guard.apply(value, .6, [])
    assert value['dead_sides'] == [] and value['p1'] == .6
    for row in (value['single_death_proof'][1], value['post_counter_bound'][0]):
        assert not row['dead'] and row['computed_dead']
        assert row['history_integrity']['blocked']
    assert original['dead'] and 'computed_dead' not in original


def test_sealed_record_keeps_history_after_boundary(tmp_path: Path) -> None:
    """実trackerが作ったrecordを境界越しに保存して開始証拠を読む。"""
    from src.exchange_event_tracker import ExchangeEventTracker
    from tests.test_exchange_event_tracker import Models, fire
    tracker, guard = ExchangeEventTracker(Models()), MatchHistoryIntegrity()
    tracker.history_integrity = guard
    tracker.boundary(0, 0.)
    guard.observe(sides(0, 0), 0, 0.)
    guard.observe(sides(2, 0), 0, 1.)
    fire(tracker)
    tracker.boundary(1, 3.)
    guard.observe(sides(0, 0), 1, 3.)
    path = tmp_path/'events.jsonl'
    tracker.save(path)
    saved = json.loads(path.read_text())
    assert set(guard.games) == {1}
    assert saved['history_integrity']['history_complete']
    assert saved['history_integrity']['match_start_observed_sec'] == 1.


def signal_result(queue: tuple, motion: bool = False) -> NS:
    """終了センサが読む実観測だけを用意する。"""
    side = NS(state=BoardState.CHAIN, next_pair=queue[:2], dnext_pair=queue[2:],
        next_slide_motion=motion, score=0, chain_event=None, confirmed_board=Board())
    return NS(p1=side, p2=side)


def test_e35_positive_control_real_chain_to_next() -> None:
    """未知手持ち前置は未証明、CHAIN省略は死亡、NEXT移動後は再び未証明。"""
    from scripts.visualize_advantage_overlay import _ExchangeEventEndSignals
    _, encoded, queue, incoming = CASES[0]
    board = Board.from_list([[int(c) for c in row] for row in encoded.split('/')])
    result = signal_result(queue)
    snapshot = NS(total_dropped_to_p1=0, total_dropped_to_p2=0)
    signal = _ExchangeEventEndSignals(0, result, snapshot, 0.)
    chain = NS(chain_id=1, trigger_sec=0., end_signal_sec=None,
        predicted_final_score=0., predicted_final_board=None)
    overlay = NS(_signals={1: signal}, _game=0,
        tracker=NS(latest_chain=lambda side: chain, _score_elapsed=0.))
    engine = PostCounterDeathBound(current_puyo_integrity_v2c=True)
    engine.observe(result, 0)
    engine.colors = [NS(active=lambda: (2, 3, 4, 5), observe=lambda side: None)] * 2
    projection = NS(_known_budget=lambda *a: True, _chaining=lambda *a: True)
    chain.predicted_final_board = board
    context = dict(hidden=[None]*2, certain=[True]*2, credit=[0]*2,
                   verified=[True]*2, replies=[board]*2)
    before = engine.prove(projection, overlay, 0, (), incoming, 1, context, 0.)
    assert not before['dead'] and before['unknown_current_pair_prepended']
    for stamp in (1., 1.1):
        signal.update(result, snapshot, stamp)
    engine.refresh_current_observation(overlay)
    revision = engine.revision
    during = engine.prove(projection, overlay, 0, (), incoming, 1, context, 1.)
    assert during['dead'] and during['unknown_current_pair_omitted_self_chain']
    moved = signal_result(queue[2:] + (2, 2), motion=True)
    for stamp in (2., 2.1):
        ended = signal.update(moved, snapshot, stamp)
    assert ended == 'next'
    engine.observe(moved, 0)
    engine.refresh_current_observation(overlay)
    assert engine.revision > revision
    after = engine.prove(projection, overlay, 0, (), incoming, 1, context, 2.1)
    assert not after['dead'] and after['unknown_current_pair_prepended']
