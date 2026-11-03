"""rt統合のAST接続・E35観測・資産識別・試合監査を検査する。"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


def test_ast_production_options_reach_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """実main→差替generate→worker構築まで本番登録のまま通す。"""
    from scripts import visualize_advantage_overlay as render
    from src.phase_j.live_bridge import RecognitionBridge, build_live_generate
    from src.phase_j.live_eval_worker import make_overlay
    from tests.test_exchange_event_static_gfe_cli import OverlayReady
    bridge = RecognitionBridge(False, lambda *a: None)
    bridge.split_evaluation = True
    captured = []

    def create(models: object, build: object, signals: object, m0: object = None,
               per_side_settled: bool = False, **options: object) -> object:
        overlay = make_overlay(Mock(), (models, signals, m0, per_side_settled, options))
        captured.append(overlay)
        raise OverlayReady

    monkeypatch.setattr(render, 'ExchangeEventOverlay', create)
    monkeypatch.setattr(render, 'generate', build_live_generate(render, bridge))
    monkeypatch.setattr('src.exchange_event_m0.FileM0Predictor', lambda path: None)
    monkeypatch.setattr(sys, 'argv', ['render', '--production-exchange-event'])
    with pytest.raises(OverlayReady):
        render.main()
    overlay = captured[0]
    assert overlay.tracker.score_attribution_v2
    assert overlay._history_integrity is not None
    assert overlay._landing_projection.death_input_integrity
    assert overlay._landing_projection.post_counter_bound.current_puyo_integrity_v2c
    assert overlay._history_integrity.boundary_score_start
    assert overlay._history_integrity.strict_integrity
    bound = overlay._landing_projection.post_counter_bound
    assert bound.completion_recent_integrity and bound.strict_integrity and bound.cap_completed
    assert not bound.retain_audit and not overlay._history_integrity.retain_audit
    assert type(overlay.tracker.models.estimators['G_fe']).__name__ == 'HistGradientBoostingClassifier'
    landing = overlay.tracker.models.landing_models
    assert type(landing.estimators['G_fe']).__name__ == 'LogisticRegression'


def test_split_observes_e35_before_landing() -> None:
    from src.phase_j.live_evaluation import SplitExchangeOverlay
    from src.exchange_post_counter_bound import PostCounterDeathBound
    from tests.test_integration_r1s_dv2d import signal_result
    engine = PostCounterDeathBound(current_puyo_integrity_v2c=True)
    projection = NS(post_counter_bound=engine, _observe_frame=Mock(), safety=None,
        hands_observation=None, _refresh_death=Mock(return_value=False), death=None)
    overlay = NS(_landing_projection=projection, _game=3, _signals={},
        tracker=NS(latest_chain=lambda side: None), _history=[[], []])
    observed = signal_result((1, 2, 3, 4), motion=True)
    engine.observe(observed, overlay._game)
    SplitExchangeOverlay._observe_landing(overlay, observed, NS(), 5.)
    assert engine.game == 3
    assert engine.next_motion == (True, True)
    assert engine.states == ('CHAIN', 'CHAIN')
    assert engine.queues == ((1, 2, 3, 4), (1, 2, 3, 4))
    assert engine.current_observation_key == (None, None)


def test_ast_d4_reaches_recognition_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    from scripts import visualize_advantage_overlay as render
    from src.phase_j.live_bridge import RecognitionBridge, build_live_generate
    helpers = importlib.import_module('tests.test_advantage_overlay_production_recognition_2026-08-13')
    helpers._stub(monkeypatch, n_frames=0)
    captured = {}

    class PipelineReady(Exception):
        """認識workerへ渡す設定を確認した時点で停止する。"""

    def capture(**kwargs: object) -> None:
        captured.update(kwargs)
        raise PipelineReady

    bridge = RecognitionBridge(False, lambda *a: None)
    bridge.pipeline_config, bridge.skip_features = capture, False
    bridge.open_capture = render.cv2.VideoCapture
    monkeypatch.setattr(render, 'generate', build_live_generate(render, bridge))
    monkeypatch.setattr('src.exchange_event_m0.FileM0Predictor', lambda path: None)
    monkeypatch.setattr(sys, 'argv', ['render', '--production-exchange-event'])
    with pytest.raises(PipelineReady):
        render.main()
    assert captured['enable_falling_puyo_integrity'] is True


def test_ast_routes_placement_log_without_accessing_config_as_pipeline(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import importlib
    from scripts import visualize_advantage_overlay as render
    from src.phase_j.live_bridge import RecognitionBridge, build_live_generate
    helpers = importlib.import_module('tests.test_advantage_overlay_production_recognition_2026-08-13')
    helpers._stub(monkeypatch, n_frames=0)
    captured = []

    class LoggingReady(Exception):
        """認識器への誤アクセスがないことを起動終盤まで確認する。"""

    def capture(path: Path) -> None:
        captured.append(path)
        raise LoggingReady

    bridge = RecognitionBridge(False, lambda *a: None)
    bridge.pipeline_config = lambda **kwargs: kwargs
    bridge.placement_log, bridge.skip_features = capture, False
    bridge.open_capture = render.cv2.VideoCapture
    monkeypatch.setattr(render, 'generate', build_live_generate(render, bridge))
    monkeypatch.setattr('src.exchange_event_m0.FileM0Predictor', lambda path: None)
    monkeypatch.setattr(sys, 'argv', ['render', '--production-exchange-event',
                                    '--out', str(tmp_path/'out.mp4'), '--no-render'])
    with pytest.raises(LoggingReady):
        render.main()
    assert captured == [tmp_path/'placement_signal_reconcile.jsonl']


def test_recognition_worker_opens_placement_log_in_child(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.phase_j import live_process
    pipe = NS(_placement_reconcile=Mock())
    monkeypatch.setattr('src.recognition_pipeline.RecognitionPipeline.load_default', lambda **kw: pipe)
    monkeypatch.setattr('src.phase_j.live_cpu.apply_runtime', lambda role: {})
    monkeypatch.setattr('src.phase_j.live_lifetime.protect_parent', lambda pid: None)
    monkeypatch.setattr(live_process, 'run_recognition_input', Mock())
    path = tmp_path/'placement.jsonl'
    live_process.recognition_worker(Mock(), Mock(), {}, '', (30, 0, 0, 1), False, None,
                                    placement_log_path=str(path))
    pipe._placement_reconcile.log_to.assert_called_once_with(path)


@pytest.mark.parametrize('reason', ['skipped', 'calculated'])
def test_recording_closes_every_notification_without_inventing_fallback(
        tmp_path: Path, reason: str) -> None:
    from src.exchange_event_record import ExchangeEventRecorder, read_records
    from src.phase_j.live_bridge import finish_recording
    from tests.test_live_b16 import inputs
    recorder = ExchangeEventRecorder(tmp_path/'input.gz', 'test', True, Path('models/exchange_event_v2'))
    for index in range(2):
        recorder.update(*inputs(float(index)), (None, None), (None, None), (False, False), defer_fallback=True)
        if reason == 'calculated':
            recorder.fallback(0., .5)
        finish_recording(dict(event_recorder=recorder))
        assert recorder.pending_update is None
    recorder.close()
    updates = [row for row in read_records(tmp_path/'input.gz') if row['kind'] == 'update']
    assert len(updates) == 2
    assert all(('fallback_p1' in row) == (reason == 'calculated') for row in updates)


def test_recognition_audits_return_to_parent() -> None:
    from src.phase_j.live_bridge import recognition_guard_audits
    from src.phase_j.live_process import ProcessRecognitionBridge
    pipe = NS(next_recolor_guard_counts={'accepted': 4}, next_recolor_guard_log=[{'frame': 8}],
              verification_pending_expired_count=3)
    audits = recognition_guard_audits(NS(pipe=pipe))
    bridge = ProcessRecognitionBridge.__new__(ProcessRecognitionBridge)
    bridge.source = NS()
    bridge._message(('summary', dict(dropped=0, sent=0, wire_bytes=0, recognition_audits=audits)))
    assert bridge.recognition_audits == audits


def test_r1s_asset_bytes_affect_prediction_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_live_pipeline_20260928 as live
    payload = b'first'

    def read(path: Path) -> bytes:
        return payload if path.as_posix().endswith('exchange_event_v1_r1_hgb/G_fe.joblib') else b'asset'

    monkeypatch.setattr(Path, 'read_bytes', read)
    before = live.asset_hashes()
    payload = b'changed'
    after = live.asset_hashes()
    assert before['prediction_model_hash'] != after['prediction_model_hash']
    assert before['recognition_model_hash'] == after['recognition_model_hash']


def test_live_memory_reports_integrity_retention() -> None:
    from src.phase_j.live_memory import evaluation_roots, retained_counts
    from src.exchange_history_integrity import MatchHistoryIntegrity
    from src.exchange_post_counter_bound import PostCounterDeathBound
    overlay = NS(_history_integrity=MatchHistoryIntegrity(),
        _landing_projection=NS(post_counter_bound=PostCounterDeathBound()))
    bridge = NS(callback=lambda: None, meter=None)
    counts = retained_counts(evaluation_roots(dict(event_overlay=overlay), bridge))
    assert counts['post_counter_bound.audit'] == 0
    assert counts['history_integrity.audit'] == 0
    assert counts['history_integrity.games'] == 0


@pytest.mark.parametrize('retain', [False, True])
def test_integrity_audits_are_bounded_inside_one_long_match(retain: bool) -> None:
    from src.phase_j.live_memory import INTEGRITY_AUDIT_LIMIT, bound_integrity_audits
    from src.exchange_history_integrity import MatchHistoryIntegrity
    from src.exchange_post_counter_bound import PostCounterDeathBound
    bound = PostCounterDeathBound(retain_audit=retain)
    history = MatchHistoryIntegrity(retain_audit=retain)
    overlay = NS(_history_integrity=history, _landing_projection=NS(post_counter_bound=bound))
    bound_integrity_audits(overlay)
    total = INTEGRITY_AUDIT_LIMIT*3
    for index in range(total):
        bound.audit.append(dict(index=index))
        history.audit.append(dict(index=index))
    for owner in (bound, history):
        assert len(owner.audit) == (total if retain else INTEGRITY_AUDIT_LIMIT)
        assert owner.audit[-1]['index'] == total-1
    bound_integrity_audits(overlay)
    assert len(bound.audit) == (total if retain else INTEGRITY_AUDIT_LIMIT)


@pytest.mark.parametrize('enabled', [False, True])
def test_worker_seals_full_audit_with_monotonic_ids(enabled: bool) -> None:
    from src.phase_j.live_eval_worker import boundary_tracker
    from src.exchange_event_tracker import ExchangeChainRecord
    from src.exchange_history_integrity import MatchHistoryIntegrity
    from tests.test_exchange_event_tracker import Models, fire
    from tests.test_falsedeath_v2d import sides
    tracker = boundary_tracker(Models())
    tracker.score_attribution_v2 = enabled
    guard = MatchHistoryIntegrity()
    if enabled:
        tracker.history_integrity = guard
    for game in range(3):
        tracker.boundary(game, float(game))
        guard.observe(sides(0, 0), game, float(game))
        fire(tracker)
        record = tracker.current
        record.chains.append(ExchangeChainRecord('1P', game, float(game), float(game)))
        record.chains[0].independent_score_required = True
        record.chains[0].finalized_evidence_score = 700.
    tracker.boundary(3, 3.)
    assert [row['exchange_id'] for row in tracker.sealed] == [1, 2, 3]
    for game, row in enumerate(tracker.sealed):
        assert ('history_integrity' in row) is enabled
        if enabled:
            assert row['history_integrity']['game'] == game
            assert row['chains'][0]['independent_score_required']
            assert row['chains'][0]['finalized_evidence_score'] == 700.
