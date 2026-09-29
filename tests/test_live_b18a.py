"""本番構成の実行時参照、子評価器と通知の追加観測を検証する。"""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import pickle
import shlex
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import production_config
from src.phase_j.live_eval_worker import make_overlay
from src.phase_j.live_evaluation import SplitExchangeOverlay
from tests.test_exchange_event_tracker import Models
from tests.test_exchange_event_overlay import build_static, Signals
from tests.test_live_b16 import inputs, m0


def test_live_default_matches_runtime_getter(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_live_pipeline_20260928 as live
    from tests.test_exchange_event_production import cli_options, RENDER
    monkeypatch.setattr(live, 'legacy_command', lambda options: [])
    flags = production_config.exchange_event_flags()
    getter = Mock(return_value=flags)
    monkeypatch.setattr(production_config, 'exchange_event_flags', getter)
    command = live.build_command(SimpleNamespace())
    getter.assert_not_called()
    effective = cli_options(RENDER, command, monkeypatch)
    getter.assert_called_once_with()
    expected = cli_options(RENDER, shlex.split(flags), monkeypatch)
    assert effective.pop('production_exchange_event') is True
    assert expected.pop('production_exchange_event') is False
    assert effective == expected


def test_live_getter_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_live_pipeline_20260928 as live
    from tests.test_exchange_event_production import cli_options, RENDER
    monkeypatch.setattr(live, 'legacy_command', lambda options: [])
    command = live.build_command(SimpleNamespace())
    monkeypatch.setattr(production_config, 'exchange_event_flags',
                        lambda: '--exchange-event-model-dir models/changed-at-runtime')
    assert cli_options(RENDER, command, monkeypatch)['exchange_event_model_dir'] == Path('models/changed-at-runtime')


def test_worker_preserves_production_tracker_state() -> None:
    options = dict(live_count=True, death_guard=True, confirmed_death_hold=True,
                   prefire_snapshot=True, hidden_row_belief=True)
    overlay = make_overlay(Mock(), (Models(), Signals, m0, True, options))
    assert overlay.tracker.live_count
    assert overlay.tracker.hidden_row_belief is overlay._prefire
    assert overlay._confirmed_death_hold
    assert overlay._e16.death_enabled
    assert overlay.tracker.sealed == []


def test_retention_preserves_production_tracker_state(tmp_path: Path) -> None:
    from contextlib import ExitStack
    from src.phase_j.live_retention import bounded_overlay
    with ExitStack() as stack:
        overlay_type = bounded_overlay(SplitExchangeOverlay, tmp_path, stack)
        overlay = overlay_type(Models(), build_static, Signals, m0, live_count=True,
                               prefire_snapshot=True, hidden_row_belief=True)
        assert overlay.tracker.live_count
        assert overlay.tracker.hidden_row_belief is overlay._prefire


def test_safety_observed_before_publication() -> None:
    overlay = SplitExchangeOverlay(Models(), build_static, Signals, m0, color_score_safety=True)
    safety = overlay._landing_projection.safety
    safety.observe = Mock(wraps=safety.observe)
    for tick in range(5):
        overlay.update(*inputs(tick/30))
    assert safety.observe.call_count == 5
    assert overlay.calculations == 0


def test_count_refresh_precedes_static_close() -> None:
    models = Models()
    models.count_features = True
    overlay = SplitExchangeOverlay(models, build_static, Signals, m0, live_count=True)
    calls = []
    overlay._refresh_live_count = lambda stamp: calls.append('count')
    overlay._mark_static = lambda snapshot, stamp: calls.append('static')
    overlay.update(*inputs(0))
    assert calls == ['count', 'static']


def test_supervisor_passes_all_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_eval_supervisor import SupervisedOverlay
    monkeypatch.setattr(SupervisedOverlay, 'start', Mock())
    overlay = SupervisedOverlay(Models(), build_static, Signals, m0, directory=tmp_path,
                                prefire_snapshot=True, hidden_row_belief=True, live_count=True)
    assert overlay.config[-1] == dict(prefire_snapshot=True, hidden_row_belief=True, live_count=True)
    for storage in (overlay.journal, overlay.archive, overlay.diagnostics):
        storage.close()


def test_confirmed_death_clears_pending_and_holds() -> None:
    overlay = SplitExchangeOverlay(Models(), build_static, Signals, m0,
                                  death_guard=True, confirmed_death_hold=True)
    for tick in range(10):
        args = list(inputs(tick/30))
        args[0].confirmed_dead_sides = ('2P',) if tick >= 5 else ()
        overlay.update(*args)
        overlay.calculate()
    assert overlay.tracker.probability == 1.0
    assert overlay.tracker.source == 'confirmed_death'
    assert overlay.tracker.pending is None


def test_notice_preserves_prediction_inputs() -> None:
    from src.phase_j.live_bridge import RecognitionNotice, PipelineView
    from src.phase_j.live_process import NoticeDeltaCodec
    from src.recognition_pipeline import PipelineResult, SideResult
    from src.board_state_machine import BoardState
    from src.board import Board
    side = SideResult(side='1P', state=BoardState.CHAIN, confirmed_board=None,
        cnn_board=None, inferred_board=None, drift=None, score=None, score_delta=None,
        chain_event=None, next_pair=None, dnext_pair=None,
        midchain_board=Board(), prefire_snapshot={'board': [[0]*6]*13})
    result = PipelineResult(frame_idx=1, time_sec=1., is_match_active=True,
                            p1=side, p2=replace(side, side='2P'),
                            terminal_evidence_available=True, confirmed_dead_sides=('2P',))
    notice = RecognitionNotice(1, 1., 0., 0., pickle.dumps(result),
                               PipelineView((0, 0), (None, None), (None, None), (False, False)), 0)
    decoded = NoticeDeltaCodec().decode(NoticeDeltaCodec().encode(notice)).result()
    assert decoded.p1.prefire_snapshot == result.p1.prefire_snapshot
    assert decoded.p1.midchain_board is not None
    assert decoded.confirmed_dead_sides == ('2P',)
    assert decoded.terminal_evidence_available


def test_evaluation_loop_does_not_read_images() -> None:
    from src.phase_j.live_bridge import _NotificationCalls
    tree = ast.parse('if snapshot_reader is not None:\n    r = snapshot_reader.update(recog_frame)\n'
                     'if terminal_detector is not None:\n    r = terminal_detector.update(recog_frame)\n'
                     'if midchain_reader is not None:\n    r = midchain_reader.read(recog_frame)\n'
                     'event_overlay.update(r)')
    assert ast.unparse(_NotificationCalls().visit(tree)) == 'event_overlay.update(r)'


def test_worker_exports_prediction_audits() -> None:
    from src.phase_j.live_eval_worker import serve
    overlay = make_overlay(Mock(), (Models(), Signals, m0, True,
        dict(midchain_completion=True, hidden_row_death=True, prefire_snapshot=True)))
    connection = Mock()
    connection.recv.side_effect = [dict(op='records'), dict(op='close')]
    serve(connection, (), overlay)
    audits = connection.send.call_args.args[0]['audits']
    assert audits['_origin_guard'] is None
    for name in ('_midchain', '_hidden_death', '_prefire'):
        assert audits[name] == getattr(overlay, name).summary()


def test_supervisor_save_exposes_audits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_eval_supervisor import SupervisedOverlay
    monkeypatch.setattr(SupervisedOverlay, 'start', Mock())
    overlay = SupervisedOverlay(Models(), build_static, Signals, m0, directory=tmp_path)
    overlay.connection = Mock()
    overlay.receive = Mock(return_value=dict(kind='ok', records=[], diagnostics=[],
        audits=dict(_origin_guard=None, _midchain={'rows': [1]}, _hidden_death={'rows': []})))
    overlay.save(tmp_path/'events.jsonl')
    assert overlay._origin_guard is None
    assert overlay._midchain.summary() == {'rows': [1]}
    assert overlay._hidden_death.summary() == {'rows': []}
    for storage in (overlay.journal, overlay.archive, overlay.diagnostics):
        storage.close()
