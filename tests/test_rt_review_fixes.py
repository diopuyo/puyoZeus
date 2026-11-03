"""統合Bの査読指摘: 監査上限・入力順序・計測障害・CPU上限。"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


def test_h1_marks_new_proofs_after_audit_limit() -> None:
    from src.phase_j.live_memory import RecentIntegrityAudit, INTEGRITY_AUDIT_LIMIT
    from src.exchange_event_landing import ExchangeLandingProjection
    from src.exchange_history_integrity import MatchHistoryIntegrity
    from tests.test_falsedeath_v2d import sides
    audit = RecentIntegrityAudit([{} for _ in range(INTEGRITY_AUDIT_LIMIT)])
    history = MatchHistoryIntegrity()
    history.observe(sides(2, 2), 1, 0.)
    projection = ExchangeLandingProjection()
    projection.post_counter_bound = NS(audit=audit)
    projection.death_only_inputs = False
    projection.multi_landing_death = False
    projection.safety = None
    projection._probability_before_death = .6

    def evaluate(*args: object) -> dict:
        audit.append(dict(dead=True))
        return dict(source='unavoidable_death', dead_sides=['1P'], p1=.01, t_sec=1.)

    projection._evaluate_single = evaluate
    value = projection._evaluate(NS(_history_integrity=history), None, (), [], (), {}, 1.)
    assert value['source'] == 'S3_landing'
    assert audit[-1]['history_integrity']['blocked']
    assert audit.discarded_count == 1 and len(audit) == INTEGRITY_AUDIT_LIMIT
    cursor = audit.append_count
    audit.clear()
    audit.append(dict(dead=False))
    assert audit.rows_since(cursor) == [dict(dead=False)]


@pytest.mark.parametrize('held', [False, True])
def test_e35_input_timeline_matches_replay(held: bool) -> None:
    from src.exchange_event_overlay import ExchangeEventOverlay
    from src.phase_j.live_evaluation import SplitExchangeOverlay
    from src.exchange_post_counter_bound import PostCounterDeathBound
    from tests.test_exchange_event_tracker import Models
    from tests.test_exchange_event_overlay import result, Signals, build_static
    traces = []
    for cls in (ExchangeEventOverlay, SplitExchangeOverlay):
        overlay = cls(Models(), build_static, Signals, lambda b, q: .5)
        engine = PostCounterDeathBound(current_puyo_integrity_v2c=True)
        overlay._landing_projection.post_counter_bound = engine
        trace = []
        original = engine.refresh_current_observation

        def refresh(owner: object) -> None:
            original(owner)
            trace.append((engine.game, engine.states, engine.queues,
                          engine.next_motion, engine.current_observation_key))

        engine.refresh_current_observation = refresh
        if held:
            overlay._e16 = Mock(before=Mock(return_value=True))
        for t in (0., 1., 1.5, 2., 2.3, 2.6, 3.):
            observed = result(t)
            observed.p1.next_slide_motion = t == 1.5
            snapshot = NS(net_balance_capped=0, forecast_p1=0,
                          total_dropped_to_p1=0, total_dropped_to_p2=0)
            final = NS(finalized_count_p1=int(t >= 2.6), finalized_count_p2=0,
                       chain_total_score_p1=700 if t >= 2.6 else 0, chain_total_score_p2=0)
            overlay.update(observed, snapshot, final, t, 1)
        traces.append(trace)
    assert traces[0] == traces[1]
    assert len(traces[0]) >= 7
    assert any(row[3][0] for row in traces[0])


def test_missing_and_zero_recognition_audits_are_distinct() -> None:
    from src.phase_j.live_bridge import RecognitionBridge, _RecognitionAudits, recognition_guard_audits
    bridge = RecognitionBridge(False, Mock())
    code = 'dict(counts=pipe.next_recolor_guard_counts, events=pipe.next_recolor_guard_log)'
    tree = ast.fix_missing_locations(_RecognitionAudits().visit(ast.parse(code, mode='eval')))
    missing = eval(compile(tree, '<audit>', 'eval'), dict(_live_bridge=bridge))
    assert missing['summary_received'] is False and missing['recognition_frames'] is None
    bridge.recognition_audits = recognition_guard_audits(NS(), 12)
    bridge.recognition_summary_received = True
    received = eval(compile(tree, '<audit>', 'eval'), dict(_live_bridge=bridge))
    assert received == dict(counts={}, events=[], summary_received=True, recognition_frames=12)


def test_memory_failure_does_not_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                       caplog: pytest.LogCaptureFixture) -> None:
    from src.phase_j.live_memory import MemoryProbe
    monkeypatch.delenv('PUYO_MEMORY_PROFILE', raising=False)
    probe = MemoryProbe('test')
    probe._sample = Mock(side_effect=OSError('容量不足'))
    probe.sample(0., {})
    assert 'メモリ計測に失敗' in caplog.text


def test_disabled_placement_logger_warns(monkeypatch: pytest.MonkeyPatch,
                                         caplog: pytest.LogCaptureFixture) -> None:
    from src.phase_j import live_process
    monkeypatch.setattr('src.recognition_pipeline.RecognitionPipeline.load_default',
                        lambda **kw: NS(_placement_reconcile=None))
    monkeypatch.setattr('src.phase_j.live_cpu.apply_runtime', lambda role: {})
    monkeypatch.setattr('src.phase_j.live_lifetime.protect_parent', lambda pid: None)
    monkeypatch.setattr(live_process, 'run_recognition_input', Mock())
    live_process.recognition_worker(Mock(), Mock(), {}, '', (30, 0, 0, 1), False, None,
                                    placement_log_path='test.jsonl')
    assert '設置照合器が無効' in caplog.text


@pytest.mark.parametrize('threads', [0, 1])
def test_histgbc_worker_is_single_thread(threads: int, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits, threadpool_info
    from src.phase_j.live_cpu import apply_runtime, THREAD_ENV
    monkeypatch.setenv(THREAD_ENV, str(threads))
    # torchの一度限りのinterop設定は本テストで変更しない。
    import torch
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda value: None)
    with threadpool_limits(limits=2):
        runtime = apply_runtime('event-evaluation', lower_priority=False)
        x = np.arange(80).reshape(40, 2)
        model = HistGradientBoostingClassifier(max_iter=2).fit(x, np.arange(40) % 2)
        model.predict_proba(x[:1])
        pools = [p for p in threadpool_info() if p['user_api'] in ('openmp', 'blas')]
        assert pools and all(p['num_threads'] == 1 for p in pools)
        assert runtime['threadpools']


def test_launch_defaults_reach_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import sys
    from scripts.run_live_pipeline_20260928 import parse_args
    from src.phase_j.live_cpu import THREAD_ENV
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    monkeypatch.setattr(sys, 'argv', ['live'])
    options = parse_args()
    assert options.cpu_threads == 1 and options.evaluation_nice == 10
    assert os.environ[THREAD_ENV] == '1'


def test_audit_discard_count_is_recorded() -> None:
    from src.phase_j.live_memory import RecentIntegrityAudit, retained_counts, INTEGRITY_AUDIT_LIMIT
    audit = RecentIntegrityAudit([{} for _ in range(INTEGRITY_AUDIT_LIMIT+3)])
    counts = retained_counts(dict(engine=NS(audit=audit)))
    assert counts['engine.audit.discarded'] == 3
    assert counts['engine.audit.appended'] == INTEGRITY_AUDIT_LIMIT+3


def test_memory_failure_preserves_worker_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_eval_worker
    from src.phase_j.live_memory import MemoryProbe
    monkeypatch.delenv('PUYO_MEMORY_PROFILE', raising=False)
    probe = MemoryProbe('test')
    probe.path = Path('unused')
    probe._sample = Mock(side_effect=RuntimeError('計測だけ失敗'))
    monkeypatch.setattr('src.phase_j.live_memory.MemoryProbe', lambda role: probe)
    monkeypatch.setattr(live_eval_worker, 'execute', lambda *args: dict(kind='ok', probability=.7))
    connection = Mock()
    connection.recv.side_effect = [dict(op='calculate', request_id=1), dict(op='close')]
    overlay = NS(tracker=NS(), _landing_projection=None, _history_integrity=None)
    live_eval_worker.serve(connection, (), overlay)
    assert connection.send.call_args.args[0] == dict(kind='ok', probability=.7,
                                                     request_id=1, op='calculate')
