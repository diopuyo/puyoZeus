"""公開を間引いても通知と物理遷移を失わないことを検査する。"""
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path

import numpy as np
import pytest

from src.phase_j.live_evaluation import DeferredTracker, SplitExchangeOverlay, sampled_ema
from tests.test_exchange_event_tracker import Models, static, fire
from tests.test_exchange_event_overlay import result, Signals, build_static


def test_deferred_latest_only() -> None:
    models = Models()
    models.predict_source_probability = Mock(wraps=models.predict_source_probability)
    tracker = DeferredTracker(models)
    tracker.boundary(1, 0)
    for t in range(20):
        tracker._evaluate(static(), 'G_fe', t)
    assert models.predict_source_probability.call_count == 0
    tracker.calculate()
    assert models.predict_source_probability.call_count == 1
    assert tracker.probability == .4
    tracker.calculate()
    assert models.predict_source_probability.call_count == 1


@pytest.mark.parametrize('boundary', [False, True])
def test_s3_state_does_not_wait_for_calculation(boundary: bool) -> None:
    tracker = DeferredTracker(Models())
    tracker.boundary(1, 0)
    tracker.begin_frame()
    fire(tracker)
    tracker.end('1P', 3, 'next')
    tracker.finalize('1P', 3, 700)
    tracker.confirm_frame_inputs(4)
    tracker.finish_frame(4)
    assert tracker._s3_sec == 4
    assert tracker.probability is None
    if boundary:
        tracker.boundary(2, 5)
    else:
        assert tracker.close_confirmed(5, (5, 5))
    tracker.calculate()
    assert tracker.pending is None
    assert tracker.current is None
    assert tracker.probability is None


@pytest.mark.parametrize('stride', [1, 2, 15, 30])
def test_all_notifications_keep_history_and_scores(stride: int) -> None:
    overlay = SplitExchangeOverlay(Models(), build_static, Signals, lambda b, q: .5,
                                   per_side_settled=True)
    for frame in range(120):
        t = frame/30
        final = SimpleNamespace(finalized_count_p1=int(t >= 2.6), finalized_count_p2=0,
            chain_total_score_p1=700 if t >= 2.6 else 0, chain_total_score_p2=0)
        snap = SimpleNamespace(net_balance_capped=0, forecast_p1=0,
                               total_dropped_to_p1=0, total_dropped_to_p2=0)
        overlay.update(result(t), snap, final, t, 1)
        if frame % stride == 0:
            overlay.calculate()
    assert overlay.notifications == 120
    assert overlay.calculations == len(range(0, 120, stride))
    chain = overlay.tracker.records[0].chains[0]
    assert chain.score_finalize_sec == 2.6
    assert chain.score_delta == 700
    assert overlay.tracker.records[0].landings[-1]['t_sec'] == 2.5


@pytest.mark.parametrize('count', [1, 2, 15, 30])
def test_sampled_ema_constant_input_is_exact(count: int) -> None:
    from scripts.visualize_advantage_overlay import _ExchangeDisplayEMA
    bridge = SimpleNamespace(notification_count=count)
    candidate = sampled_ema(_ExchangeDisplayEMA, bridge)()
    reference = _ExchangeDisplayEMA()
    for i in range(count):
        expected = reference.apply(80, .8, i)
    assert candidate.apply(80, .8, count) == pytest.approx(expected)
    assert candidate.apply(80, .8, count) == pytest.approx(expected)


@pytest.mark.parametrize('latest', [0, 1, 5])
@pytest.mark.parametrize('remaining', [0, 1])
def test_scheduler_drains_state_before_calculating(latest: int, remaining: int) -> None:
    import time
    from src.phase_j.live_bridge import RecognitionBridge
    bridge = RecognitionBridge(True, Mock())
    bridge.latest_frame = SimpleNamespace(value=latest)
    bridge.batch_remaining = remaining
    bridge.state_started = time.perf_counter()
    assert bridge.calculation_due(SimpleNamespace(frame=1, t_sec=0)) == (latest <= 1 and remaining == 0)
    assert bridge.notification_count == 1
    assert len(bridge.state_updates) == 1


@pytest.mark.parametrize('realtime', [False, True])
def test_scheduler_caps_calculations(monkeypatch: pytest.MonkeyPatch, realtime: bool) -> None:
    from src.phase_j.live_bridge import RecognitionBridge
    bridge = RecognitionBridge(realtime, Mock())
    for i in range(31):
        now = i/30
        monkeypatch.setattr('src.phase_j.live_bridge.time.perf_counter', lambda: now)
        bridge.state_started = now
        packet = SimpleNamespace(frame=i, t_sec=now)
        if bridge.calculation_due(packet):
            bridge.calculation_finished(packet)
    assert len(bridge.state_updates) == 31
    assert len(bridge.calculation_rows) == 3
    assert [r['frame'] for r in bridge.calculation_rows] == [0, 15, 30]


@pytest.mark.parametrize('state', ['STABLE', 'CHAIN', 'TSUMO_FALL', 'OJAMA_FALL'])
def test_update_never_calls_probability_models(state: str) -> None:
    from src.board_state_machine import BoardState
    model = Models()
    model.predict_source_probability = Mock(side_effect=AssertionError('状態更新中の推論'))
    m0 = Mock(side_effect=AssertionError('状態更新中のM0'))
    overlay = SplitExchangeOverlay(model, build_static, Signals, m0)
    snap = SimpleNamespace(net_balance_capped=0, forecast_p1=0,
                           total_dropped_to_p1=0, total_dropped_to_p2=0)
    final = SimpleNamespace(finalized_count_p1=0, finalized_count_p2=0,
                            chain_total_score_p1=0, chain_total_score_p2=0)
    for i in range(60):
        r = result(i/30)
        if i > 0:
            r.p2.state = BoardState[state]
        overlay.update(r, snap, final, i/30, 1)
    assert overlay.notifications == 60
    assert model.predict_source_probability.call_count == m0.call_count == 0


def test_split_ast_moves_forecast_and_guards_numeric_path() -> None:
    import ast
    import inspect
    import scripts.visualize_advantage_overlay as overlay
    from src.phase_j.live_bridge import adapt_loop, split_loop
    tree = ast.parse(inspect.getsource(overlay.generate))
    loop = next(n for n in tree.body[0].body if isinstance(n, ast.For))
    adapt_loop(loop)
    split_loop(loop)
    code = ast.unparse(loop)
    assert code.count('fctracker.update(') == 1
    assert code.index('event_overlay.update(') < code.index('calculation_due(')
    assert code.index('fctracker.update(') < code.index('calculation_due(')
    assert code.index('calculation_due(') < code.index('event_overlay.calculate(')
    assert code.index('event_overlay.calculate(') < code.index('hcache.update(')


@pytest.mark.parametrize('enabled', [False, True])
def test_cli_can_select_legacy_for_comparison(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    from scripts.run_live_pipeline_20260928 import parse_args
    argv = ['live', '--split-evaluation' if enabled else '--no-split-evaluation']
    monkeypatch.setattr('sys.argv', argv)
    assert parse_args().split_evaluation is enabled
    monkeypatch.setattr('sys.argv', ['live', '--compare'])
    assert parse_args().split_evaluation is False


@pytest.mark.parametrize('state', ['CHAIN', 'TSUMO_FALL', 'OJAMA_FALL'])
def test_nonstable_board_never_enters_static_features(state: str) -> None:
    from src.board_state_machine import BoardState
    m0 = Mock(return_value=.5)
    overlay = SplitExchangeOverlay(Models(), build_static, Signals, m0)
    snap = SimpleNamespace(net_balance_capped=0, forecast_p1=0,
                           total_dropped_to_p1=0, total_dropped_to_p2=0)
    final = SimpleNamespace(finalized_count_p1=0, finalized_count_p2=0,
                            chain_total_score_p1=0, chain_total_score_p2=0)
    overlay.update(result(0), snap, final, 0, 1)
    r = result(.1)
    r.p1.state = r.p2.state = BoardState[state]
    r.p1.confirmed_board._grid[:] = 9
    overlay.update(r, snap, final, .1, 1)
    overlay.calculate()
    assert not np.any(m0.call_args.args[0])


def test_boundary_clears_unsent_static_input() -> None:
    from src.board_state_machine import BoardState
    m0 = Mock(return_value=.5)
    overlay = SplitExchangeOverlay(Models(), build_static, Signals, m0)
    snap = SimpleNamespace(net_balance_capped=0, forecast_p1=0,
                           total_dropped_to_p1=0, total_dropped_to_p2=0)
    final = SimpleNamespace(finalized_count_p1=0, finalized_count_p2=0,
                            chain_total_score_p1=0, chain_total_score_p2=0)
    overlay.update(result(0), snap, final, 0, 1)
    r = result(.1)
    r.p1.state = r.p2.state = BoardState.TSUMO_FALL
    overlay.update(r, snap, final, .1, 2)
    overlay.calculate()
    assert not m0.called
    assert overlay.tracker.probability is None


def test_failed_deferred_model_never_publishes_placeholder() -> None:
    model = Models()
    model.predict_source_probability = Mock(side_effect=ValueError('欠測'))
    tracker = DeferredTracker(model)
    tracker._evaluate(static(), 'G_fe', 0)
    tracker.calculate()
    assert tracker.probability is None
    assert tracker.source == 'waiting_confirmed'


@pytest.mark.parametrize('idle', [False, True])
def test_measurement_cpu_once_or_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, idle: bool) -> None:
    import scripts.measure_live_b7 as measurement
    monkeypatch.setattr('sys.argv', ['measure', '--output', str(tmp_path), '--commit', 'test-commit'])
    monkeypatch.setattr(measurement.os, 'nice', lambda value: 0)
    monkeypatch.setattr(measurement, 'wait_idle', lambda *a: idle)
    run = Mock(return_value={'state': 'failed', 'returncode': 1})
    monkeypatch.setattr(measurement, 'run_condition', run)
    measurement.main()
    assert run.call_count == int(idle)
    if idle:
        assert run.call_args.args[0] == 'cpu'


def test_evaluation_driven_publisher_does_not_repeat_at_close() -> None:
    import time
    from tests.phase_j.test_live_publish import ASSETS, row
    from src.phase_j.live_publish import LivePublisher
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0, evaluation_driven=True)
    publisher.start()
    try:
        publisher.offer(row())
        deadline = time.monotonic()+2
        while not publisher.publications and time.monotonic() < deadline:
            time.sleep(.01)
    finally:
        publisher.close()
    assert len(publisher.publications) == 1


@pytest.mark.parametrize('stride', [1, 15, 30])
def test_state_transitions_equal_offline(stride: int) -> None:
    from src.exchange_event_overlay import ExchangeEventOverlay
    from scripts.compare_live_b7 import physical_records
    pair = [cls(Models(), build_static, Signals, lambda b, q: .5, per_side_settled=True)
            for cls in (ExchangeEventOverlay, SplitExchangeOverlay)]
    for frame in range(120):
        t = frame/30
        final = SimpleNamespace(finalized_count_p1=int(t >= 2.6), finalized_count_p2=0,
            chain_total_score_p1=700 if t >= 2.6 else 0, chain_total_score_p2=0)
        snap = SimpleNamespace(net_balance_capped=0, forecast_p1=0,
                               total_dropped_to_p1=0, total_dropped_to_p2=0)
        for overlay in pair:
            overlay.update(result(t), snap, final, t, 1)
        if frame % stride == 0:
            pair[1].calculate()
    assert physical_records(pair[0]) == physical_records(pair[1])
