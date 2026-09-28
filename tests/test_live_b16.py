"""B15のS3欠測と、評価例外からの配信復帰を故障注入で検証する。"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import Mock

import pytest

from src.phase_j.live_evaluation import DeferredTracker, SplitExchangeOverlay
from src.phase_j.live_eval_supervisor import SupervisedOverlay, EvaluationError, FAILURE_LIMIT
from tests.test_exchange_event_tracker import Models, fire
from tests.test_exchange_event_overlay import build_static, result, Signals


def inputs(t: float, game: int = 1) -> tuple:
    final = SimpleNamespace(finalized_count_p1=int(t >= 2.6), finalized_count_p2=0,
        chain_total_score_p1=700 if t >= 2.6 else 0, chain_total_score_p2=0)
    snap = SimpleNamespace(net_balance_capped=0, forecast_p1=0,
                           total_dropped_to_p1=0, total_dropped_to_p2=0)
    return result(t), snap, final, t, game


def m0(boards: object, queue: object) -> float:
    return .5


def test_closed_deferred_s3_is_restored_on_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.exchange_event_tracker import ExchangeEventTracker
    tracker = DeferredTracker(Models())
    tracker.boundary(1, 0)
    fire(tracker)
    tracker.end('1P', 3, 'next')
    tracker.finalize('1P', 3, 700)
    tracker.confirm_frame_inputs(4)
    tracker.finish_frame(4)
    record = tracker.current
    assert record.values[-1]['p1'] is None
    tracker.close_confirmed(5, (5, 5))
    assert tracker.pending is None

    def resume(self: object, observations: tuple) -> bool:
        self.current = record
        return True

    monkeypatch.setattr(ExchangeEventTracker, '_resume_existing', resume)
    tracker._resume_existing(())
    tracker.calculate()
    assert record.values[-1]['p1'] == .8
    assert record.values[-1]['deferred'] is False
    tracker.boundary(2, 6)
    assert not tracker._deferred_records


@pytest.mark.parametrize('source', ['S3', 'S3_provisional'])
def test_uncomputed_projection_holds_last_value(source: str) -> None:
    from src.exchange_event_landing import ExchangeLandingProjection
    projection = ExchangeLandingProjection()
    projection._observe_frame = Mock()
    projection._refresh_death = Mock(return_value=False)
    projection._evaluate = Mock(side_effect=AssertionError('未計算S3の数値合成'))
    tracker = SimpleNamespace(current=SimpleNamespace(values=[dict(source=source, p1=None)]),
                              probability=.73)
    overlay = SimpleNamespace(tracker=tracker, _m0=object(), _history=[[1], [1]])
    projection.update(overlay, None, SimpleNamespace(total_dropped_to_p1=0, total_dropped_to_p2=0), 1)
    assert tracker.probability == .73


def test_full_frame_success_acknowledges_failure_count() -> None:
    from src.phase_j.live_bridge import RecognitionBridge
    bridge = RecognitionBridge(False, Mock())
    bridge.event_evaluator = SimpleNamespace(succeeded=Mock())
    notice = SimpleNamespace(frame=1)
    bridge.observe(notice, .6, 20, None, None, 1, 0)
    bridge.event_evaluator.succeeded.assert_called_once()
    bridge.callback = Mock(side_effect=RuntimeError('公開前の故障'))
    with pytest.raises(RuntimeError):
        bridge.observe(notice, .6, 20, None, None, 1, 0)
    assert bridge.event_evaluator.succeeded.call_count == 1


@pytest.fixture
def remote(tmp_path: Path) -> Iterator[SupervisedOverlay]:
    overlay = SupervisedOverlay(Models(), build_static, Signals, m0,
                                 per_side_settled=True, directory=tmp_path)
    yield overlay
    overlay.close()
    assert not overlay.process.is_alive()


def test_remote_outputs_equal_direct_and_boundary_archive(remote: SupervisedOverlay, tmp_path: Path) -> None:
    direct = SplitExchangeOverlay(Models(), build_static, Signals, m0, per_side_settled=True)
    for frame in range(120):
        args = inputs(frame/30, 1 if frame < 100 else 2)
        for overlay in (direct, remote):
            overlay.update(*args)
        if frame % 15 == 0:
            for overlay in (direct, remote):
                overlay.calculate()
            assert remote.tracker.probability == direct.tracker.probability
            assert remote.tracker.source == direct.tracker.source
    target = tmp_path/'events.jsonl'
    remote.save(target)
    direct.tracker.save(tmp_path/'direct.jsonl')
    assert target.read_text() == (tmp_path/'direct.jsonl').read_text()


def test_single_failure_recovers_next_update(remote: SupervisedOverlay, tmp_path: Path) -> None:
    remote.update(*inputs(0))
    remote.calculate()
    previous = remote.process.pid
    remote.update(*inputs(.5))
    remote.fault = 'single'
    with pytest.raises(EvaluationError):
        remote.calculate()
    remote.update(*inputs(1))
    remote.calculate()
    assert remote.process.pid == previous
    assert remote.tracker.probability == .6
    error = json.loads((tmp_path/'evaluation_errors.jsonl').read_text())
    assert error['t_sec'] == .5 and error['exception_type'] == 'RuntimeError'
    assert 'Traceback' in error['stack'] and error['utc']


def test_consecutive_failure_restarts_and_replays_match(remote: SupervisedOverlay, tmp_path: Path) -> None:
    direct = SplitExchangeOverlay(Models(), build_static, Signals, m0, per_side_settled=True)
    previous = remote.process.pid
    hold_pids = []
    remote.on_error = lambda: hold_pids.append(remote.process.pid)
    for index in range(FAILURE_LIMIT):
        args = inputs(index/2)
        remote.update(*args)
        direct.update(*args)
        remote.fault = 'consecutive'
        with pytest.raises(EvaluationError):
            remote.calculate()
    assert remote.process.pid != previous and remote.restarts == 1
    assert hold_pids == [previous]*FAILURE_LIMIT
    for overlay in (direct, remote):
        overlay.update(*inputs(1.5))
        overlay.calculate()
    assert remote.tracker.probability == direct.tracker.probability
    events = [json.loads(line) for line in (tmp_path/'evaluation_errors.jsonl').read_text().splitlines()]
    assert [r['kind'] for r in events].count('evaluation_error') == FAILURE_LIMIT
    assert events[-1]['replay_from'] == 'match_boundary'


def test_dead_worker_is_restarted(remote: SupervisedOverlay) -> None:
    previous = remote.process.pid
    remote.process.terminate()
    remote.process.join()
    remote.update(*inputs(0))
    with pytest.raises(EvaluationError):
        remote.calculate()
    assert remote.process.pid != previous
    remote.update(*inputs(.5))
    remote.calculate()
    assert remote.tracker.probability == .4


@pytest.mark.parametrize('hold_started', [None, 1.3])
def test_error_hold_dto_and_next_success(hold_started: float | None) -> None:
    from tests.phase_j.test_live_publish import ASSETS, row
    from src.phase_j.live_publish import initial_snapshot, result_snapshot
    from src.phase_j.validator import validate_snapshot
    initial = initial_snapshot(ASSETS)
    snapshot = result_snapshot(initial, dict(row(), evaluation_error=True), 1.5, 1, hold_started)
    assert validate_snapshot(snapshot).is_valid
    assert snapshot.display['status'] == 'hold'
    assert snapshot.display['message'] == '評価エラー'
    assert snapshot.evaluations['practical']['p1_win_probability'] is None
    recovered = result_snapshot(initial, row(), 2, 2, None)
    assert recovered.display['status'] == 'live'
    assert recovered.evaluations['practical']['p1_win_probability'] == .6


def test_guard_keeps_consuming_recognition_after_fault() -> None:
    from src.phase_j.live_eval_supervisor import guard_loop
    tree = ast.parse('for packet in notices:\n fi=packet\n t=packet\n r=packet\n pipe=None\n evaluate(packet)')
    guard_loop(tree.body[0])
    consumed, held = [], []

    def evaluate(packet: int) -> None:
        consumed.append(packet)
        if packet == 1:
            raise TypeError('注入')

    namespace = dict(notices=range(4), evaluate=evaluate, _live_bridge=object(),
                     _evaluation_failure=lambda bridge, packet, local, error: held.append(packet))
    exec(compile(ast.fix_missing_locations(tree), '<fault>', 'exec'), namespace)
    assert consumed == [0, 1, 2, 3] and held == [1]


def test_late_publish_cannot_overwrite_error_hold() -> None:
    from tests.phase_j.test_live_publish import ASSETS, row
    from src.phase_j.live_publish import LivePublisher
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0)
    publisher.evaluation_hold(SimpleNamespace(frame=2, t_sec=2, captured_at=1, recognized_at=1.1), 1)
    publisher._publish(row(), 1, None)
    assert publisher.hub.latest.display['message'] == '評価エラー'
    publisher.offer(dict(row(), frame=3))
    publisher._publish(publisher.latest, 2, None)
    assert publisher.hub.latest.display['status'] == 'live'
