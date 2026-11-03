"""B19の境界再構築・遅延応答・親I/O障害・借用画像を故障注入で固定する。"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from src.phase_j.live_eval_supervisor import EvaluationError, SupervisedOverlay, evaluation_failure
from src.phase_j.live_notification_eval import NotificationExchangeOverlay
from tests.test_live_b16 import inputs, m0, remote
from tests.test_exchange_event_tracker import Models
from tests.test_exchange_event_overlay import Signals, build_static

GAMES, FRAMES, FPS = 3, 90, 30


def game_inputs(frame: int, game: int) -> tuple:
    args = list(inputs(frame/FPS, game))
    offset = game*FRAMES/FPS
    args[3] += offset
    if args[0].p1.chain_event is not None:
        args[0].p1.chain_event.trigger_sec += offset
        args[0].p1.chain_event.end_sec += offset
    return tuple(args)


@pytest.mark.parametrize('stage', ['advance', 'batch', 'restart', 'save'])
@pytest.mark.parametrize('boundary', [1, 2])
def test_boundary_reconstruction_preserves_every_event_and_id(
        remote: SupervisedOverlay, tmp_path: Path, stage: str, boundary: int,
        monkeypatch: pytest.MonkeyPatch) -> None:
    direct = NotificationExchangeOverlay(Models(), build_static, Signals, m0, per_side_settled=True)
    original = remote.request
    def faulty(request: dict) -> dict:
        if request['op'] == 'advance':
            monkeypatch.setattr(remote, 'request', original)
            request = dict(request, fault='境界update後に故障')
        return original(request)
    for game in range(GAMES):
        for frame in range(FRAMES):
            args = game_inputs(frame, game)
            direct.update(*args)
            inject = game == boundary and frame == 0
            if inject and stage == 'advance':
                monkeypatch.setattr(remote, 'request', faulty)
                with pytest.raises(EvaluationError):
                    remote.update(*args)
            else:
                remote.update(*args)
            if inject and stage in ('batch', 'restart', 'save'):
                for _ in range(3 if stage == 'restart' else 1):
                    remote.fault = '境界公開時に故障'
                    with pytest.raises(EvaluationError):
                        remote.calculate()
            if inject and stage == 'save':
                remote.save(tmp_path/'intermediate.jsonl')
            remote.calculate()
            assert remote.tracker.probability == direct.tracker.probability
    direct.tracker.save(tmp_path/'direct.jsonl')
    remote.save(tmp_path/'remote.jsonl')
    assert (tmp_path/'direct.jsonl').read_text() == (tmp_path/'remote.jsonl').read_text()
    rows = [json.loads(line) for line in (tmp_path/'remote.jsonl').read_text().splitlines()]
    assert [row['exchange_id'] for row in rows] == list(range(1, GAMES+1))
    assert remote.restarts == int(stage == 'restart')


@pytest.mark.parametrize('operation', ['advance', 'batch', 'records'])
def test_late_reply_is_discarded_by_request_id(
        remote: SupervisedOverlay, monkeypatch: pytest.MonkeyPatch, operation: str,
        tmp_path: Path) -> None:
    remote.auto_acknowledge = False
    original = remote.receive
    def delayed(**kwargs: Any) -> dict:
        assert remote.connection.poll(10)
        raise TimeoutError('応答到着直後のタイムアウト注入')
    monkeypatch.setattr(remote, 'receive', delayed)
    with pytest.raises(EvaluationError):
        if operation == 'advance':
            remote.update(*inputs(0))
        elif operation == 'batch':
            remote.calculate()
        else:
            remote.save(tmp_path/'early.jsonl')
    monkeypatch.setattr(remote, 'receive', original)
    for stamp in (.5, 1., 1.5):
        remote.update(*inputs(stamp))
        remote.calculate()
    assert remote.restarts == 0 and remote.error_count == 1
    assert remote.failures == 1
    rows = [json.loads(line) for line in (tmp_path/'evaluation_errors.jsonl').read_text().splitlines()]
    assert any(row['kind'] == 'stale_evaluation_reply' for row in rows)


def test_completed_advance_reply_cannot_be_used_as_batch(
        remote: SupervisedOverlay, monkeypatch: pytest.MonkeyPatch) -> None:
    remote.auto_acknowledge = False
    original, receive, poll = remote.receive, remote.connection.recv, remote.connection.poll
    delayed_reply: list[dict] = []
    def delayed(**kwargs: Any) -> dict:
        delayed_reply.append(original(**kwargs))
        raise TimeoutError('完了したadvance応答を遅延')
    monkeypatch.setattr(remote, 'receive', delayed)
    with pytest.raises(EvaluationError):
        remote.update(*inputs(0))
    monkeypatch.setattr(remote, 'receive', original)
    monkeypatch.setattr(remote.connection, 'recv', lambda: delayed_reply.pop() if delayed_reply else receive())
    monkeypatch.setattr(remote.connection, 'poll', lambda timeout: bool(delayed_reply) or poll(timeout))
    remote.calculate()
    assert remote.tracker.probability == .4
    assert remote.restarts == 0 and remote.error_count == 1


@pytest.mark.parametrize('filename,kind', [
    ('live_publish.py', 'publication_error'), ('live_spool.py', 'spool_error'),
    ('another_stage.py', 'processing_error')])
def test_parent_failures_hold_and_log_without_restarting(
        remote: SupervisedOverlay, tmp_path: Path, filename: str, kind: str) -> None:
    bridge = SimpleNamespace(event_evaluator=remote, evaluation_hold=Mock())
    remote.on_error = Mock()
    notice = SimpleNamespace(t_sec=1.)
    pid = remote.process.pid
    for _ in range(4):
        try:
            exec(compile('raise OSError("故障注入")', filename, 'exec'))
        except OSError as error:
            evaluation_failure(bridge, notice, {'game_idx': 2}, error)
    assert remote.failures == remote.error_count == remote.restarts == 0
    assert remote.process.pid == pid
    assert bridge.evaluation_hold.call_count == 4
    assert remote.on_error.call_count == 0
    rows = [json.loads(line) for line in (tmp_path/'evaluation_errors.jsonl').read_text().splitlines()]
    assert all(row['kind'] == kind and 'Traceback' in row['stack'] for row in rows)


def test_spool_failure_rolls_back_archive_and_replays(
        remote: SupervisedOverlay, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for frame in range(FRAMES):
        remote.update(*game_inputs(frame, 0))
        remote.calculate()
    original = remote.diagnostics.extend
    def broken(rows: Any) -> None:
        raise OSError('交換回収後の診断spool故障')
    monkeypatch.setattr(remote.diagnostics, 'extend', broken)
    with pytest.raises(OSError):
        remote.update(*game_inputs(0, 1))
    assert not len(remote.archive) and remote.dirty
    assert remote.failures == 0 and remote.restarts == 0
    monkeypatch.setattr(remote.diagnostics, 'extend', original)
    remote.update(*game_inputs(1, 1))
    remote.save(tmp_path/'recovered.jsonl')
    rows = [json.loads(line) for line in (tmp_path/'recovered.jsonl').read_text().splitlines()]
    assert [row['exchange_id'] for row in rows] == [1]


def test_reused_capture_buffer_keeps_previous_roi_owned() -> None:
    from src.image_reader import DEFAULT_P1_REGION
    from src.phase_j.live_source import VideoFileSource
    from src.phase_j.live_snapshot_quality import SnapshotQuality
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    class Capture:
        def read(self) -> tuple:
            frame[:] = (255, 0, 0) if frame[0, 0, 2] else (0, 0, 255)
            return True, frame
    quality = SnapshotQuality()
    packets = iter(VideoFileSource(Capture(), FPS, 0, 2, 1))
    first = next(packets)
    assert not quality.observe(first.image, DEFAULT_P1_REGION)[0]
    before = quality.previous.copy()
    second = next(packets)
    assert first.image is second.image
    np.testing.assert_array_equal(quality.previous, before)
    assert quality.observe(second.image, DEFAULT_P1_REGION)[0]
    assert not np.shares_memory(frame, quality.previous)


def test_frozen_setup_failure_restores_cwd_modules_and_path(tmp_path: Path) -> None:
    import os
    from tests.conftest import isolated_frozen_imports
    import src
    saved, paths, directory = dict(sys.modules), list(sys.path), Path.cwd()
    with pytest.raises(RuntimeError):
        with isolated_frozen_imports():
            os.chdir(tmp_path)
            sys.path.insert(0, str(tmp_path))
            sys.modules['src'] = SimpleNamespace(injected=True)
            sys.modules['_video38_frozen_collector'] = SimpleNamespace(injected=True)
            raise RuntimeError('loader setupの途中故障')
    assert Path.cwd() == directory and sys.path == paths
    assert sys.modules['src'] is src
    assert sys.modules.get('_video38_frozen_collector') is saved.get('_video38_frozen_collector')


def test_new_functions_stay_within_limit() -> None:
    paths = ('src/phase_j/live_eval_supervisor.py', 'src/phase_j/live_eval_worker.py',
             'src/phase_j/live_eval_recovery.py', 'tests/test_live_b19.py', 'tests/conftest.py')
    assert not [(path, node.name) for path in paths
        for node in ast.walk(ast.parse(Path(path).read_text(encoding='utf-8')))
        if isinstance(node, ast.FunctionDef) and node.end_lineno-node.lineno+1 > 50]
