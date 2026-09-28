"""完走ログを再走・消失させずに後段だけを再開する。"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import measure_live_b16 as measure
from tests.test_live_b16 import remote, inputs
from src.phase_j.live_eval_supervisor import EvaluationError, SupervisedOverlay


@pytest.fixture
def completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    measure.save(tmp_path/'launch.json', dict(source_hashes={'source': 'same'}))
    measure.save(tmp_path/'report.json', dict(reached_end=True, evaluation_error_count=0))
    measure.save(tmp_path/'realtime/metrics.json', {})
    monkeypatch.setattr(measure, 'source_hashes', lambda: {'source': 'same'})
    monkeypatch.setattr(measure, 'realtime', Mock(side_effect=AssertionError('再走禁止')))
    monkeypatch.setattr(measure, 'wait_idle', Mock(return_value=True))
    return tmp_path


def test_resume_merges_quality_without_losing_runtime(completed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference = Mock()
    monkeypatch.setattr(measure, 'reference', reference)
    monkeypatch.setattr(measure, 'quality_report', lambda root: {'sampled_seconds': 1200})
    measure.resume_reference(completed)
    reference.assert_called_once_with(completed)
    report = json.loads((completed/'report.json').read_text())
    assert report == dict(reached_end=True, evaluation_error_count=0, quality={'sampled_seconds': 1200})


@pytest.mark.parametrize('changed_during_wait', [False, True])
def test_resume_rejects_changed_recognition(completed: Path, monkeypatch: pytest.MonkeyPatch,
                                           changed_during_wait: bool) -> None:
    hashes = [{'source': 'same'}, {'source': 'changed'}] if changed_during_wait else [{'source': 'changed'}]
    monkeypatch.setattr(measure, 'source_hashes', Mock(side_effect=hashes))
    reference = Mock()
    monkeypatch.setattr(measure, 'reference', reference)
    with pytest.raises(ValueError):
        measure.resume_reference(completed)
    reference.assert_not_called()


def test_resume_does_not_start_without_idle(completed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(measure, 'wait_idle', Mock(return_value=False))
    reference = Mock()
    monkeypatch.setattr(measure, 'reference', reference)
    measure.resume_reference(completed)
    reference.assert_not_called()


def test_resume_requires_completed_realtime(completed: Path) -> None:
    measure.save(completed/'report.json', dict(reached_end=False))
    with pytest.raises(ValueError, match='完走'):
        measure.resume_reference(completed)


def test_updates_finish_before_publication(remote: SupervisedOverlay, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = remote.request

    def request(row: dict) -> dict:
        calls.append((row['op'], len(row['commands'])))
        return original(row)

    monkeypatch.setattr(remote, 'request', request)
    for tick in range(15):
        remote.update(*inputs(tick/30))
        assert not remote.pending
    assert remote.tracker.probability is None
    remote.calculate()
    assert calls[:-1] == [('advance', 1)]*15
    assert calls[-1] == ('batch', 0)
    assert remote.tracker.probability is not None


def test_update_fault_isolated_and_next_match_recovers(remote: SupervisedOverlay) -> None:
    malformed = list(inputs(0))
    malformed[0] = None
    with pytest.raises(EvaluationError):
        remote.update(*malformed)
    assert remote.error_count == 1 and remote.dirty
    remote.update(*inputs(.5, game=2))
    remote.calculate()
    assert remote.tracker.probability is not None
    assert not remote.dirty and remote.failures == 0


def test_journal_retains_only_current_match(remote: SupervisedOverlay) -> None:
    for tick in range(10):
        remote.update(*inputs(tick/30))
    remote.calculate()
    assert len(remote.journal) == 11
    remote.update(*inputs(1, game=2))
    assert len(remote.journal) == 1
    assert not remote.pending


def test_save_keeps_boundary_records_before_next_publication(remote: SupervisedOverlay,
                                                            tmp_path: Path) -> None:
    from src.phase_j.live_evaluation import SplitExchangeOverlay
    from tests.test_exchange_event_tracker import Models
    from tests.test_exchange_event_overlay import build_static, Signals
    from tests.test_live_b16 import m0
    direct = SplitExchangeOverlay(Models(), build_static, Signals, m0, per_side_settled=True)
    for tick in range(120):
        args = inputs(tick/30, game=1 if tick < 100 else 2)
        for overlay in (direct, remote):
            overlay.update(*args)
            if tick < 100 and tick % 15 == 0:
                overlay.calculate()
    direct.tracker.save(tmp_path/'direct.jsonl')
    remote.save(tmp_path/'remote.jsonl')
    assert (tmp_path/'remote.jsonl').read_text() == (tmp_path/'direct.jsonl').read_text()
    assert (tmp_path/'remote.jsonl').read_text().strip()
