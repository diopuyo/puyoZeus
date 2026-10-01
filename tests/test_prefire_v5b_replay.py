"""採点前の遅れ固定と、少数標本であることの明示。"""
import json
from pathlib import Path

import pytest

from scripts import prefire_v5b_replay as replay


def test_latency_uses_notification_p95_and_rounds_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replay, 'OUT', tmp_path)
    directory = tmp_path/'notifications'
    directory.mkdir()
    for index, cost in zip(replay.TIMING_INDICES, (50., 100., 150., 200., 300.)):
        (directory/f'benchmark_{index:03d}.json').write_text(json.dumps(dict(milliseconds=cost)))
    result = replay.freeze_latency()
    assert result['notifications'] == 5
    assert result['p95_ms'] == pytest.approx(280.)
    assert result['latency_sec'] == .3
    assert result['within_200ms'] is False
    assert result['limitation']
    assert replay.freeze_latency() == result


def test_missing_benchmark_does_not_freeze_latency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replay, 'OUT', tmp_path)
    with pytest.raises(FileNotFoundError):
        replay.freeze_latency()
    assert not (tmp_path/'latency.json').exists()
