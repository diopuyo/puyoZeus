"""v4.9fの異常を終了コード0で隠さず、試験の原票へ残す。"""
import json
from pathlib import Path
from queue import Queue

import pytest

from src.phase_j.live_v49f import LiveV49F


@pytest.mark.parametrize('enabled', [False, True])
def test_worker_failure_keeps_reason_and_trial_evidence(
        monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setenv('PUYO_OBS_FAST_AUDIT', str(int(enabled)))
    captured = {}
    monkeypatch.setattr(Path, 'write_text',
        lambda path, text, **kwargs: captured.update({path.name: json.loads(text)}))
    owner = object.__new__(LiveV49F)
    owner.output = Path('unused')
    owner.outputs = Queue()
    owner.failure = None
    fatal = dict(fatal='KeyError: 0', stack='保存対象のworker stack', generation=1)
    owner.outputs.put(fatal)
    owner.receive()
    assert owner.failure == 'KeyError: 0'
    assert captured == ({'v49f_failure.json': fatal} if enabled else {})
