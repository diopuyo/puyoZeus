"""設置の判断を変えず、監査の順序・投入時の値・全行を保存する。"""
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.phase_j.live_process import configure_placement_log
from src.placement_signal_runtime import PlacementSignalRuntime


def test_placement_audit_equal_after_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = []
    for enabled in ('0', '1'):
        monkeypatch.setenv('PUYO_OBS_FAST_AUDIT', enabled)
        recorder = PlacementSignalRuntime.__new__(PlacementSignalRuntime)
        recorder.audit, recorder.sink = [], None
        path = tmp_path/enabled/'placement.jsonl'
        with ExitStack() as storage:
            configure_placement_log(storage, SimpleNamespace(_placement_reconcile=recorder), str(path))
            for index in range(8):
                row = dict(frame=index, reason='採否の原票', cells=[index, None])
                recorder.record(row)
                row['cells'][0] = -1
        outputs.append(path.read_bytes())
        rows = [json.loads(line) for line in outputs[-1].splitlines()]
        assert [row['cells'][0] for row in rows] == list(range(8))
        assert recorder.audit == []
    assert outputs[0] == outputs[1]
