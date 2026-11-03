"""圧縮原票の保存を分離しても、全JSON bytesと完了マーカーを保持する。"""
import gzip
import json
from pathlib import Path

import pytest

from src.exchange_event_record import ExchangeEventRecorder


def test_compressed_record_keeps_all_logical_bytes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = []
    for enabled in ('0', '1'):
        monkeypatch.setenv('PUYO_OBS_FAST_AUDIT', enabled)
        path = tmp_path/f'record-{enabled}.jsonl.gz'
        recorder = ExchangeEventRecorder(path, 'same-video', True, Path('same-model'))
        for index in range(300):
            row = dict(kind='audit', index=index, values=[index/301, None, '観測値'])
            recorder.write(row)
            row['values'][0] = -1
        recorder.close()
        with gzip.open(path, 'rb') as stream:
            outputs.append(stream.read())
        rows = [json.loads(line) for line in outputs[-1].splitlines()]
        assert len(rows) == 302 and rows[-1]['kind'] == 'complete'
        assert rows[1]['values'][0] == 0
    assert outputs[0] == outputs[1]
