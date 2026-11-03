"""段別時間は通知を変えず、所有スレッド・役割ごとにspoolへ記録する。"""
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.phase_j import live_stage_timing as timing


def test_off_has_no_file_or_clock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv('PUYO_LIVE_STAGE_TIMING', raising=False)
    with ExitStack() as stack:
        timing.install(stack, tmp_path, 'recognition')
        timing.begin('recognition', 1.)
        assert timing.measure('capture', lambda: 42) == 42
        timing.finish()
    assert list(tmp_path.iterdir()) == []


def test_per_frame_record_and_no_payload_change(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('PUYO_LIVE_STAGE_TIMING', '1')
    payload: dict[str, Any] = dict(value=42)
    with ExitStack() as stack:
        timing.install(stack, tmp_path, 'recognition')
        timing.begin('recognition', 1.)
        timing.identify(SimpleNamespace(index=3, media_sec=.5, captured_at=1.), 7)
        assert timing.measure('capture', lambda: payload) is payload
        timing.detail(screen_result=dict(captured_at=.8, duration_sec=.08))
        timing.finish(held=False)
        timing.begin('recognition', 2.)
        timing.finish(held=True)
    rows = [json.loads(line) for line in (tmp_path/'frame_timing_recognition.jsonl').read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]['frame'] == 3 and rows[0]['epoch'] == 7
    assert rows[0]['stages']['capture']['calls'] == 1
    assert rows[1]['stages']['capture']['calls'] == 0
    assert rows[0]['stages']['capture']['wall_ms'] >= 0
    assert payload == dict(value=42)
    assert timing.CURRENT.get() is None


def test_exception_and_publication_restore_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('PUYO_LIVE_STAGE_TIMING', '1')
    @timing.publication
    def publish(owner: Any, row: dict) -> int:
        return row['frame']
    with ExitStack() as stack:
        for role in ('evaluation', 'publication'):
            timing.install(stack, tmp_path, role)
        timing.begin('evaluation', 1., 4, .2)
        previous = timing.CURRENT.get()
        assert publish(None, dict(frame=4, captured_at=1., t_sec=.2)) == 4
        assert timing.CURRENT.get() is previous
        with pytest.raises(ZeroDivisionError):
            timing.measure('evaluation', lambda: 1/0)
        timing.finish()
    assert all((tmp_path/f'frame_timing_{role}.jsonl').stat().st_size for role in ('evaluation', 'publication'))


def test_new_flags_default_off_and_delivered(tmp_path: Path) -> None:
    from src.phase_j.launcher import parse_user_config, build_pipeline_config
    from src.phase_j.live_speed import validate
    flags = ('b3_share_score_read', 'live_async_screen_verify', 'live_stage_timing',
             'live_reference_times_only', 'live_border_grace')
    assert all(validate({})[key] is False for key in flags)
    user = parse_user_config(dict(device_name='試験', **dict.fromkeys(flags, True)), tmp_path)
    assert all(build_pipeline_config(user)[key] is True for key in flags)


def test_event_worker_records_reference_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """親CPUに含まれない参照処理を子の同一通知へ結び付ける。"""
    from src.phase_j.live_eval_worker import execute
    monkeypatch.setenv('PUYO_LIVE_STAGE_TIMING', '1')
    monkeypatch.setenv('PUYO_B3_JUDGMENT', '1')
    tracker = SimpleNamespace(records=[], current=None, source='waiting', probability=None,
                              sealed=[], sealed_diagnostics=[])
    overlay = SimpleNamespace(tracker=tracker, smoothing_state=lambda: (0., .5, None))
    with ExitStack() as stack:
        timing.install(stack, tmp_path, 'event_evaluation')
        timing.begin('event_evaluation', 1., 12, .4)
        reply = execute(overlay, dict(commands=[], op='advance'))
        timing.finish(request_id=7)
    row = json.loads((tmp_path/'frame_timing_event_evaluation.jsonl').read_text())
    assert row['frame'] == 12 and row['request_id'] == 7
    assert row['stages']['event_reference']['calls'] == 1
    assert reply['b3_reference'] == dict(source='waiting', probability=None, physical=[])
