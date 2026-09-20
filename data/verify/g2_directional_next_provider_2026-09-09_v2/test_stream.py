"""初期静穏とscope境界の人工試験。live callerの認証ではない。"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

import dependencies as D
from stream import Stream

C = D.load('candidate')
SOURCE = 'a' * 64
STARTS = ((480, 120), (480, 175), (515, 205), (515, 245))


def dto(**values: Any) -> Any:
    return SimpleNamespace(**values)


def invocation(frame: int) -> Any:
    return SimpleNamespace(frame=frame, time_sec=frame / C.FPS)


def make() -> Stream:
    value = Stream(C, SimpleNamespace(MotionObservation=dto, MotionCandidate=dto), SOURCE, '1P')
    value.start(invocation(100), 0)
    return value


def packet(stream: Stream, before: int, moving: bool = False, empty: bool = False) -> dict[str, Any]:
    slots = {}
    for index, (name, roi) in enumerate(zip(C.SLOTS, C.ROIS['1P'])):
        x, y = STARTS[index]
        dy = -3.0 if moving and index in (0, 3) else 0.0
        points = [] if empty else [{'index': i, 'start': [float(x + i), float(y)],
            'end': [float(x + i), float(y) + dy], 'back': [float(x + i), float(y)],
            'forward_status': 1, 'backward_status': 1, 'forward_in_image': True,
            'valid_both_status_in_image': True, 'dx': 0.0, 'dy': dy, 'fb_distance': 0.0}
            for i in range(10)]
        slots[name] = {'roi': list(roi), 'seed_count': len(points), 'points': points}
    return {'schema': C.SCHEMA, 'source_sha256': SOURCE, 'sequence_id': stream.sequence(),
            'side': '1P', 'frame_before': before, 'frame_after': before + 2,
            'time_before': before / C.FPS, 'time_after': (before + 2) / C.FPS, 'slots': slots}


def test_no_initial_magic_arm_then_one_candidate() -> None:
    stream = make()
    assert stream.latest.quiet_frames is None and stream.engine is None
    assert stream.update(invocation(102), packet(stream, 100, moving=True)).candidate is None
    assert stream.update(invocation(104), packet(stream, 102)).quiet_frames is None
    assert stream.update(invocation(106), packet(stream, 104)).quiet_frames == (104, 106)
    value = stream.update(invocation(108), packet(stream, 106, moving=True))
    assert value.candidate.sequence_number == 1
    assert value.candidate.first_support_frame == value.candidate.available_frame == 108
    assert value.invocation.frame == 108
    assert stream.update(invocation(110), packet(stream, 108, moving=True)).candidate is None


def test_empty_is_not_quiet_and_invalidate_forgets_support() -> None:
    stream = make()
    for before in (100, 102):
        value = stream.update(invocation(before + 2), packet(stream, before, empty=True))
        assert value.quiet_frames is None and stream.engine is None
    old_sequence = stream.sequence()
    stream.invalidate()
    assert stream.latest is None and stream.last_frame is None
    stream.start(invocation(200), 1)
    assert stream.sequence() != old_sequence
    assert stream.update(invocation(202), packet(stream, 200, moving=True)).candidate is None


@pytest.mark.parametrize('bad', ['scope', 'gap', 'clock', 'side'])
def test_invalid_packet_changes_no_stream_state(bad: str) -> None:
    stream = make()
    value = packet(stream, 100)
    if bad == 'scope':
        value['sequence_id'] = 'old'
    elif bad == 'gap':
        value['frame_before'], value['time_before'] = 98, 98 / C.FPS
    elif bad == 'clock':
        value['time_after'] += 0.5
    else:
        value['side'] = '2P'
    before = (stream.last_frame, stream.epoch, stream.segment, stream.quiet_run, stream.latest)
    with pytest.raises(ValueError):
        stream.update(invocation(102), value)
    assert before == (stream.last_frame, stream.epoch, stream.segment, stream.quiet_run, stream.latest)


def test_future_packet_not_needed() -> None:
    streams = [make(), make()]
    outputs = []
    for stream in streams:
        rows = []
        for before in (100, 102, 104):
            value = stream.update(invocation(before + 2), packet(stream, before, moving=before == 104))
            rows.append((value.frame_idx, value.quiet_frames, vars(value.candidate) if value.candidate else None))
        outputs.append(deepcopy(rows))
    assert outputs[0] == outputs[1]
