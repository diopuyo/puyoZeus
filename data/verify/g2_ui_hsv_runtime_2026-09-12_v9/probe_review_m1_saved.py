"""原M1要求票検査を再利用し、短い人工区間と実whole終端を混同しない。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parent / 'video38_ui_hsv_candidate_v9'
FIRST, LAST, STRIDE = 29052, 36298, 2
FRAMES = (35370, 35410)
SIDES = ('1P', '2P')


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('whole_M1_saved:' + reason)


def capture_scope(capture: dict, output: Path, last: int) -> None:
    require(capture['first_frame'] == FIRST, 'producer_first')
    require(capture['last_frame'] == last, 'producer_last')
    require(capture['observed_count'] == (last - FIRST) // STRIDE + 1, 'producer_count')
    require(capture['identity']['run_id'] == str(output), 'producer_run')


def metadata_scope(rows: Any) -> int:
    expected = ((frame, side) for frame in range(FIRST, LAST + STRIDE, STRIDE) for side in SIDES)
    count = 0
    for row, pair in zip(rows, expected, strict=True):
        require((row['frame_idx'], row['side']) == pair, 'metadata_order')
        count += 1
    return count


def inspect(output: Path) -> dict:
    sys.path.insert(0, str(ROOT.parent / 'g2_joint_collector_runtime_2026-09-12_v1'))
    import verify_joint_saved as original
    waited = original.read(output / 'TARGET_PARENT_WAIT.json')
    require(waited['child_exit_code'] == 0 and waited['resource_guard_exit'] == 0, 'actual_wait')
    require(original.read(output / 'ENTRY_RESULT.json')['exit_code'] == 0, 'entry_exit')
    names = tuple(f'BELIEF_M1_{frame}.json' for frame in FRAMES)
    packets = original.packets(output, 'JOINT_EVENTS.jsonl', names, FRAMES)
    session = original.read(output / 'BELIEF_M1_SESSION.json')
    original.lifecycle(session, original.read(output / 'BELIEF_M1_COLLECTOR_ATTACH.json'),
                       original.read(output / 'JOINT_RUNTIME_SOURCE.json'))
    for index, (name, frame, saved) in enumerate(zip(names, FRAMES, session['saved'], strict=True)):
        require(saved['path'] == str(output / name), 'session_path')
        require(saved['sha256'] == hashlib.sha256((output / name).read_bytes()).hexdigest(), 'session_sha')
        ticket = original.read(output / f'JOINT_EVENTS.jsonl.request{index}.json')
        capture_scope(ticket['producer'], output, frame)
    capture_scope(original.read(output / 'JOINT_PRODUCER_CAPTURE.json'), output, LAST)
    with (output / 'collector_metadata.jsonl').open() as stream:
        count = metadata_scope(json.loads(line) for line in stream)
    return dict(frames=FRAMES, metadata_sides=count, original_packet_checks=True,
                supported=[packet['result']['supported'] for packet in packets],
                terminal_producer_separate=True, quality_gate_clear=False, production_permission=False)


if __name__ == '__main__':
    result = inspect(OUTPUT)
    with (ROOT / 'M1_SAVED_REVIEW_v1.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)
