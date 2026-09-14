"""保存後の採録順序/終端/実ファイルを独立再計算。資格自体の証明とは分離する。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EARLIEST = (35370, 35410)
END, STRIDE, GAP = 36298, 2, 40


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('saved_schedule:' + reason)


def order(rows: list[dict]) -> tuple[int, ...]:
    require(bool(rows) and type(rows[0]['frame']) is int and rows[0]['frame'] < EARLIEST[0], 'prefix')
    accepted: list[int] = []
    previous = rows[0]['frame'] - STRIDE
    for row in rows:
        frame, reasons = row['frame'], row['reasons']
        require(type(frame) is int and frame == previous + STRIDE and 0 <= frame <= END and frame % STRIDE == 0, 'clock')
        require(type(reasons) is list and all(type(v) is str and v for v in reasons)
                and len(reasons) == len(set(reasons)), 'reasons')
        require(row.get('error') is None and row['quality_gate_clear'] is False, 'failed_or_authorized')
        due = None if len(accepted) == len(EARLIEST) else max(EARLIEST[len(accepted)], accepted[-1] + GAP if accepted else EARLIEST[0])
        expected = 'COMPLETE' if due is None else 'REQUEST' if frame >= due and not reasons else 'WAIT'
        require(row['action'] == expected and row['due'] == due, 'first_eligible_order')
        require(row['original_sentinel'] is (frame in EARLIEST), 'sentinel')
        require(type(row['saved']) is bool and row['saved'] is (expected == 'REQUEST'), 'save_ack')
        if row['saved']: accepted.append(frame)
        previous = frame
    require(previous == END and len(accepted) == len(EARLIEST), 'coverage')
    return tuple(accepted)


def verify(root: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in (root / 'M1_CAPTURE_SCHEDULE.jsonl').read_text(encoding='utf-8').splitlines()]
    frames = order(rows)
    attach = json.loads((root / 'WHOLE_M1_DRIVER_STATUS.json').read_text(encoding='utf-8'))
    require(type(attach['start_frame']) is int and rows[0]['frame'] == attach['start_frame'] + STRIDE, 'attach_prefix')
    require(all(attach[k] is True for k in ('installed', 'closed', 'stopped', 'consumer_forbidden', 'session_restored'))
            and attach['last_frame'] == END and attach['quality_gate_clear'] is False
            and attach['error'] is None and attach['body_error'] is None, 'attach_failure')
    status = json.loads((root / 'BELIEF_M1_SESSION.json').read_text(encoding='utf-8'))
    require(status['original_targets'] == list(EARLIEST) and status['schedule_rows'] == len(rows), 'status_coverage')
    require(tuple(r['frame'] for r in status['saved']) == frames, 'status_saved')
    require(status['schedule'] == dict(last=END, accepted=list(frames), pending=None), 'status_state')
    for saved, frame in zip(status['saved'], frames, strict=True):
        path = root / f'BELIEF_M1_{frame}.json'
        require(Path(saved['path']).resolve() == path.resolve(), 'saved_path')
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == saved['sha256'], 'saved_sha')
        packet = json.loads(raw)
        require(packet['result']['frame'] == frame and packet['parent_recapture_verified'] is True, 'saved_payload')
    return dict(frames=list(frames), rows=len(rows), order_and_saved_files_verified=True,
                qualification_proven=False, physical_replay_proven=False, quality_gate_clear=False)
