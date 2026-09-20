"""候補単体で合格にせず、元要求・原保存・外側終了を既存検査で照合する。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_joint_ledger_engine_2026-09-12_v1'))
import parent_transport as T
import parent_validation as V

FIRST, LAST, INTERVAL = 34772, 35410, 2
FRAMES = (35370, 35410)
SIDES = ('1P', '2P')


def read(path: Path) -> dict:
    return T.P.decoded(path.read_bytes())


def candidate(packet: dict, ticket: dict, part: Path) -> None:
    T.P.require(packet['schema'] == 'g2-joint-ledger-parent-candidate/v2', 'saved_candidate_schema')
    T.P.require(packet['parent_recapture_verified'] is True and all(packet[k] is False
                for k in ('actual_video', 'production_permission', 'quality_gate_clear')), 'saved_authority')
    T.P.require(packet['observation'] == ticket['observation'], 'saved_request_mismatch')
    T.P.require(packet['producer_sha256'] == V.digest(ticket['producer']), 'saved_producer_mismatch')
    T.P.require(packet['session_completion_required'] == str(part) + '.complete.json', 'saved_completion_path')
    V.validated(packet['result'], ticket['producer'], ticket['observation'])


def packets(output: Path, part_name: str, names: tuple[str, ...], frames: tuple[int, ...]) -> list[dict]:
    part = output / part_name
    complete = read(Path(str(part) + '.complete.json'))
    T.P.require(complete['schema'] == 'g2-joint-parent-session-completion/v2'
                and complete['closed'] is True and complete['child_exit_code'] == 0
                and complete['accepted_count'] == len(frames), 'saved_completion')
    T.P.require(complete['production_permission'] is False and complete['quality_gate_clear'] is False,
                'saved_completion_authority')
    T.P.require(not Path(str(part) + '.failure.json').exists(), 'saved_failure_present')
    values, identity = [], None
    for index, (name, frame) in enumerate(zip(names, frames, strict=True)):
        ticket, packet = read(Path(str(part) + f'.request{index}.json')), read(output / name)
        identity = ticket['identity'] if identity is None else identity
        T.P.require(ticket['identity'] == identity and ticket['seq'] == index and ticket['op'] == 'join'
                    and ticket['schema'] == T.P.SCHEMA and packet['result']['frame'] == frame, 'saved_ticket')
        candidate(packet, ticket, part)
        values.append(packet)
    T.readback(part, identity, frames[-1], (), complete['event_sha256'])
    batches = tuple(T.S.iter_committed_batches(part, expected_first_seq=0))
    for packet, frame in zip(values, frames, strict=True):
        prefix = [b for b in batches if b.events[0]['timing']['available_frame'] <= frame]
        T.P.require(prefix and prefix[-1].events[0]['timing']['available_frame'] == frame, 'saved_prefix_cutoff')
        digest = hashlib.sha256(b''.join(T.P.encoded(b.events) for b in prefix)).hexdigest()
        T.P.require(digest == packet['result']['event_sha256'], 'saved_prefix_sha')
    T.P.require(values[-1]['result']['event_sha256'] == complete['event_sha256'], 'saved_final_sha')
    return values


def lifecycle(session: dict, attach: dict, source: dict) -> None:
    T.P.require(all(session[k] is True for k in ('restored', 'observer_closed', 'witness_closed'))
                and all(session[k] is None for k in ('error', 'session_error', 'observer_error', 'witness_error')),
                'saved_session_cleanup')
    T.P.require(tuple(row['frame'] for row in session['saved']) == FRAMES, 'saved_session_frames')
    T.P.require(all(attach[k] is True for k in ('installed', 'restored', 'pipeline_update_unchanged'))
                and attach['error'] is None and attach['body_error'] is None, 'saved_attach_cleanup')
    T.P.require(source['original_exit'] == 0 and source['unchanged'] is True
                and source['entry_restored'] is True, 'saved_source_or_entry')


def verify(output: Path) -> dict:
    names = tuple(f'BELIEF_M1_{frame}.json' for frame in FRAMES)
    values = packets(output, 'JOINT_EVENTS.jsonl', names, FRAMES)
    session = read(output / 'BELIEF_M1_SESSION.json')
    lifecycle(session, read(output / 'BELIEF_M1_COLLECTOR_ATTACH.json'), read(output / 'JOINT_RUNTIME_SOURCE.json'))
    for name, row in zip(names, session['saved'], strict=True):
        T.P.require(row['path'] == str(output / name)
                    and row['sha256'] == hashlib.sha256((output / name).read_bytes()).hexdigest(), 'saved_session_candidate')
    capture = read(output / 'JOINT_PRODUCER_CAPTURE.json')
    last_ticket = read(output / f'JOINT_EVENTS.jsonl.request{len(FRAMES) - 1}.json')
    T.P.require(capture == last_ticket['producer'], 'saved_final_producer_changed')
    T.P.require(capture['observed_count'] == (LAST - FIRST) // INTERVAL + 1
                and capture['last_frame'] == LAST and capture['identity']['run_id'] == str(output), 'saved_capture_coverage')
    with (output / 'collector_metadata.jsonl').open(encoding='utf-8') as stream:
        metadata = [(v['frame_idx'], v['side']) for line in stream if (v := json.loads(line))]
    expected = [(frame, side) for frame in range(FIRST, LAST + 1, INTERVAL) for side in SIDES]
    T.P.require(metadata == expected, 'saved_metadata_coverage')
    return dict(frames=list(FRAMES), metadata_sides=len(metadata), updates=capture['observed_count'],
                supported=[p['result']['supported'] for p in values], candidate_and_completion_verified=True,
                actual_video=False, quality_gate_clear=False)


if __name__ == '__main__':
    directory = Path(sys.argv[1]).resolve()
    result = verify(directory)
    with (directory / 'JOINT_SAVED_VERIFIED.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
