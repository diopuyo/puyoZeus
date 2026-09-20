"""可変有資格採録を原親要求検査とwhole終端へ接続。旧collector票は要求しない。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from review_pacing import verify as verify_pacing

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
CANDIDATE = ROOT.parent / 'g2_m1_completion_candidate_2026-09-12_v1'
sys.path.insert(0, str(ROOT.parent / 'g2_probabilistic_scope_candidate_2026-09-11_v1'))
sys.path.insert(0, str(CANDIDATE))
import qualification_audit as Q
sys.path.insert(0, str(ROOT.parent / 'g2_joint_collector_runtime_2026-09-12_v1'))
import verify_joint_saved as ORIGINAL

SPEC = importlib.util.spec_from_file_location('_g2_m1_whole_scope', ROOT.parent / 'g2_ui_hsv_runtime_2026-09-12_v9/probe_review_m1_saved.py')
WHOLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WHOLE)


def inspect(output: Path) -> dict:
    waited = ORIGINAL.read(output / 'TARGET_PARENT_WAIT.json')
    WHOLE.require(waited['child_exit_code'] == waited['resource_guard_exit'] == 0
                  and waited['source'] == 'actual_wait'
                  and ORIGINAL.read(output / 'ENTRY_RESULT.json')['exit_code'] == 0, 'actual_wait')
    pacing = verify_pacing(output)
    qualification = Q.verify(output)
    frames = tuple(qualification['frames'])
    names = tuple(f'BELIEF_M1_{frame}.json' for frame in frames)
    packets = ORIGINAL.packets(output, 'JOINT_EVENTS.jsonl', names, frames)
    WHOLE.require(all(p['result']['supported'] is True for p in packets), 'M1_supported')
    session = ORIGINAL.read(output / 'BELIEF_M1_SESSION.json')
    WHOLE.require(all(session[k] is True for k in ('restored', 'observer_closed', 'witness_closed', 'evaluation_flags_closed')),
                  'session_cleanup')
    for index, frame in enumerate(frames):
        ticket = ORIGINAL.read(output / f'JOINT_EVENTS.jsonl.request{index}.json')
        WHOLE.capture_scope(ticket['producer'], output, frame)
    WHOLE.capture_scope(ORIGINAL.read(output / 'JOINT_PRODUCER_CAPTURE.json'), output, WHOLE.LAST)
    with (output / 'collector_metadata.jsonl').open(encoding='utf-8') as stream:
        count = WHOLE.metadata_scope(json.loads(line) for line in stream)
    return qualification | pacing | dict(original_parent_packet_checks=True, metadata_sides=count,
        supported=[packet['result']['supported'] for packet in packets], terminal_producer_separate=True,
        runtime_source_verification_required=True, production_permission=False, quality_gate_clear=False)
