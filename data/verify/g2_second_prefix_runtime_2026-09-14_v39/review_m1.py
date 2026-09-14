"""元の保存検査を再利用し、採録終端だけを子と同じ限定観測区間へ束縛する。"""
from __future__ import annotations

from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import owned_adapter as A
W = A.W

SOURCE = W.ROOT.parent / 'g2_m1_second_runtime_2026-09-13_v21/review_m1.py'


def unsupported_saved(output: Path) -> dict | None:
    read = lambda name: json.loads((output / name).read_bytes())
    status = read('LEGACY_M1_COMPATIBILITY.json')
    if status['legacy_m1_complete']:
        return None
    require = A.COMPAT.C.require
    packet = read('BELIEF_M1_SESSION.json')
    require(status['status'] == 'unsupported' and status['old_finish_error'] == A.COMPAT.C.INCOMPLETE
        and status['observation_reached_end'] and status['quality_gate_clear'] is False, 'saved_compatibility')
    state = status['schedule']
    require(state == packet['schedule'] and state['last'] == W.LAST and state['pending'] is None
        and state['accepted'] == [row['frame'] for row in packet['saved']], 'saved_compatibility_schedule')
    require(packet['error'] is None and packet['session_error'] is None and packet['restored']
        and packet['observer_closed'] and packet['witness_closed'] and packet['evaluation_flags_closed'],
        'saved_compatibility_cleanup')
    completion = read('JOINT_EVENTS.jsonl.complete.json')
    require(completion['closed'] and completion['child_exit_code'] == 0
        and completion['accepted_count'] == len(packet['saved']), 'saved_compatibility_child_exit')
    schedule = [json.loads(line) for line in (output/'M1_CAPTURE_SCHEDULE.jsonl').read_text().splitlines()]
    require(schedule and schedule[-1]['frame'] == W.LAST, 'saved_compatibility_end')
    holds_path = output/'LEGACY_M1_COMPATIBILITY.jsonl'
    holds = [json.loads(line) for line in holds_path.read_text().splitlines()] if holds_path.exists() else []
    decisions = {row['frame']: row for row in schedule}
    require(len({row['frame'] for row in holds}) == len(holds) and all(
        row['reason'] == A.COMPAT.C.REASON and not row['requested']
        and row['reason'] in decisions[row['frame']]['reasons']
        and decisions[row['frame']]['action'] == 'WAIT' and not decisions[row['frame']]['saved']
        for row in holds), 'saved_compatibility_holds')
    return dict(legacy_M1_compatibility='unsupported', legacy_M1_complete=False,
        actual_saved_count=len(packet['saved']), explicit_raw_stable_holds=len(holds),
        compatibility_observation_closed=True, quality_gate_clear=False)


def bind_timeline(stack: Any, original: Any, packet: dict, saved: dict) -> None:
    """再計算済みの同一対象だけを旧資格検査へ渡し、所有scope終了時に復元する。"""
    def timeline(actual: dict, steps: dict, end: int) -> dict:
        original.WHOLE.require(actual == packet and end == W.LAST,
                               'terminal_qualification_scope')
        original.WHOLE.require(all(frame in steps for frame in saved),
                               'terminal_qualification_source_coverage')
        return dict(saved)
    A.A.A.A.V4.replace_owned(stack, original.Q.P, 'timeline', timeline)


def inspect(output: Path) -> dict:
    """旧source/資格/採録数を変えず、成功票と全metadataを元検査へ渡す。"""
    import postrun
    terminal_result, terminal_packet, terminal_timeline = postrun.verify(output)
    compatibility = unsupported_saved(output)
    if compatibility is not None:
        return compatibility | dict(terminal_saved_replay=terminal_result,
            terminal_pending_from_full_mathematical_replay=True)
    with ExitStack() as stack:
        previous = list(sys.path)
        stack.callback(sys.path.__setitem__, slice(None), previous)
        spec = importlib.util.spec_from_file_location('_split_original_m1_review', SOURCE)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        replace = A.A.A.A.V4.replace_owned
        original.WHOLE.require(original.Q.S.END == original.WHOLE.LAST == W.OLD_LAST,
                               'split_original_bounds')
        replace(stack, original.Q.S, 'END', W.LAST)
        replace(stack, original.WHOLE, 'LAST', W.LAST)
        bind_timeline(stack, original, terminal_packet, terminal_timeline)
        return original.inspect(output) | dict(terminal_saved_replay=terminal_result,
            terminal_pending_from_full_mathematical_replay=True)
