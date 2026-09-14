"""旧raw-STABLE M1の未資格を記録し、新確率観測の失敗へ混同しない。"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace as N
from typing import Any

SIDES = ('1P', '2P')
REASON = 'legacy_m1_current_raw_stable_missing'
INCOMPLETE = 'capture_schedule:incomplete_coverage'


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError('legacy_m1_compatibility:' + reason)


def producer_ready(value: dict, frame: int) -> bool:
    require(type(frame) is int and value['last_frame'] == frame, 'producer_clock')
    metadata, rows = value['metadata'], value['stable_snapshots']
    require([(r['frame_idx'], r['side']) for r in metadata]
            == [(frame, side) for side in SIDES], 'producer_metadata_scope')
    require(type(rows) is list and len(rows) % len(SIDES) == 0, 'stable_pair_shape')
    if not rows:
        return False
    pair = rows[-len(SIDES):]
    last = pair[0]['frame_idx']
    require(type(last) is int and last <= frame and
        [(r['frame_idx'], r['side']) for r in pair] == [(last, side) for side in SIDES],
        'stable_pair_scope')
    return last == frame


def write_hold(session: Any, value: dict, frame: int) -> None:
    path = session.state['output'] / 'LEGACY_M1_COMPATIBILITY.jsonl'
    first = not getattr(session, '_legacy_m1_hold_started', False)
    record = dict(frame=frame, status='unsupported', reason=REASON,
        last_raw_stable=None if not value['stable_snapshots'] else value['stable_snapshots'][-1]['frame_idx'],
        producer_identity=value['identity'], requested=False, quality_gate_clear=False)
    with path.open('x' if first else 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    session._legacy_m1_hold_started = True


def completed(original: Any, session: Any, frame: int) -> Any:
    namespace = original.__globals__
    def reasons(owner: Any, tick: int) -> tuple[str, ...]:
        found = namespace['E'].reasons(owner, tick)
        if found:
            return found
        producer = owner.state['joint_producer_capture']
        value = producer.snapshot()
        require(value['identity'] == producer.identity, 'producer_identity')
        if producer_ready(value, tick):
            return ()
        write_hold(owner, value, tick)
        return (REASON,)
    function = FunctionType(original.__code__, namespace | {'E': N(reasons=reasons)},
        original.__name__, original.__defaults__, original.__closure__)
    return function(session, frame)


def finished(schedule: Any, state: Any, output: Path) -> tuple[int, ...]:
    schedule.check(state)
    require(state.last == schedule.END and state.pending is None, 'observation_incomplete')
    try:
        accepted, failure = schedule.finish(state), None
    except ValueError as error:
        if str(error) != INCOMPLETE:
            raise
        accepted, failure = state.accepted, str(error)
        require(len(accepted) < len(schedule.EARLIEST), 'unexpected_coverage_failure')
    result = dict(status='compatible' if failure is None else 'unsupported',
        old_finish_error=failure, schedule=asdict(state), legacy_m1_complete=failure is None,
        observation_reached_end=True, quality_gate_clear=False, production_permission=False)
    with (output / 'LEGACY_M1_COMPATIBILITY.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    return accepted


def closed(original: Any, session: Any, kind: Any, body: Any, trace: Any) -> bool:
    namespace = original.__globals__
    schedule = namespace['S']
    proxy = N(**(vars(schedule) | {'finish': lambda state: finished(schedule, state, session.state['output'])}))
    function = FunctionType(original.__code__, namespace | {'S': proxy},
        original.__name__, original.__defaults__, original.__closure__)
    return function(session, kind, body, trace)
