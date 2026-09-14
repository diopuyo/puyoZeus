"""A38の終端着弾を原J/原prefix数学/新保存票で検収し、M1のpending照合へ渡す。"""
from __future__ import annotations
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import target_entry as T

ROOT, TERM = T.ROOT, T.A.TERMINAL


def read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def lines(path: Path) -> list:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def modules(stack: Any, owner: Any, parts: Any) -> tuple:
    load = owner.bootstrap().load
    aliases = []
    def selected(name: str, path: Path, injection: dict | None = None) -> Any:
        alias = '_a38_saved_' + name
        T.require(alias not in sys.modules, 'saved_foreign_alias')
        value = load(alias, path, injection)
        aliases.append(alias)
        return value
    stack.callback(lambda: [sys.modules.pop(name, None) for name in aliases])
    state = selected('arrival_state', TERM / 'second_arrival_state.py')
    math = selected('terminal_math', T.A.TERMINAL_MATH)
    warning = selected('warning', T.A.WARNING_SOURCE)
    consumer = selected('terminal', T.A.TERMINAL_SAVED, dict(
        second_arrival_state=state, observed_terminal_drop=math, warning_source=warning))
    old_pending = selected('pending', ROOT.parent / 'g2_m1_completion_candidate_2026-09-12_v1/second_pending_replay.py',
                           dict(native_consumption=parts.mode.CORE.V1.N))
    # 元SP3保存consumerの既存遅延factoryを使い、型を混在させない。
    second = [m for m in tuple(sys.modules.values()) if getattr(m, '__file__', None)
              and Path(m.__file__).resolve() == T.A.PC.SECOND]
    T.require(len(second) == 1, 'saved_second_selection')
    second[0].session_class(N(Session=object), parts.mode)
    old_saved = sys.modules[T.A.PC.NAMES[3]]
    services = N(commit=sys.modules['_g2_prefix_commit'], lane=sys.modules['_g2_prefix_lane'],
                 notice=sys.modules['_g2_second_notice_v21'])
    return consumer, old_pending, old_saved, services


def source_rows(output: Path, journal: list) -> dict:
    source = lines(output / 'SECOND_ENQUEUE_SOURCE.jsonl')
    fields = [row['source_fields'] for row in source]
    original = {row['frame_idx']: row for row in journal if row['kind'] == 'enqueue' and row['side'] == '2P'}
    T.require(fields and len({row['frame_idx'] for row in fields}) == len(fields), 'saved_source_duplicate')
    for packet, row in zip(source, fields):
        T.require(original[row['frame_idx']] == row and packet['native_step_verified'] is False
                  and packet['physical_certified'] is False, 'saved_source_not_original_J')
    status = read(output / 'SECOND_ENQUEUE_SOURCE_STATUS.json')
    T.require(status['closed'] and status['restored'] and status['error'] is None
              and status['rows'] == len(fields), 'saved_source_lifetime')
    return {row['frame_idx']: row for row in fields}


def inspect(output: Path, parts: Any, dependency: tuple) -> tuple:
    consumer, original, saved, services = dependency
    packets = read(output / 'BELIEF_M1_SESSION.json')['modes']
    T.require(len(packets) == 1, 'terminal_saved_single_G2_target_scope')
    packet = packets[0]
    call = packet['initial']['source_call_token']
    name = sha256(call.encode()).hexdigest()[:16]
    terminal_path = output / ('SECOND_TERMINAL_' + name + '.jsonl')
    T.require(set(output.glob('SECOND_TERMINAL_*.jsonl')) == {terminal_path}, 'terminal_saved_unique_handoff')
    start = packet['initial']['state']['frame']
    journal = lines(output / 'atomic_journal.jsonl')
    source = source_rows(output, journal)
    steps = {row['frame_idx']: row for row in journal if row['kind'] == 'step'
             and row['side'] == '2P' and row['frame_idx'] >= start}
    contexts = {row['frame_idx']: row for row in lines(output / 'provisional_context.jsonl')
                if row['frame_idx'] >= start}
    old = saved.Replay(parts.mode.C.T, parts.binding.S, parts.mode.CORE.V1.N, original, packet, steps, contexts)
    replay = consumer.Replay(parts, services, old, source,
        tuple(lines(output / ('SECOND_WARNING_' + name + '.jsonl'))), lines(terminal_path))
    result = replay.run(lines(output / ('SECOND_PREFIX_' + call.replace(':', '_') + '.jsonl')), T.W.LAST)
    arrival = lines(output / ('SECOND_ARRIVAL_' + name + '.jsonl'))
    T.require(arrival[-1]['kind'] == 'finish' and arrival[-1]['error'] is None
              and arrival[-1]['state'] == consumer.normalized(__import__('dataclasses').asdict(replay.ledger)),
              'terminal_saved_arrival_finish')
    return result, packet, replay.pending_timeline


def verify(output: Path) -> tuple:
    T.approved(output)
    waited, entry = read(output / 'TARGET_PARENT_WAIT.json'), read(output / 'ENTRY_RESULT.json')
    T.require(waited == dict(child_exit_code=0, resource_guard_exit=0, source='actual_wait')
              and entry['exit_code'] == 0, 'terminal_saved_actual_exit')
    with ExitStack() as stack:
        T.A.configured(stack)
        sys.path.insert(0, str(ROOT.parent / 'g2_settled_basis_actual_mismatch_2026-09-12_v1'))
        import probe_native_merge
        owner = sys.modules[T.A.A.A.A.V4.OWNED_ALIAS]
        parts = owner.dependencies().modules()
        result = inspect(output, parts, modules(stack, owner, parts))
    T.approved(output)
    return result


def main() -> None:
    output = ROOT.parent / T.OUTPUT_NAME
    result, _, _ = verify(output)
    with (output / 'TERMINAL_SAVED_REVIEW.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
