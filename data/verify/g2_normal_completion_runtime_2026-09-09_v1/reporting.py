"""同call完了の実票と有限復帰条件を別sinkへ保存する。元判定は免除しない。"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
from pathlib import Path
from typing import Any
import transaction as T

SIDECAR = 'normal_completion.jsonl'
STATUS = 'NORMAL_COMPLETION_STATUS.json'
RECEIPT = 'NORMAL_COMPLETION_RECEIPT.json'
REQUIRED = frozenset((SIDECAR, STATUS, RECEIPT))
FIRST, LAST, STRIDE, SIDE = 34796, 34910, 2, '1P'
EARLY_FRAME, EXPECTED_COMPLETE = 34802, 34898
OLD_PAIR, NEW_PAIR = [4, 3], [2, 5]


def write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Sink:
    def __init__(self, output: Path, controller: Any, journal: Any) -> None:
        self.output, self.controller, self.journal = output, controller, journal
        self.stream = (output / SIDECAR).open('x', encoding='utf-8')
        self.rows: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.closed = False

    def before_close(self, item: Any, result: Any, error: Any) -> Any:
        frame = item['frame']
        T.require(frame is not None and frame.f_code in self.journal.codes, 'completion_log_actual_frame')
        values = frame.f_locals
        ticket = self.controller.tickets.get(id(frame))
        observation = self.controller.pending.get(SIDE)
        original = type(self.journal).__init__.__globals__
        return {'kind': 'completion_step', **copy.deepcopy(item['scope']), 'software_reset': item['epoch'],
            'code_sha256': hashlib.sha256(frame.f_code.co_code).hexdigest(),
            'journal_token': item['token'], 'error': None if error is None else repr(error),
            'original_call_count': 1, 'state': None if result is None else result.state.value,
            'raw': original['board'](values.get('cnn_board')),
            'previous_confirmed': original['board'](values.get('prev_confirmed')),
            'returned_confirmed': original['board'](None if result is None else result.confirmed_board),
            'accounting': original['account'](item['pipe'], SIDE),
            'committed': original['scalar'](values.get('committed')),
            'falling_pair': original['scalar'](values.get('falling_pair')),
            'grace_pair': original['scalar'](values.get('falling_pair_for_grace')),
            'candidate_token': None if observation is None else observation.token,
            'ticket_token': None if ticket is None else ticket['item'].token,
            'ticket_infer_calls': None if ticket is None else ticket['infer_calls'],
            'physical_certified': False, 'quality_gate_clear': False}

    def save(self, row: Any) -> None:
        decisions = [r for r in self.controller.records if r['frame'] == row['frame_idx'] and r['side'] == SIDE]
        row['decisions'] = copy.deepcopy(decisions)
        row['pending_after'] = SIDE in self.controller.pending
        row['controller_error'] = self.controller.sticky_error
        value = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
        self.stream.write(value + '\n')
        self.rows.append(json.loads(value))

    def close(self) -> None:
        self.stream.close()
        self.closed = True
        write(self.output / STATUS, {'closed': True, 'errors': self.errors})


def install(stack: Any, state: Any) -> None:
    journal, controller = state['atomic_journal_observer'], state['normal_completion_controller']
    sink = Sink(Path(state['output']), controller, journal)
    state['normal_completion_sink'] = sink
    stack.callback(sink.close)
    original = journal.complete_step

    @functools.wraps(original)
    def completed(item: Any, result: Any, error: Any, profile: Any) -> None:
        scope = item['scope']
        if scope['side'] != SIDE or not FIRST <= scope['frame_idx'] <= LAST:
            return original(item, result, error, profile)
        row, original_error = None, None
        try:
            row = sink.before_close(item, result, error)
        except BaseException as failure:
            sink.errors.append('before_close:' + repr(failure))
        try:
            original(item, result, error, profile)
        except BaseException as failure:
            original_error = failure
            sink.errors.append('original_complete:' + repr(failure))
            raise
        finally:
            if row is not None:
                try:
                    sink.save(row)
                except BaseException as failure:
                    sink.errors.append('save:' + repr(failure))
                    if error is None and original_error is None:
                        raise
    journal.complete_step = completed
    stack.callback(setattr, journal, 'complete_step', original)


def check(output: Path) -> dict[str, Any]:
    status = json.loads((output / STATUS).read_text())
    T.require(status == {'closed': True, 'errors': []}, 'normal_sink_failed')
    rows = [json.loads(line) for line in (output / SIDECAR).read_text().splitlines()]
    T.require([r['frame_idx'] for r in rows] == list(range(FIRST, LAST + 1, STRIDE)), 'normal_scope_coverage')
    T.require(all(r['side'] == SIDE and r['error'] is None and r['controller_error'] is None
        and r['original_call_count'] == 1 and r['physical_certified'] is False
        and r['quality_gate_clear'] is False for r in rows), 'normal_scope_failure')
    releases = [r for r in rows if any(d.get('execution_complete') is True for d in r['decisions'])]
    T.require([r['frame_idx'] for r in releases] == [EXPECTED_COMPLETE], 'finite_completion_not_observed')
    first = next(r for r in rows if r['frame_idx'] == EARLY_FRAME)
    T.require(first['state'] != 'stable' and first['committed'] is None, 'old_premature_exit_not_closed')
    complete = releases[0]
    T.require(complete['ticket_token'] is not None and complete['ticket_infer_calls'] == 1
        and complete['committed'] == complete['falling_pair'] == OLD_PAIR
        and complete['accounting']['pending_tsumo'] == [NEW_PAIR], 'old_pair_not_exclusively_completed')
    later = [r for r in rows if r['frame_idx'] > EXPECTED_COMPLETE]
    T.require(all(r['committed'] is None for r in later) and not rows[-1]['pending_after'], 'duplicate_or_unclosed_completion')
    return {'early_hold_frame': EARLY_FRAME, 'completion_frame': EXPECTED_COMPLETE, 'rows': len(rows),
        'source_id': complete['source_id'], 'run_id': complete['run_id'],
        'committed_token': complete['ticket_token'], 'physical_certified': False, 'quality_gate_clear': False,
        'sha256': {name: sha(output / name) for name in (SIDECAR, STATUS)}}


def finish(state: Any) -> None:
    write(Path(state['output']) / RECEIPT, check(Path(state['output'])))


def verify(output: Path) -> None:
    T.require(json.loads((output / RECEIPT).read_text()) == check(output), 'normal_receipt_changed')
