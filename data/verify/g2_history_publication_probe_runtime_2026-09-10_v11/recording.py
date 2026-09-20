"""原Jの同call閉鎖を外から採録。履歴/現在writerへ書き込まない。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import functools
import io
from typing import Any
import common as K

SIDECAR, STATUS = 'directional_history.jsonl', 'HISTORY_PROBE_STATUS.json'
OBSERVER_KEYS = ('sink', 'guard_sink', 'completion_sink', 'hsv_witness',
    'hidden_probability_observer', 'generation_publication_hold', 'provisional_current_connection',
    'current_scope_sink', 'provisional_context_observer', 'floating_exit_sink',
    'private_publication_consumer', 'collector_metadata_sink', 'atomic_journal_observer',
    'palette_evidence_veto', 'normal_completion_sink', 'evidence_sink')


def observer_status(state: Any) -> dict[str, Any]:
    result = {}
    for key in OBSERVER_KEYS:
        K.require(key in state, 'missing_observer:' + key)
        value = state[key]
        inner = value if hasattr(value, 'closed') else value.base_sink
        result[key] = {name: getattr(inner, name) for name in
            ('closed', 'errors', 'failures', 'sticky_error', 'error') if hasattr(inner, name)}
        K.require(result[key].get('closed') is True, 'unclosed_observer:' + key)
        K.require(all(v in (None, [], False) for n, v in result[key].items() if n != 'closed'),
                  'observer_failure:' + key)
    return result


def stream_handles(state: Any) -> dict[str, Any]:
    found, visited = {}, set()
    def visit(value: Any, depth: int) -> None:
        if id(value) in visited or depth > 4:
            return
        visited.add(id(value))
        if isinstance(value, io.IOBase):
            found[str(getattr(value, 'name', id(value)))] = value
            return
        for name in ('stream', 'base_sink', 'rows', 'writer', 'sink', 'ledger_stream'):
            if hasattr(value, name):
                visit(getattr(value, name), depth + 1)
    for value in state.values():
        visit(value, 0)
    return found


class Sink:
    def __init__(self, state: Any, base: Any) -> None:
        self.state, self.base = state, base
        self.path = state['output'] / SIDECAR
        self.stream = self.path.open('x', encoding='utf-8')
        self.errors: list[str] = []
        self.rows, self.closed, self.updates = 0, False, []

    def before(self, item: Any, result: Any, error: Any) -> dict[str, Any]:
        journal, control = self.state['atomic_journal_observer'], self.state['normal_completion_controller']
        caller = item['frame']
        K.require(caller is not None and caller.f_code in journal.codes, 'history_report_actual_code')
        local, call = caller.f_locals, control.calls.get(id(caller))
        K.require(call is not None and call['frame'] is caller, 'history_report_actual_call')
        view, binding = call['view'], call['binding']
        native = type(journal).__init__.__globals__
        raw = journal.history.capture.get(id(local['self']._stable_color_memory_1p))
        return self.base.json_value(dict(scope=item['scope'], journal_token=item['token'],
            code_sha256=__import__('hashlib').sha256(caller.f_code.co_code).hexdigest(),
            raw_capture=raw, processed_cnn=native['board'](local.get('cnn_board')),
            returned=native['board'](None if result is None else result.confirmed_board),
            original_state=None if result is None else result.state.value,
            counter_before=call['counter'], first_move_before=call['first_move'],
            account_after=native['account'](local['self'], '1P'),
            committed=native['scalar'](local.get('committed')), view_tokens=view.tokens,
            added=view.added, baseline_grid=binding.current, inventory_grid=binding.grid,
            prepared=None if call['prepared'] is None else call['prepared']['proof'],
            window=local['signals'].effect_gate_window_active,
            error=None if error is None else repr(error)))

    def after(self, row: Any, control: Any) -> None:
        import json
        frame = row['scope']['frame_idx']
        decisions = [value for value in control.records if value['frame'] == frame and value['side'] == '1P']
        K.require(len(decisions) == 1, 'history_report_decision')
        binding = control.history['1P']
        row.update(decision=decisions[0], grid_after=binding.grid,
            next_token=binding.next_token, consumed_tokens=sorted(binding.consumed_tokens),
            live_identity_checked=True, content_sha_is_signature=False)
        self.stream.write(json.dumps(self.base.json_value(row), ensure_ascii=False, allow_nan=False) + '\n')
        self.rows += 1

    def close(self) -> None:
        self.stream.close()
        self.closed = True
        K.write(self.state['output'] / STATUS, dict(closed=True, errors=self.errors,
            rows=self.rows, updates=self.updates, bounds=K.bounds()))


def install(stack: Any, state: Any, collector: Any, base: Any) -> Sink:
    sink = Sink(state, base)
    stack.callback(sink.close)
    journal, control = state['atomic_journal_observer'], state['normal_completion_controller']
    original = journal.complete_step
    @functools.wraps(original)
    def complete(item: Any, result: Any, error: Any, profile: Any) -> Any:
        scope, row = item['scope'], None
        try:
            if scope['side'] == '1P' and K.HISTORY_FIRST <= scope['frame_idx'] < K.END:
                row = sink.before(item, result, error)
        except BaseException as caught:
            sink.errors.append('before:' + repr(caught))
        returned = original(item, result, error, profile)
        if row is not None:
            try:
                sink.after(row, control)
            except BaseException as caught:
                sink.errors.append('after:' + repr(caught))
                if error is None:
                    raise
        return returned
    base.patch(stack, journal, 'complete_step', complete)
    original_update = collector.RecognitionPipeline.update
    @functools.wraps(original_update)
    def update(pipe: Any, frame: int, clock: float, pixels: Any) -> Any:
        K.require(len(sink.updates) < len(K.FRAMES) and frame == K.FRAMES[len(sink.updates)]
            and clock == frame / K.FPS, 'actual_update_bounds')
        result = original_update(pipe, frame, clock, pixels)
        sink.updates.append(frame)
        return result
    base.patch(stack, collector.RecognitionPipeline, 'update', update)
    return sink
