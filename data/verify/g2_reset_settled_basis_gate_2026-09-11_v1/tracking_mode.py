"""整数基準を偽装せず、確率基準取得後の原NEXT消費を別経路で連続採録する。"""
from __future__ import annotations
from dataclasses import asdict
import json
from types import MethodType
from typing import Any
import native_consumption as N

B = N.B
KEY = 'probabilistic_tracking_mode'


def patch(stack: Any, obj: Any, name: str, value: Any) -> None:
    existed, stored = name in vars(obj), vars(obj).get(name)
    original = getattr(obj, name)
    def restore() -> None:
        B.require(vars(obj).get(name) is value, 'tracking_restore_binding')
        setattr(obj, name, stored) if existed else vars(obj).pop(name)
        B.require(getattr(obj, name) == original, 'tracking_restore_original')
    stack.callback(restore)
    setattr(obj, name, value)


class Mode:
    def __init__(self, connection: Any, state: dict[str, Any], stream: Any) -> None:
        self.connection, self.state, self.stream = connection, state, stream
        self.native: N.Recorder | None = None
        self.activation: dict[str, Any] | None = None
        self.rows = 0
        self.error: str | None = None
        self.failure_item_repr: str | None = None

    def activate(self, item: Any) -> None:
        c, r = self.connection, self.connection.recovery
        if c.binding is None or self.native is not None:
            return
        value = c.registry.current(c.binding)
        B.require(item['token'] == c.binding.initial_call_token and item['scope']['frame_idx'] == value.frame,
                  'tracking_activation_call')
        pending = r.pending
        B.require(pending is not None and not pending['used'] and pending['epoch'] == value.scope[2]
                  and r.baseline_count == 0 and value.scope[-1] not in r.control.history, 'tracking_activation_state')
        self.native = N.Recorder(c)
        self.activation = dict(frame=value.frame, source_call_token=item['token'], prior_pending=dict(pending),
            mode='probabilistic_only', integer_baseline_established=False,
            acquisition_deadline=c.observer.gate.deadline, tracking_deadline=value.deadline)
        # 有資格の別基準を取得した時だけ待機を終了する。整数baseline_countは増やさない。
        r.pending = None

    def observe(self, item: Any, result: Any, error: Any) -> dict[str, Any]:
        B.require(self.native is not None, 'tracking_not_active')
        event = self.native.observe(item, error)
        r = self.connection.recovery
        typed = r.state['postcommit_current_receiver'].rec.side_value(result)
        return dict(kind='probabilistic_tracking_observation', scope=item['scope'], journal_token=item['token'],
            state=typed['state_value'], native_consumption=None if event is None else asdict(event),
            pending_occurrences=[e.occurrence_token for e in self.native.pending],
            physical_transition_applied=False, integer_current_published=False, quality_gate_clear=False,
            reason='awaiting_physical_transition' if self.native.pending else 'no_native_consumption')

    def save(self, row: dict[str, Any]) -> None:
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        self.stream.flush()
        self.rows += 1

    def fail(self, error: BaseException) -> None:
        self.error = repr(error)
        self.connection.recovery.error = self.error
        self.connection.recovery.failure = error

    def close(self) -> None:
        result = dict(activation=self.activation, rows=self.rows, error=self.error,
            failure_item_repr=self.failure_item_repr,
            pending_native_consumptions=0 if self.native is None else len(self.native.pending),
            physical_tracking_complete=False, quality_gate_clear=False)
        with (self.state['output'] / 'PROBABILISTIC_TRACKING_STATUS.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)


def handlers(value: Mode, recovery: Any, sink: Any) -> tuple[Any, ...]:
    old_complete, old_selected, old_before, old_after = recovery.complete, recovery.provider.selected, sink.before, sink.after
    def complete(self: Any, item: Any, result: Any, error: Any) -> Any:
        returned = old_complete(item, result, error)
        if error is None:
            value.activate(item)
        return returned
    def selected(pipe: Any, side: str, frame: int, clock: float) -> bool:
        if value.native is None or side != value.connection.binding.scope[-1]:
            return old_selected(pipe, side, frame, clock)
        B.require(pipe is recovery.pipe, 'tracking_selected_pipe')
        # 原Controllerの整数履歴処理を選ばず、原native処理へ戻す。FIFOを手で消費しない。
        return False
    def before(item: Any, result: Any, error: Any) -> Any:
        if value.native is None or item['scope']['side'] != value.connection.binding.scope[-1]:
            return old_before(item, result, error)
        if error is not None:
            value.fail(error)
            return recovery.complete(item, result, error)
        try:
            returned = recovery.complete(item, result, error)
            B.require(returned is None, 'tracking_unexpected_integer_row')
            return value.observe(item, result, error)
        except BaseException as failure:
            value.failure_item_repr = repr({key: item.get(key) for key in ('scope', 'token', 'epoch', 'events')})
            value.fail(failure)
            raise
    def after(row: Any, control: Any) -> Any:
        if row.get('kind') != 'probabilistic_tracking_observation':
            return old_after(row, control)
        value.save(row)
        return None
    return MethodType(complete, recovery), selected, before, after


def install(stack: Any, connection: Any, state: dict[str, Any]) -> Mode:
    B.require(KEY not in state, 'duplicate_tracking_mode')
    stream = stack.enter_context((state['output'] / 'PROBABILISTIC_TRACKING.jsonl').open('x', encoding='utf-8'))
    value = Mode(connection, state, stream)
    state[KEY] = value
    stack.callback(value.close)
    r, sink = connection.recovery, state['live_history_sink']
    complete, selected, before, after = handlers(value, r, sink)
    patch(stack, r, 'complete', complete)
    patch(stack, r.provider, 'selected', selected)
    patch(stack, sink, 'before', before)
    patch(stack, sink, 'after', after)
    return value
