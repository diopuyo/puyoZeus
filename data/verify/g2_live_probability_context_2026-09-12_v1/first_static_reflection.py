"""原取得票と生きた1P所有者を照合し、静止初回だけの公開資格を追加する。"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
from types import FunctionType, MethodType, SimpleNamespace as N
from typing import Any

VERIFY = Path(__file__).resolve().parent.parent
PUB = VERIFY / 'g2_belief_live_publication_2026-09-11_v1'
KEY = '_g2_probabilistic_scope_registry'


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('first_static_reflection:' + reason)


def receipt(connection: Any, value: Any, serializer: Any, output: Path) -> dict:
    stream = connection.stream
    path = output / 'PROBABILISTIC_BASIS.jsonl'
    require(not stream.closed and Path(stream.name).resolve() == path.resolve(), 'basis_stream')
    require(connection.rows == 1, 'basis_row_count')
    stream.flush()
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == connection.rows, 'basis_saved_count')
    row = rows[0]
    require(row['kind'] == 'actual_settled_probabilistic_basis', 'basis_kind')
    require(row['source_call_token'] == connection.binding.initial_call_token, 'basis_saved_token')
    require(serializer.decode(row['state']) == value, 'basis_saved_state')
    require(row['tracking_deadline'] == value.deadline == connection.deadline, 'basis_tracking_deadline')
    require(row['basis_deadline'] == connection.observer.gate.deadline, 'basis_acquisition_deadline')
    candidate = json.loads(json.dumps(asdict(connection.observer.gate.candidate), allow_nan=False))
    require(row['initial_candidate'] == candidate, 'basis_saved_candidate')
    require(all(row[k] is False for k in ('integer_current_published', 'legacy_collector_append', 'quality_gate_clear')), 'basis_permissions')
    return dict(basis_frame=value.frame, initial_token=row['source_call_token'],
                receipt_sha256=hashlib.sha256(raw).hexdigest())


class Verifier:
    def __init__(self, state: dict, factory: Any, first: Any, original: Any, serializer: Any, observer_type: type) -> None:
        self.state, self.factory, self.first = state, factory, first
        self.connection = first.connection
        self.original, self.serializer, self.observer_type = original, serializer, observer_type
        self.report: dict[str, Any] = dict(checks=[], restored=False, error=None, quality_gate_clear=False)

    def verify(self, mode: Any, value: Any, token: str, frame: int) -> None:
        if mode is not self.first or mode.applied or token == mode.connection.binding.initial_call_token:
            return self.original.verify(mode, value, token, frame)
        try:
            self.check(mode, value, token, frame)
        except BaseException as error:
            self.report['error'] = repr(error)
            raise

    def check(self, mode: Any, value: Any, token: str, frame: int) -> None:
        require(not getattr(mode, 'closed', False) and getattr(mode, 'retired_receipt', None) is None, 'reflection_closed')
        state, c, factory = self.state, self.connection, self.factory
        require(state['probabilistic_tracking_mode'] is mode and mode.connection is c, 'mode_owner')
        require(state['probabilistic_basis_connection'] is c and c.recovery.factory is factory, 'connection_owner')
        require(getattr(factory, KEY) is state[KEY] is c.registry and c.registry.factory is factory, 'registry_owner')
        require(c.registry.current(c.binding) is value and value.scope[-1] == '1P', 'registry_current')
        observer, r, native = c.observer, c.recovery, mode.native
        require(type(observer) is self.observer_type and observer.recovery is r, 'observer_owner')
        require(mode.error is None and r.error is None and r.failure is None and observer.error is None, 'prior_error')
        require(not r.journal.closed and not r.journal.errors and r.journal.active is None, 'journal_lifetime')
        gate, activation = observer.gate, mode.activation
        require(gate.state == 'ISSUED' and gate.candidate is not None, 'issued_gate')
        candidate = gate.candidate
        require(candidate.scope == value.scope and candidate.frame == value.frame == activation['frame'], 'basis_identity')
        require(candidate.source_call_token == c.binding.initial_call_token == activation['source_call_token'], 'initial_token')
        require(activation['tracking_deadline'] == value.deadline and activation['acquisition_deadline'] == gate.deadline, 'activation_deadlines')
        require(native is not None and native.connection is c, 'native_owner')
        require(native.last_frame == frame and token in native.seen_calls, 'native_current_J')
        require(not native.pending and not native.seen_occurrences and not mode.applied, 'native_unconsumed')
        require(not mode.origins and mode.basis_origin is None and mode.basis_cascade_closed is False, 'unsettled_origin')
        require(type(frame) is int and value.frame <= frame <= value.deadline, 'evaluation_clock')
        saved = receipt(c, value, self.serializer, Path(state['output']))
        self.report['checks'].append(dict(**saved, evaluation_frame=frame, evaluation_token=token))


def install(stack: Any, session: Any) -> dict:
    original = session.capture
    require('capture' not in vars(session), 'capture_already_owned')
    first = session.state['probabilistic_tracking_mode']
    require(Path(inspect.getfile(type(first))).resolve() == VERIFY / 'g2_basis_cascade_candidate_2026-09-11_v1/mode_v2.py', 'first_mode_source')
    v3 = original.__func__.__globals__['V']
    require(Path(v3.__file__).resolve() == PUB / 'live_binding_v3.py', 'binding_source')
    prior = v3._CURRENT
    require(Path(prior.__code__.co_filename).resolve() == PUB / 'live_binding_v2.py', 'current_source')
    old = prior.__globals__['R']
    require(Path(old.__file__).resolve() == PUB / 'reflection_v3.py', 'reflection_source')
    observer_type = type(first.connection).__init__.__globals__['A'].Observer
    verifier = Verifier(session.state, session.factory, first, old, old.S, observer_type)
    current = FunctionType(prior.__code__, dict(prior.__globals__, R=N(verify=verifier.verify)))
    capture = FunctionType(original.__func__.__code__, dict(original.__func__.__globals__, V=N(current=current)))
    bound = MethodType(capture, session)
    def restore() -> None:
        require(vars(session).get('capture') is bound, 'capture_foreign_hook')
        del session.capture
        require(session.capture == original, 'capture_restore')
        verifier.report['restored'] = True
    stack.callback(restore)
    session.capture = bound
    return verifier.report
