"""原私有基点/開始の成功返却を同callで固定する。状態と返却は変更しない。"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class Record:
    kind: str
    frame: int
    payload_json: str
    sha256: str


def record(kind: str, frame: int, payload: Any) -> Record:
    value = encoded(payload)
    return Record(kind, frame, value, hashlib.sha256(value.encode()).hexdigest())


def source(binding: Any, basis: Any) -> dict[str, Any]:
    return dict(scope=basis.scope, basis_frame=basis.frame, basis_clock=basis.clock,
        basis_action=basis.action, basis_grid=basis.grid, basis_state=asdict(basis.state),
        tail_proof=basis.proof, tail_digest=basis.digest, successor_proof=basis.source.proof,
        source_tokens=basis.source.tokens, source_ref_ids=[id(v) for v in basis.source.refs],
        head_ref_id=id(basis.head), queue_ref_id=id(basis.queue), suffix_token=basis.token,
        current_permission=False, probability_permission=False, evaluation_permission=False,
        physical_certified=False, binding_ref_id=id(binding), basis_ref_id=id(basis))


def install(stack: Any, factory: Any, module: Any, patch: Any) -> None:
    control = factory.controller
    assert not hasattr(control, 'private_suffix_evidence')
    control.private_suffix_evidence = []
    old_capture, old_start = module.capture, module.start
    def capture(actual: Any, binding: Any, call: Any) -> Any:
        value = old_capture(actual, binding, call)
        if actual is control:
            payload = source(binding, value)
            payload.update(consumed=call['consumed'], call_frame=call['view'].frame,
                post_queue_ref_ids=[id(v) for v in call['view'].queue])
            control.private_suffix_evidence.append(record('private_suffix_basis_capture/v1', value.frame, payload))
        return value
    def start(actual: Any, binding: Any, view: Any) -> bool:
        basis = getattr(binding, 'private_suffix_basis', None)
        before = None if basis is None else basis.started
        result = old_start(actual, binding, view)
        if actual is control and basis is not None and before is None and basis.started is not None:
            payload = source(binding, basis)
            payload.update(started=basis.started, started_state=asdict(basis.started_state),
                scope_at_start=view.scope, frame_at_start=view.frame, clock_at_start=view.clock,
                quiet=view.quiet, tokens=view.tokens, added=view.added,
                next_pair=view.next_pair, dnext_pair=view.dnext_pair,
                refs_at_start=[id(v) for v in view.refs], queue_at_start=id(view.queue),
                original_return=result)
            control.private_suffix_evidence.append(record('private_suffix_start_return/v1', view.frame, payload))
        return result
    patch(stack, module, 'capture', capture)
    patch(stack, module, 'start', start)


def verify(control: Any) -> list[Any]:
    values = control.private_suffix_evidence
    result = []
    for value in values:
        assert type(value) is Record
        assert hashlib.sha256(value.payload_json.encode()).hexdigest() == value.sha256
        payload = json.loads(value.payload_json)
        assert all(payload[name] is False for name in ('current_permission',
            'probability_permission', 'evaluation_permission', 'physical_certified'))
        result.append(asdict(value))
    return result
