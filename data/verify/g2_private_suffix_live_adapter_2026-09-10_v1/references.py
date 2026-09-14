"""既存candidate設置後の参照へ、追加層だけが先に戻ったことを個別保存する。"""
from __future__ import annotations
import hashlib
import marshal
from dataclasses import asdict
from typing import Any

KEY = 'private_suffix_reference_status'
FILENAME = 'PRIVATE_SUFFIX_REFERENCES.json'
EVIDENCE_FILE = 'PRIVATE_SUFFIX_LIVE_EVIDENCE.json'


def targets(factory: Any, modules: Any) -> dict[str, Any]:
    cls, history = type(factory.controller), modules.history
    v1 = cls.prepared.__globals__['V1']
    return {'tail.source': history.T.source, 'tail.vote': history.T.vote,
        'tail.consumed': history.consumed, 'controller.hand': cls.hand,
        'v1.prepared': v1.prepared, 'controller.call': cls.call,
        'controller.consumed_history': cls.consumed_history,
        'natural.eligible': modules.exits.N.eligible, 'conditional.make': modules.exits.C.make,
        'basis.capture': modules.basis.capture, 'basis.start': modules.basis.start,
        'placement.consumed': modules.placement.consumed}


def description(value: Any) -> dict[str, Any]:
    function = getattr(value, '__func__', value)
    code = getattr(function, '__code__', None)
    return dict(object_id=id(value), function_id=id(function),
        name=getattr(function, '__qualname__', type(function).__name__),
        code_sha256=None if code is None else hashlib.sha256(marshal.dumps(code)).hexdigest())


def save_evidence(factory: Any, state: Any, modules: Any, write: Any) -> None:
    control = factory.controller
    present = hasattr(control, 'private_suffix_evidence')
    starts = getattr(control, 'private_suffix_evidence', [])
    completions = getattr(control, 'private_suffix_completion_rows', [])
    payload = dict(basis_start_present=present, basis_start=[asdict(row) for row in starts],
        completion_present=hasattr(control, 'private_suffix_completion_rows'),
        completion=[asdict(row) for row in completions], integrity_verified=False,
        completion_semantics_verified=False, physical_certified=False, quality_gate_clear=False)
    try:
        if present:
            assert modules.evidence.verify(control) == payload['basis_start']
        for row in completions:
            assert type(row) is modules.completion.Receipt
            assert hashlib.sha256(row.payload_json.encode()).hexdigest() == row.sha256
        payload['integrity_verified'] = present and payload['completion_present']
    except BaseException as exc:
        payload['error'] = type(exc).__name__+':'+str(exc)
        raise
    finally:
        write(state['output']/EVIDENCE_FILE, payload)


def closed(factory: Any, state: Any, modules: Any, before: Any, write: Any) -> None:
    after, receipt = targets(factory, modules), state[KEY]
    equal = {name: after[name] is value for name, value in before.items()}
    receipt.update(closed=True, equal=equal, restored=all(equal.values()),
        after={name: description(value) for name, value in after.items()})
    try:
        save_evidence(factory, state, modules, write)
    except BaseException as exc:
        receipt['evidence_error'] = type(exc).__name__+':'+str(exc)
        raise
    finally:
        write(state['output']/FILENAME, receipt)
    assert receipt['restored'], 'private_suffix_reference_not_restored'


def watch(stack: Any, factory: Any, state: Any, modules: Any, write: Any) -> None:
    assert KEY not in state, 'private_suffix_reference_duplicate'
    before = targets(factory, modules)
    state[KEY] = dict(installed=False, closed=False, restored=False, error=None,
        boundary='after_original_candidate_before_private_addon',
        before={name: description(value) for name, value in before.items()},
        physical_certified=False, quality_gate_clear=False)
    stack.callback(closed, factory, state, modules, before, write)


def verify(state: Any) -> dict[str, Any]:
    receipt = state[KEY]
    assert receipt['installed'] and receipt['closed'] and receipt['restored'], 'private_suffix_unclosed'
    assert receipt['error'] is None and all(receipt['equal'].values()), 'private_suffix_install_failed'
    assert receipt.get('evidence_error') is None, 'private_suffix_evidence_failed'
    return receipt
