"""私有配置の原成功返却を当時の状態として固定する。後続手を巻き戻さない。"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import CodeType
from typing import Any

PRIVATE = Path(__file__).resolve().parent.parent/'g2_private_suffix_continuation_2026-09-10_v1'


@dataclass(frozen=True)
class Receipt:
    frame: int
    payload_json: str
    sha256: str


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def capture(factory: Any, E: Any, basis: Any, placement: Any, call: Any) -> None:
    control, binding, view = factory.controller, call['binding'], call['view']
    value = binding.private_suffix_basis
    basis.validate(control, binding, value)
    proof = placement.committed(basis, control, binding)
    assert call['consumed'] is True and not view.queue and binding.next_token is None
    state = binding.owner.state
    policy = basis.logs(binding)
    payload = dict(kind='private_suffix_original_commit_return/v1', frame=view.frame, clock=view.clock,
        scope=view.scope, basis=E.source(binding, value), started=value.started,
        started_state=asdict(value.started_state), state=asdict(state), proof=proof,
        policy_prefix=policy, queue_ref_id=id(view.queue), consumed=True, current_permission=False, physical_certified=False)
    text = encoded(payload)
    receipt = Receipt(view.frame, text, hashlib.sha256(text.encode()).hexdigest())
    assert not any(r.frame == receipt.frame for r in control.private_suffix_completion_rows)
    control.private_suffix_completion_rows.append(receipt)
    control.private_suffix_completion_refs[receipt.sha256] = (factory, binding, value, state, proof, policy)


def install(stack: Any, factory: Any, E: Any, basis: Any, placement: Any, patch: Any) -> None:
    control, original = factory.controller, placement.consumed
    path = PRIVATE/'placement.py'
    assert Path(placement.__file__).resolve() == path and original.__globals__ is vars(placement)
    declared = next(c for c in compile(path.read_bytes(), str(path), 'exec', dont_inherit=True).co_consts
        if isinstance(c, CodeType) and c.co_name == 'consumed')
    assert original.__code__ == declared
    assert not hasattr(control, 'private_suffix_completion_rows')
    control.private_suffix_completion_rows, control.private_suffix_completion_refs = [], {}
    def consumed(actual: Any, who: Any, call: Any, caller: Any, rows: Any) -> Any:
        result = original(actual, who, call, caller, rows)
        if who is control:
            assert actual is basis
            capture(factory, E, basis, placement, call)
        return result
    patch(stack, placement, 'consumed', consumed)


def related(E: Any, basis: Any, binding: Any, rows: Any) -> list[Any]:
    found = [r for r in rows if json.loads(r['payload_json'])['basis_ref_id'] == id(basis)]
    assert len(found) == 2 and [r['kind'] for r in found] == [
        'private_suffix_basis_capture/v1', 'private_suffix_start_return/v1']
    expected = E.source(binding, basis)
    first, last = [json.loads(r['payload_json']) for r in found]
    for value in (first, last):
        assert encoded({k: value[k] for k in expected}) == encoded(expected)
    assert first['consumed'] is True and first['call_frame'] == found[0]['frame'] == basis.frame
    assert last['quiet'] is True and last['original_return'] is True and last['added'] == []
    assert last['frame_at_start'] == found[1]['frame'] == basis.started[0] > basis.frame
    assert last['clock_at_start'] == basis.started[1]
    assert encoded(last['scope_at_start']) == encoded(basis.scope)
    assert last['refs_at_start'] == [id(basis.head)] and last['queue_at_start'] == id(basis.queue)
    assert first['post_queue_ref_ids'] == [id(basis.head)]
    assert encoded(last['started_state']) == encoded(asdict(basis.started_state))
    assert encoded(last['tokens']) == encoded((basis.token,))
    assert encoded(last['next_pair']) == encoded(basis.source.proof['new_accepted'])
    assert encoded(last['dnext_pair']) == encoded(basis.source.proof['dnext'])
    return found


def verify(factory: Any, E: Any, journal: Any, history: Any, step: Any) -> dict[str, Any]:
    control = factory.controller
    saved, actual = E.verify(control), control.private_suffix_completion_rows
    assert len(saved) == len(actual)*2 and actual
    prepared = {(r['scope']['frame_idx'], r['scope']['side']): r for r in history
        if r['prepared'] and r['prepared']['kind'] == 'hidden_private_suffix_history/v1'}
    found = set()
    for receipt in actual:
        assert type(receipt) is Receipt and hashlib.sha256(receipt.payload_json.encode()).hexdigest() == receipt.sha256
        owned, binding, basis, state, proof, policy = control.private_suffix_completion_refs[receipt.sha256]
        value = json.loads(receipt.payload_json)
        assert owned is factory and value['frame'] == receipt.frame
        assert value['kind'] == 'private_suffix_original_commit_return/v1' and value['consumed'] is True
        assert encoded(value['scope']) == encoded(basis.scope) and value['queue_ref_id'] == id(basis.queue)
        assert encoded(value['basis']) == encoded(E.source(binding, basis))
        assert encoded(value['state']) == encoded(asdict(state)) and encoded(value['proof']) == encoded(proof)
        assert encoded(value['policy_prefix']) == encoded(policy[:len(value['policy_prefix'])])
        current = binding.owner.state
        assert control.history[value['scope'][-1]] is binding and current.scope == state.scope
        assert current.history[:len(state.history)] == state.history
        now_policy = getattr(binding.policy, 'accounting', binding.policy).proofs
        assert encoded(value['policy_prefix']) == encoded(now_policy[:len(value['policy_prefix'])])
        assert encoded(value['started_state']) == encoded(asdict(basis.started_state))
        assert value['current_permission'] is value['physical_certified'] is False
        control.inventory.S.validate_accounting(state)
        assert state.counter == control.inventory.S.color_counts(tuple(map(tuple, proof['grid'])))
        key = receipt.frame, value['scope'][-1]
        assert key not in found and key in prepared
        assert encoded(prepared[key]['prepared']) == encoded(proof)
        assert encoded(prepared[key]['decision']['history_state']) == encoded(asdict(state))
        for row in related(E, basis, binding, saved):
            step(journal, row['frame'], value['scope'])
        start = [r for r in history if r['scope']['frame_idx'] == basis.started[0]
            and r['scope']['side'] == value['scope'][-1]]
        assert len(start) == 1 and encoded(start[0]['decision']['history_state']) == encoded(asdict(basis.started_state))
        step(journal, receipt.frame, value['scope'])
        found.add(key)
    assert found == set(prepared)
    return dict(private_commit_samecall_verified=True, commits=len(actual),
        later_state_reinterpreted=False, physical_certified=False, quality_gate_clear=False)
