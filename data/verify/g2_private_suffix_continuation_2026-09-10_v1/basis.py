"""原tail配置後の私有基点。PB・評価・current・発火の権限は発行しない。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any
import tail_suffix_provenance as SP

KIND = 'hidden_single_tail_history/v1'
FPS, SUFFIX_SIZE = 60, 1


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('private_suffix_' + reason)


def logs(binding: Any) -> list[Any]:
    return getattr(binding.policy, 'accounting', binding.policy).proofs


@dataclass
class Basis:
    binding: Any
    owner: Any
    grid: Any
    action: int
    frame: int
    clock: float
    state: Any
    state_content: Any
    scope: Any
    head: Any
    token: str
    queue: Any
    proof: dict[str, Any]
    digest: str
    proof_content: Any
    source: Any
    source_content: Any
    policy_count: int
    policy_content: Any
    consumed: frozenset[str]
    started: tuple[int, float] | None = None
    started_state: Any = None
    current_permission: bool = False
    probability_permission: bool = False
    evaluation_permission: bool = False
    physical_certified: bool = False


def history(control: Any, binding: Any, record: Basis) -> None:
    p, state = control.inventory, binding.owner.state
    require(record.owner is binding.owner and record.scope == binding.scope
        and SP.content(record.scope) == SP.content(record.source.scope), 'owner_scope')
    require(SP.content(asdict(record.state)) == record.state_content, 'old_state_changed')
    require(state.scope == record.state.scope and state.action >= record.action
        and state.history[:len(record.state.history)] == record.state.history, 'history_prefix')
    require(not state.origins and not state.debts and not state.consumed_ids, 'unclosed_origin')
    p.S.validate_accounting(state)
    p.S.validate_grid(binding.grid)
    require(state.counter == p.S.color_counts(binding.grid), 'counter_grid')
    require(SP.content(logs(binding)[:record.policy_count]) == record.policy_content, 'policy_prefix')
    require(SP.content(record.proof) == record.proof_content and p.digest(record.proof) == record.digest,
        'proof_changed')
    require(record.state.history[-1].event_id == record.digest
        and record.state.history[-1].kind == 'placement', 'tail_event')
    require(record.state.counter == p.S.color_counts(record.grid)
        and record.action == record.state.action, 'tail_basis')
    require(record.consumed <= frozenset(binding.consumed_tokens), 'consumption_history')
    require(all(value is False for value in (record.current_permission, record.probability_permission,
        record.evaluation_permission, record.physical_certified)), 'permission')


def validate(control: Any, binding: Any, record: Basis) -> Basis:
    require(type(record) is Basis and record.binding is binding
        and getattr(binding, 'private_suffix_basis', None) is record, 'record_identity')
    history(control, binding, record)
    source = record.source
    require(type(source) is SP.Successor and source.binding is binding
        and SP.content(source.proof) == source.sealed == record.source_content, 'successor_proof')
    require(source.queue is record.queue and source.refs[1] is record.head
        and source.tokens[1] == record.token and source.tokens[0] in record.consumed,
        'successor_tail_binding')
    require(record.proof['kind'] == KIND and record.proof['token'] == source.tokens[0]
        and record.proof['inferred_final'] == record.grid
        and record.proof['available_frame'] == record.frame
        and record.proof['available_time'] == record.clock, 'tail_proof')
    return record


def post_commit(control: Any, binding: Any, call: Any) -> Any:
    p, view, prepared = control.inventory, call['view'], call['prepared']
    votes, source, state = prepared['tail_votes'], binding.hidden_tail_successor, binding.owner.state
    require(call['binding'] is binding and call['consumed'] is True
        and getattr(binding, 'hidden_tail_consumed', False) is True, 'uncommitted')
    require(type(source) is SP.Successor and source.state is prepared['old_state'] is votes.state
        and votes.item is source.item and SP.content(source.proof) == source.sealed
        and SP.content(votes.successor_proof) == source.sealed, 'source_commit')
    require(prepared['kind'] == prepared['proof']['kind'] == KIND
        and prepared['proof']['current_permission'] is False
        and prepared['proof']['physical_certified'] is False, 'tail_kind')
    evidence = prepared['evidence']
    require(type(evidence) is p.S.PlacementEvidence and evidence.event_id == p.digest(prepared['proof'])
        and state.history[:-1] == source.state.history
        and state.history[-1].event_id == evidence.event_id
        and state.history[-1].delta == evidence.added and state.action == evidence.action,
        'original_placement')
    require(binding.grid == votes.final == prepared['grid'] and binding.scope == view.scope == source.scope
        and binding.next_token is binding.next_started is binding.candidate is None, 'posttail_state')
    require(tuple(view.queue) == (source.refs[1],) and view.queue[0] is source.refs[1]
        and votes.token in binding.consumed_tokens and source.tokens[1] not in binding.consumed_tokens,
        'posttail_suffix')
    row = control.provider.link.current(view)
    owner = control.provider.journal.fifo.entries[(id(row['pipe']), view.scope[-1])]
    require(owner['queue'] is view.queue is source.queue and tuple(owner['tokens']) == (source.tokens[1],)
        and len(owner['refs']) == SUFFIX_SIZE and owner['refs'][0] is source.refs[1], 'posttail_J')
    return source


def capture(control: Any, binding: Any, call: Any) -> Basis:
    require(getattr(binding, 'private_suffix_basis', None) is None, 'duplicate_capture')
    source = post_commit(control, binding, call)
    p, state, view = control.inventory, binding.owner.state, call['view']
    proof = deepcopy(call['prepared']['proof'])
    policy = logs(binding)
    matching = [row for row in policy if row['purpose'] == 'placement'
        and row['proof_sha'] == p.digest(proof) and SP.content(row['proof']) == SP.content(proof)]
    require(len(matching) == 1, 'placement_policy_log')
    record = Basis(binding, binding.owner, binding.grid, state.action, view.frame, view.clock,
        state, SP.content(asdict(state)), binding.scope, source.refs[1], source.tokens[1], source.queue,
        proof, p.digest(proof), SP.content(proof), source, source.sealed, len(policy),
        SP.content(policy), frozenset(binding.consumed_tokens))
    binding.private_suffix_basis = record
    return validate(control, binding, record)


def current(control: Any, binding: Any, view: Any, record: Basis) -> Any:
    require(view.scope == record.scope and SP.content(view.scope) == SP.content(record.scope)
        and type(view.frame) is int and view.frame > record.frame
        and view.clock == view.frame / FPS, 'view_scope_clock')
    require(len(view.refs) == len(view.tokens) == len(view.queue) == SUFFIX_SIZE
        and view.queue is record.queue and view.refs[0] is view.queue[0] is record.head
        and view.tokens == (record.token,) and view.added == (), 'head_token')
    provider, side = control.provider, view.scope[-1]
    row = provider.link.current(view)
    owner = provider.journal.fifo.entries[(id(row['pipe']), side)]
    scope = provider.journal.scope(row['pipe'], side, view.frame, view.clock)
    provider._parts.V.Provider.check_enqueue(provider, provider.enqueues[side], scope, row['epoch'], owner)
    require(tuple(provider.enqueues[side]['added_occurrence_tokens']) == view.added
        and view.scope[5] == scope['generation']['reset_epoch'], 'current_J_scope')
    require(owner['queue'] is record.queue and tuple(owner['tokens']) == view.tokens
        and len(owner['refs']) == SUFFIX_SIZE and owner['refs'][0] is record.head, 'current_J_owner')
    proof = record.source.proof
    require(row['epoch'] == proof['software_epoch'] and row['segment'] == proof['segment_id']
        and row['accepted'] == view.next_pair == proof['new_accepted']
        and row['dnext'] == view.dnext_pair == proof['dnext'], 'current_NEXT_basis')
    quiet, _ = provider.journal.controller._quiet(row['invocation'], side, view.next_pair)
    require(view.quiet is quiet, 'original_quiet')
    return row


def start(control: Any, binding: Any, view: Any) -> bool:
    record = getattr(binding, 'private_suffix_basis', None)
    if record is None:
        return False
    validate(control, binding, record)
    if record.started is not None and record.token in binding.consumed_tokens:
        require(not view.refs and binding.next_token is None, 'completed_new_head')
        return True
    current(control, binding, view, record)
    if record.started is not None:
        require(binding.next_token == record.token and binding.next_started == record.started
            and binding.owner.state is record.started_state, 'started_state')
        return True
    require(binding.owner.state is record.state and binding.grid == record.grid
        and binding.next_token is binding.next_started is binding.candidate is None, 'not_startable')
    if view.quiet is not True:
        return False
    control._parts.C.H.start(control.inventory, binding, record.token, view.frame, view.clock)
    state = binding.owner.state
    require(state.action == record.action + 1 and state.history == record.state.history
        and state.counter == record.state.counter and binding.next_started == (view.frame, view.clock),
        'original_start_result')
    record.started, record.started_state = binding.next_started, state
    return True
