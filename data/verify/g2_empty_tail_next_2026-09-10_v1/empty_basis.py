"""原tail空FIFOと後発の原H.startを結ぶ私有基点。評価・発火権限は持たない。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import marshal
from pathlib import Path
import sys
from types import CodeType
from typing import Any

VERIFY = Path(__file__).resolve().parent.parent
TAIL = VERIFY / 'g2_hidden_tail_candidate_2026-09-10_v1'
SUFFIX = VERIFY / 'g2_hidden_tail_suffix_candidate_2026-09-10_v1'
HISTORY = VERIFY / 'g2_historical_completion_runtime_2026-09-09_v1'
PINS = {
    TAIL/'prefix_provenance.py': '09138159c4bb65b395ccc64cf782411fed950db418ef25d8397d4864adab39eb',
    TAIL/'tail_witness.py': '0aebf6bccb89eff1ae6fc5c8053b04ed02c0fbd11e346d750cf0274564828174',
    SUFFIX/'tail_suffix_provenance.py': '654d85f78d851fbbde5886d3a188cdca12fc5c89cebad047848f9379d7a75100',
    HISTORY/'history_state.py': '9d66ab677dea73d596fc33623d67fbcbd111ba92e86147de5dc5b61c63392114',
    SUFFIX/'install.py': '63140e945b76d665b5fba1febe9331d72ec12e7ae8099dca8b1cf559b3c82d6d',
    SUFFIX/'source.py': '09cdcdd5b46095eceda308d0f94d00ac37baf784e87b36512420dbc755f40aaa',
    SUFFIX/'votes.py': 'b9fa5d651986cd1a3a348be90419c0060f2eedd4169747924e15b85718bf3047',
}
KIND, FPS, REQUIRED_VOTES = 'hidden_single_tail_history/v1', 60, 2
CODE_FORMAT = 2  # 共有参照表に依存せず、元CodeType全内容を照合する。


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('empty_tail_' + reason)


def load(name: str, path: Path) -> Any:
    require(hashlib.sha256(path.read_bytes()).hexdigest() == PINS[path], 'source_changed')
    spec = importlib.util.spec_from_file_location(__name__ + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V = load('_prefix', TAIL/'prefix_provenance.py')
SP = load('_content', SUFFIX/'tail_suffix_provenance.py')


def pin(module: Any, name: str, path: Path, function: Any = None) -> None:
    source, function = path.read_bytes(), function or getattr(module, name)
    require(Path(module.__file__).resolve() == path and hashlib.sha256(source).hexdigest() == PINS[path],
        'source_changed')
    require(function.__globals__ is vars(module) and function.__closure__ is None, 'function_globals')
    code = compile(source, function.__code__.co_filename, 'exec', dont_inherit=True)
    expected = [c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == name]
    require(len(expected) == 1 and marshal.dumps(expected[0], CODE_FORMAT) == marshal.dumps(function.__code__, CODE_FORMAT),
        'function_code')


def witness_original(module: Any, name: str) -> Any:
    function = getattr(module, name)
    if function.__closure__ is None:
        return function
    path = SUFFIX/'install.py'
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == PINS[path]
        and Path(function.__globals__['__file__']).resolve() == path, 'wrapper_source')
    original_module = sys.modules[function.__globals__['__name__']]
    pin(original_module, 'install', path)
    require(function.__globals__ is vars(original_module), 'wrapper_globals')
    code = compile(source, function.__code__.co_filename, 'exec', dont_inherit=True)
    install = next(c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == 'install')
    expected = next(c for c in install.co_consts if isinstance(c, CodeType) and c.co_name == name)
    require(marshal.dumps(function.__code__, CODE_FORMAT) == marshal.dumps(expected, CODE_FORMAT), 'wrapper_code')
    free = dict(zip(function.__code__.co_freevars, (c.cell_contents for c in function.__closure__)))
    require(set(free) == {'old_' + name}, 'wrapper_closure')
    if name == 'source':
        proxy = original_module.S
        for member in ('source', 'owner'):
            pin(proxy, member, SUFFIX/'source.py')
        for member in ('content', 'item_content', 'capture', 'check', 'ready'):
            pin(proxy.SP, member, SUFFIX/'tail_suffix_provenance.py')
    else:
        for member in ('vote', 'matches'):
            pin(original_module.V, member, SUFFIX/'votes.py')
    return free['old_' + name]


def origin(control: Any, votes: Any) -> None:
    for name in ('require', 'contents', 'identity', 'clocks', 'committed', 'validate'):
        pin(V, name, TAIL/'prefix_provenance.py')
    pin(SP, 'content', SUFFIX/'tail_suffix_provenance.py')
    module = sys.modules[type(votes).__module__]
    require(type(votes) is module.TailVotes, 'votes_type')
    for name in ('source', 'vote'):
        pin(module, name, TAIL/'tail_witness.py', witness_original(module, name))
    for name in ('start', 'clock', 'require'):
        pin(control._parts.C.H, name, HISTORY/'history_state.py')


def logs(binding: Any) -> list[Any]:
    return getattr(binding.policy, 'accounting', binding.policy).proofs


def source_value(control: Any, binding: Any, previous: Any) -> Any:
    source = V.validate(control, binding, previous)
    held = previous.held
    entries = [r for r in logs(binding) if r['purpose'] == 'placement'
        and r['proof']['kind'] == 'hidden_two_hand_prefix_history/v1'
        and r['proof']['token'] == held.tokens[0]]
    require(len(entries) == 1, 'prefix_policy')
    require(SP.content(entries[0]['proof']['inferred_path']) == SP.content(asdict(previous.support)),
        'support_policy')
    return SP.content((source, asdict(previous.support), previous.first, previous.last, previous.count))


def vote_value(votes: Any) -> Any:
    return SP.content((votes.scope, votes.token, votes.final, votes.raw, votes.raw_proof,
        votes.first, votes.last, votes.count, votes.item.baseline, votes.item.next_pair,
        votes.item.dnext_pair, votes.item.started))


@dataclass
class Basis:
    binding: Any
    owner: Any
    grid: Any
    action: int
    frame: int
    clock: float
    state: Any
    scope: Any
    queue: Any
    proof: Any
    digest: str
    state_content: Any
    proof_content: Any
    policy_content: Any
    policy_count: int
    source: Any
    source_content: Any
    votes: Any
    votes_content: Any
    old_current: Any
    current_content: Any
    consumed: frozenset[str]
    journal_content: Any
    head: Any = None
    token: str | None = None
    started: Any = None
    started_state: Any = None
    started_content: Any = None
    item: Any = None
    bound_content: Any = None
    current_permission: bool = False
    probability_permission: bool = False
    evaluation_permission: bool = False
    physical_certified: bool = False
    firing_permission: bool = False


def history(control: Any, binding: Any, record: Basis) -> None:
    p, state = control.inventory, binding.owner.state
    require(record.owner is binding.owner and SP.content(record.scope) == SP.content(binding.scope), 'scope')
    require(SP.content(asdict(record.state)) == record.state_content, 'old_state')
    require(state.scope == record.state.scope and state.action >= record.action
        and state.history[:len(record.state.history)] == record.state.history, 'history_prefix')
    require(not state.origins and not state.debts and not state.consumed_ids, 'origin')
    p.S.validate_accounting(state)
    p.S.validate_grid(binding.grid)
    require(state.counter == p.S.color_counts(binding.grid), 'counter_grid')
    require(SP.content(logs(binding)[:record.policy_count]) == record.policy_content, 'policy_prefix')
    require(SP.content(record.proof) == record.proof_content and p.digest(record.proof) == record.digest,
        'tail_proof_changed')
    require(record.state.history[-1].event_id == record.digest and record.state.history[-1].kind == 'placement'
        and record.state.counter == p.S.color_counts(record.grid) and record.action == record.state.action,
        'tail_history')
    require(SP.content(record.old_current) == record.current_content
        and state.current == record.state.current, 'old_integer_slot')
    require(record.consumed <= frozenset(binding.consumed_tokens), 'consumed_history')
    require(all(x is False for x in (record.current_permission, record.probability_permission,
        record.evaluation_permission, record.physical_certified, record.firing_permission)), 'permission')


def validate(control: Any, binding: Any, record: Basis) -> Basis:
    require(type(record) is Basis and record.binding is binding
        and getattr(binding, 'empty_tail_basis', None) is record, 'record_identity')
    origin(control, record.votes)
    history(control, binding, record)
    require(source_value(control, binding, record.source) == record.source_content, 'prefix_changed')
    votes = record.votes
    require(votes.binding is binding and votes.queue is record.queue
        and votes.item.queue is record.queue and votes.item.pair is votes.head
        and vote_value(votes) == record.votes_content, 'tail_votes')
    proof = record.proof
    require(proof['kind'] == KIND and proof['token'] == votes.token and votes.token in record.consumed
        and proof['inferred_final'] == record.grid == votes.final
        and proof['available_frame'] == record.frame and proof['available_time'] == record.clock,
        'tail_binding')
    if record.started is None:
        require(record.head is record.token is record.item is record.started_state is None, 'future_head')
    else:
        require(record.item.pair is record.head and record.item.queue is record.queue
            and SP.content((record.token, record.started, record.item.token, record.item.scope,
                record.item.baseline, record.item.next_pair, record.item.dnext_pair, record.item.started))
                == record.bound_content, 'bound_head')
        require(SP.content(asdict(record.started_state)) == record.started_content, 'started_changed')
    return record


def post_journal(control: Any, binding: Any, call: Any, caller: Any) -> Any:
    rec, view, values = control.provider.journal, call['view'], caller.f_locals
    item = rec.active
    require(call.get('frame') is caller and control.calls.get(id(caller)) is call
        and item is not None and item['frame'] is caller and caller.f_code in rec.codes, 'actual_caller')
    pipe, side = values['self'], values['side']
    scope = rec.scope(pipe, side, view.frame, view.clock)
    require(item['pipe'] is pipe and item['scope'] == scope and item['epoch'] == view.scope[2]
        and scope['source_id'] == view.scope[0] and scope['run_id'] == view.scope[1]
        and id(pipe) == view.scope[3] and id(values['sm']) == view.scope[4]
        and scope['generation']['reset_epoch'] == view.scope[5] and side == view.scope[-1], 'journal_scope')
    events = [row for row in item['events'] if row['stage'] == 'fifo_after']
    require(len(events) == 1 and events[0]['enqueue_occurrence_token'] == view.tokens[0]
        and not rec.errors, 'journal_pop')
    provider = control.provider
    owner = provider._parts.V.Provider.owner(provider, pipe, side, view.scope[2])
    require(owner['queue'] is view.queue and not owner['refs'] and not owner['tokens']
        and not view.queue, 'tail_not_empty')
    control.unchanged(call, pipe, side)
    require(values['committed'] is call['prepared']['tail_votes'].head
        and control._parts.T.board_key(values['sm'].context.confirmed_board) == binding.current,
        'tail_native_current')
    return SP.content((item['token'], scope, events))


def post_state(control: Any, binding: Any, call: Any) -> Any:
    p, prepared, view = control.inventory, call['prepared'], call['view']
    votes, state = prepared['tail_votes'], binding.owner.state
    origin(control, votes)
    require(call['binding'] is binding and call['consumed'] is True
        and binding.hidden_tail_consumed is True, 'uncommitted')
    require(prepared['kind'] == prepared['proof']['kind'] == KIND
        and prepared['proof']['current_permission'] is False
        and prepared['proof']['physical_certified'] is False, 'tail_kind')
    require(len(view.refs) == len(view.tokens) == 1 and view.refs[0] is votes.head
        and view.tokens == (votes.token,) and not view.added, 'tail_view')
    require(votes.binding is binding and votes.state is prepared['old_state']
        and votes.scope == view.scope == binding.scope and votes.last == (view.frame, view.clock)
        and type(votes.count) is int and votes.count >= REQUIRED_VOTES, 'tail_vote_state')
    proof = prepared['proof']
    require(SP.content((proof['occurred'], proof['clear_last'], proof['clear_observations'],
        proof['observed_raw'], proof['raw_capture'])) == SP.content((votes.first, votes.last,
        votes.count, votes.raw, votes.raw_proof)), 'tail_vote_proof')
    evidence = prepared['evidence']
    require(type(evidence) is p.S.PlacementEvidence and evidence.event_id == p.digest(proof)
        and state.history[:-1] == votes.state.history and state.history[-1].event_id == evidence.event_id
        and state.history[-1].delta == evidence.added and state.action == evidence.action, 'original_placement')
    require(binding.grid == votes.final == prepared['grid']
        and binding.next_token is binding.next_started is binding.candidate is None
        and votes.token in binding.consumed_tokens, 'posttail_state')
    matched = [row for row in logs(binding) if row['purpose'] == 'placement'
        and row['proof_sha'] == evidence.event_id and SP.content(row['proof']) == SP.content(proof)]
    require(len(matched) == 1, 'tail_policy')
    return votes


def capture(control: Any, binding: Any, call: Any, caller: Any) -> Basis:
    require(getattr(binding, 'empty_tail_basis', None) is None, 'duplicate_capture')
    require(getattr(binding, 'private_suffix_basis', None) is None, 'different_basis')
    votes = post_state(control, binding, call)
    journal = post_journal(control, binding, call, caller)
    previous, state, view = binding.hidden_prefix_votes, binding.owner.state, call['view']
    proof, policy = deepcopy(call['prepared']['proof']), logs(binding)
    source = source_value(control, binding, previous)
    require(SP.content(proof['prefix_source']) == SP.content(V.validate(control, binding, previous))
        and votes.final == previous.support.final, 'tail_prefix_link')
    record = Basis(binding, binding.owner, binding.grid, state.action, view.frame, view.clock,
        state, tuple(binding.scope), view.queue, proof, control.inventory.digest(proof),
        SP.content(asdict(state)), SP.content(proof), SP.content(policy), len(policy), previous, source,
        votes, vote_value(votes), binding.current, SP.content(binding.current),
        frozenset(binding.consumed_tokens), journal)
    binding.empty_tail_basis = record
    return validate(control, binding, record)


def current_journal(control: Any, binding: Any, view: Any, item: Any, record: Basis) -> Any:
    require(SP.content(view.scope) == SP.content(record.scope) and type(view.frame) is int
        and view.frame > record.frame and view.clock == view.frame / FPS, 'head_scope_clock')
    require(len(view.refs) == len(view.tokens) == len(view.queue) == 1
        and view.queue is record.queue is item.queue and view.refs[0] is view.queue[0] is item.pair
        and view.tokens == (item.token,) and item is binding.candidate, 'head_reference')
    provider, side = control.provider, view.scope[-1]
    row = provider.link.current(view)
    owner = provider._parts.V.Provider.owner(provider, row['pipe'], side, view.scope[2])
    scope = provider.journal.scope(row['pipe'], side, view.frame, view.clock)
    provider._parts.V.Provider.check_enqueue(provider, provider.enqueues[side], scope, row['epoch'], owner)
    require(tuple(provider.enqueues[side]['added_occurrence_tokens']) == view.added
        and owner['queue'] is record.queue and tuple(owner['tokens']) == view.tokens
        and len(owner['refs']) == 1 and owner['refs'][0] is item.pair, 'head_J')
    require(scope['source_id'] == view.scope[0] and scope['run_id'] == view.scope[1]
        and row['epoch'] == view.scope[2] and id(row['pipe']) == view.scope[3]
        and id(getattr(row['pipe'], '_sm_' + side.lower())) == view.scope[4]
        and scope['generation']['reset_epoch'] == view.scope[5], 'head_J_scope')
    require(row['accepted'] == view.next_pair == item.next_pair
        and row['dnext'] == view.dnext_pair == item.dnext_pair, 'head_NEXT')
    quiet, _ = provider.journal.controller._quiet(row['invocation'], side, view.next_pair)
    require(view.quiet is quiet is True and side not in provider.handoff_proofs, 'head_quiet')
    return row


def bind_head(control: Any, binding: Any, view: Any, item: Any) -> Basis:
    record = validate(control, binding, binding.empty_tail_basis)
    current_journal(control, binding, view, item, record)
    state, p = binding.owner.state, control.inventory
    require(binding.grid == record.grid and SP.content(binding.current) == record.current_content
        and item.baseline == record.grid and item.scope == record.scope, 'start_grid_current')
    require(state.action == record.action + 1 and state.history == record.state.history
        and state.counter == record.state.counter and state.current == record.state.current
        and binding.next_token == item.token and item.token not in binding.consumed_tokens, 'start_state')
    if record.started is not None:
        require(item is record.item and item.pair is record.head and item.token == record.token
            and binding.next_started == record.started and state is record.started_state
            and view.added == (() if view.frame > record.started[0] else (record.token,)), 'head_rebind')
        return record
    require(view.added == (item.token,) and binding.next_started == (view.frame, view.clock)
        and item.started == view.frame and item.token not in record.consumed, 'new_original_head')
    expected = control._parts.C.H.clock(p, view.frame, view.clock, control._parts.C.H.NEXT_OPERATION)
    require(state.action_since == state.clock == expected and type(state.action) is int, 'start_clock')
    record.head, record.token, record.item = item.pair, item.token, item
    record.started, record.started_state = binding.next_started, state
    record.started_content = SP.content(asdict(state))
    record.bound_content = SP.content((record.token, record.started, item.token, item.scope,
        item.baseline, item.next_pair, item.dnext_pair, item.started))
    return validate(control, binding, record)
