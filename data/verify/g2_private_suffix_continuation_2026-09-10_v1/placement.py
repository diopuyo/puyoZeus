"""評価current未生成でも、検証済み私有配置基点から原単head配置を検証する。"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
import continuation_history as H

P, W, T = H.P, H.W, H.T
KIND = 'hidden_private_suffix_history/v1'


def source(basis: Any, control: Any, binding: Any, item: Any, view: Any) -> Any:
    record = basis.validate(control, binding, binding.private_suffix_basis)
    P.require(view.scope == binding.scope == item.scope == record.scope, 'private_suffix_scope')
    P.require(len(view.refs) == len(view.tokens) == len(view.queue) == 1, 'private_suffix_slots')
    P.require(view.queue is record.queue is item.queue and view.refs[0] is record.head is item.pair
        and view.queue[0] is item.pair and view.tokens[0] == record.token == item.token
        and item.token == binding.next_token and item.token not in binding.consumed_tokens,
        'private_suffix_head')
    P.require(item.baseline == binding.grid == record.grid
        and binding.owner.state.counter == control.inventory.S.color_counts(record.grid), 'private_suffix_grid')
    P.require(binding.owner.state.action == record.action + 1 and binding.next_started is not None
        and item.started == binding.next_started[0] > record.frame, 'private_suffix_action')
    P.require(view.scope[-1] not in control.provider.handoff_proofs, 'private_suffix_stale_handoff')
    row = control.provider.link.current(view)
    owner = control.provider.journal.fifo.entries[(id(row['pipe']), view.scope[-1])]
    P.require(owner['queue'] is view.queue and tuple(owner['tokens']) == view.tokens
        and len(owner['refs']) == 1 and owner['refs'][0] is item.pair, 'private_suffix_J')
    if not (view.frame > item.started and view.quiet is True and not view.added
        and row['accepted'] == view.next_pair == item.next_pair
        and row['dnext'] == view.dnext_pair == item.dnext_pair):
        return None
    return record


def prepare(basis: Any, lifecycle: Any, core: Any, control: Any, binding: Any,
            item: Any, sm: Any, signals: Any, view: Any) -> Any:
    record = source(basis, control, binding, item, view)
    captured = W.observed(control, binding, signals, view)
    if not (record is not None and captured is not None and signals.is_match_active is True
        and signals.chain_event is None and signals.effect_gate_window_active is False):
        binding.hidden_continuation_votes = None
        return None
    raw, raw_proof = captured
    possible = tuple(grid for grid in P.options(core, binding.grid, P.pair(item.pair)) if P.compatible(raw, grid))
    if len(possible) != 1 or core.simulate_chain(P.board(core, possible[0])).chain_count:
        binding.hidden_continuation_votes = None
        return None
    votes = T.vote(control, binding, item, possible[0], raw, raw_proof, view)
    if votes.count < W.MIN_VOTES:
        return None
    binding.clear_first, binding.clear_last = votes.first, votes.last
    binding.clear_grid, binding.clear_count = votes.final, votes.count
    proof = dict(kind=KIND, token=item.token, pair=item.pair, occurred=votes.first,
        available_frame=view.frame, available_time=view.clock, clear_last=votes.last,
        clear_observations=votes.count, available_window=False, new_token=None,
        observed_raw=raw, raw_capture=raw_proof, inferred_final=votes.final,
        previous_private_placement=record.digest, previous_private_frame=record.frame,
        prefix_source=T.V.validate(control, binding, binding.hidden_prefix_votes),
        inferred_grid_is_observed=False, physical_certified=False, current_permission=False)
    value = lifecycle.prepare(control.inventory, binding, view, votes.final, proof)
    P.require(value is not None, 'private_suffix_lifecycle')
    return value | dict(kind=KIND, continuation_votes=votes, private_basis=record)


def committed(basis: Any, control: Any, binding: Any) -> Any:
    saved = binding.private_suffix_placement
    previous = basis.validate(control, binding, saved['basis'])
    state, proof, votes = binding.owner.state, saved['proof'], saved['votes']
    entries = [entry for entry in state.history if entry.kind == 'placement' and entry.action == votes.state.action]
    P.require(len(entries) == 1 and entries[0].event_id == control.inventory.digest(proof), 'private_suffix_event')
    policy = getattr(binding.policy, 'accounting', binding.policy)
    logs = [row for row in policy.proofs if row['purpose'] == 'placement' and row['proof_sha'] == entries[0].event_id]
    P.require(len(logs) == 1 and control.inventory.digest(logs[0]['proof']) == entries[0].event_id,
        'private_suffix_policy')
    P.require(proof['kind'] == KIND and proof['scope'] == asdict(state.scope)
        and proof['previous_private_placement'] == previous.digest and proof['before_grid'] == previous.grid,
        'private_suffix_previous')
    P.require(proof['grid'] == binding.grid == votes.final and proof['token'] == previous.token
        and proof['token'] in binding.consumed_tokens and votes.head is previous.head,
        'private_suffix_committed_grid_token')
    P.require(state.action == votes.state.action == previous.action + 1
        and proof['available_frame'] == votes.last[0] == entries[0].available_at.frame
        and proof['available_time'] == votes.last[1] == entries[0].available_at.time_sec,
        'private_suffix_committed_clock')
    return T.V.deepcopy(proof)


def consumed(basis: Any, control: Any, call: Any, caller: Any, rows: list[Any]) -> None:
    binding, view, prepared = call['binding'], call['view'], call['prepared']
    values, votes = caller.f_locals, prepared['continuation_votes']
    record = basis.validate(control, binding, prepared['private_basis'])
    state, current = binding.owner.state, binding.current
    P.require(state is prepared['old_state'] is votes.state, 'private_suffix_consume_state')
    P.require(values['committed'] is votes.head is record.head and len(view.queue) == 0
        and votes.token == record.token, 'private_suffix_native_pop')
    control.unchanged(call, values['self'], values['side'])
    control.provider._parts.V.Provider.after_history_consume(control.provider, call, caller)
    binding.policy.arm('placement', prepared['evidence'], state, prepared['proof'])
    binding.owner.add_placement(prepared['evidence'], prepared['evidence'].available_at)
    binding.grid = votes.final
    binding.consumed_tokens.add(votes.token)
    binding.next_token = binding.next_started = binding.candidate = None
    binding.clear_grid = binding.clear_first = binding.clear_last = None
    binding.clear_count, call['consumed'] = 0, True
    binding.private_suffix_placement = dict(basis=record, votes=votes, proof=T.V.deepcopy(prepared['proof']))
    binding.hidden_continuation_votes = None
    control.unchanged(call, values['self'], values['side'])
    P.require(control._parts.T.board_key(values['sm'].context.confirmed_board) == binding.current == current,
        'private_suffix_current_changed')
    P.require(not binding.owner.state.origins and not binding.owner.state.debts, 'private_suffix_origin')
    committed(basis, control, binding)
    rows.append(dict(kind=KIND, frame=view.frame, current_permission=False, source=prepared['proof'],
        state=asdict(binding.owner.state), old_current=current, inferred_final=binding.grid,
        native_counter_unchanged=True, physical_certified=False, next_action_created=False))
