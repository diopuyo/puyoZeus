"""原消費・原S記帳成功後だけ次tailのprefix来歴を更新する。"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
import rolling_prepare as R

PREFIX_KIND = 'hidden_two_hand_prefix_history/v1'
SUPPORT_KEYS = frozenset(('prefix', 'final', 'raw', 'pairs', 'physical_certified',
    'probability_assigned', 'accounting_permission', 'current_permission'))
FLAGS = ('physical_certified', 'probability_assigned', 'accounting_permission', 'current_permission')
RAW_KEYS = frozenset(('captured_frame', 'raw_grid'))


def same(control: Any, require: Any, left: Any, right: Any, reason: str) -> None:
    """既存digestで全内容を比較する。bool/数値や未知追加キーも同一視しない。"""
    require(control.inventory.digest(left) == control.inventory.digest(right), reason)


def committed_proof(provenance: Any, control: Any, binding: Any, votes: Any) -> Any:
    """原検査成功済みeventから全policy proofを取り出す。出所を新しく作らない。"""
    provenance.validate(control, binding, votes)
    entries = [entry for entry in binding.owner.state.history if entry.kind == 'placement'
        and entry.action == votes.held.state.action and entry.available_at.frame == votes.last[0]]
    provenance.require(len(entries) == 1, 'rolling_policy_event')
    accounting = getattr(binding.policy, 'accounting', binding.policy)
    rows = [row for row in accounting.proofs if row['purpose'] == 'placement'
        and row['proof_sha'] == entries[0].event_id]
    provenance.require(len(rows) == 1 and type(rows[0]['proof']) is dict, 'rolling_policy_source')
    proof = rows[0]['proof']
    provenance.require(control.inventory.digest(proof) == entries[0].event_id, 'rolling_policy_digest')
    return proof


def bound_votes(provenance: Any, control: Any, votes: Any, proof: Any) -> None:
    require, held = provenance.require, votes.held
    support = asdict(votes.support)
    require(type(proof) is dict and proof['kind'] == PREFIX_KIND, 'rolling_proof_kind')
    require(set(support) == SUPPORT_KEYS and type(proof['inferred_path']) is dict
        and set(proof['inferred_path']) == SUPPORT_KEYS, 'rolling_support_fields')
    require(all(support[key] is False for key in FLAGS), 'rolling_support_permissions')
    same(control, require, proof['inferred_path'], support, 'rolling_support_changed')
    same(control, require, proof['directional_commit'], held.proof, 'rolling_directional_source')
    require(control.inventory.digest(held.proof) == held.proof_digest, 'rolling_holder_digest')
    require(type(votes.count) is int and votes.count >= 2
        and type(proof['clear_observations']) is int, 'rolling_vote_count_type')
    expected = dict(occurred=votes.first, clear_last=votes.last, clear_observations=votes.count,
        available_frame=votes.last[0], available_time=votes.last[1], token=held.tokens[0],
        new_token=held.tokens[1], pair=held.refs[0], before_grid=held.item.baseline,
        grid=votes.support.prefix, observed_raw=votes.support.raw, scope=asdict(held.state.scope))
    for key, value in expected.items():
        same(control, require, proof[key], value, 'rolling_vote_bound:' + key)
    require(held.view is not None and (held.view.frame, held.view.clock) == votes.last,
        'rolling_vote_view')
    require(proof['available_window'] is False and proof['intermediate_grid_is_observed'] is False
        and proof['physical_certified'] is False and proof['current_permission'] is False, 'rolling_permissions')
    raw = proof['raw_capture']
    require(type(raw) is dict and set(raw) in (RAW_KEYS, RAW_KEYS | {'capture'}), 'rolling_raw_fields')
    same(control, require, raw['raw_grid'], votes.support.raw, 'rolling_capture_raw')
    same(control, require, raw['captured_frame'], votes.last[0], 'rolling_capture_clock')


def before(provenance: Any, control: Any, call: Any) -> tuple[Any, Any, Any]:
    binding, prepared = call['binding'], call['prepared']
    old, new = prepared['rolling_previous_votes'], prepared['hidden_prefix_votes']
    require = provenance.require
    provenance.validate(control, binding, old)
    require(binding.hidden_prefix_votes is old and getattr(binding, R.VOTES) is new,
        'rolling_pending_identity')
    require(new.binding is binding and new.held.binding is binding and not new.held.consumed,
        'rolling_new_holder')
    require(new.held.item.baseline == old.support.prefix == binding.grid, 'rolling_grid_link')
    require(new.held.tokens[0] == old.held.tokens[1] == binding.next_token
        and new.held.refs[0] is old.held.refs[1], 'rolling_token_link')
    require(new.held.state is binding.owner.state and prepared['old_state'] is binding.owner.state,
        'rolling_before_state')
    require(new.held.view is call['view'] and call['deferred_handoff'] is new.held,
        'rolling_call_holder')
    bound_votes(provenance, control, old, committed_proof(provenance, control, binding, old))
    evidence = prepared['evidence']
    require(type(evidence) is control.inventory.S.PlacementEvidence
        and evidence.event_id == control.inventory.digest(prepared['proof']), 'rolling_prepared_evidence_digest')
    bound_votes(provenance, control, new, prepared['proof'])
    same(control, require, prepared['grid'], new.support.prefix, 'rolling_prepared_grid')
    return old, new, binding.owner.state


def consumed(original: Any, provenance: Any, control: Any, call: Any, caller: Any) -> None:
    old, new, state = before(provenance, control, call)
    binding, current = call['binding'], call['binding'].current
    original(control, call, caller)
    require = provenance.require
    require(call['consumed'] and new.held.consumed and binding.current == current,
        'rolling_original_consumption')
    require(binding.owner.state is not state and binding.owner.state.action == state.action+1,
        'rolling_original_next_action')
    require(binding.grid == new.support.prefix and len(call['view'].queue) == 1
        and call['view'].queue[0] is new.held.refs[1], 'rolling_native_result')
    require(binding.hidden_prefix_votes is old, 'rolling_premature_promotion')
    bound_votes(provenance, control, old, committed_proof(provenance, control, binding, old))
    bound_votes(provenance, control, new, committed_proof(provenance, control, binding, new))
    archive = getattr(binding, 'hidden_rolling_prefix_archive', ())
    require(type(archive) is tuple and all(item is not old for item in archive), 'rolling_archive')
    binding.hidden_rolling_prefix_archive = (*archive, old)
    binding.hidden_prefix_votes = new
    setattr(binding, R.VOTES, None)
