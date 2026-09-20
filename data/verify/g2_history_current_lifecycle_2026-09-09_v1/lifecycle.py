"""既存履歴prepareのcurrent前提だけを連続復帰に対応させる。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import history_state as H

OLD_SHA = '9d66ab677dea73d596fc33623d67fbcbd111ba92e86147de5dc5b61c63392114'


def compatible_current(state: Any, binding: Any) -> bool:
    current = state.current
    return current is None or (current.action <= state.action and current.grid == binding.current)


def prepare(p: Any, binding: Any, view: Any, grid: Any, proof: dict[str, Any]) -> Any:
    H.require(hashlib.sha256(Path(H.__file__).read_bytes()).hexdigest() == OLD_SHA, 'fixed_history_state')
    state = binding.owner.state
    H.require(binding.scope == view.scope and compatible_current(state, binding), 'history_scope_or_current')
    H.require(binding.next_token == proof['token'] == view.tokens[0], 'history_token')
    H.require(proof['available_frame'] == view.frame and proof['available_time'] == view.clock,
              'history_available')
    H.require(binding.clear_first is not None and proof['occurred'] == binding.clear_first,
              'history_observed')
    added = tuple(a - b for a, b in zip(p.S.color_counts(grid), state.counter))
    H.require(all(value >= 0 for value in added) and sum(added) == H.PAIR_SIZE, 'history_delta')
    bound = dict(proof, scope=H.asdict(state.scope), before_grid=binding.grid, grid=grid)
    occurred = H.clock(p, *binding.clear_first)
    now = H.clock(p, view.frame, view.clock, H.PLACEMENT_OPERATION)
    evidence = p.S.PlacementEvidence(state.scope, p.digest(bound), state.action, now, added, occurred)
    H.require(not state.debts and not state.origins, 'unregistered_or_unsettled_origin')
    H.require(not any(entry.action == state.action or entry.event_id == evidence.event_id
                    for entry in state.history), 'history_duplicate')
    p.S.available(state.action_since, occurred)
    p.S.available(occurred, now)
    return {'evidence': evidence, 'proof': bound, 'old_state': state, 'grid': grid}
