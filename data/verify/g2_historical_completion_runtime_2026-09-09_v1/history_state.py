"""既存SplitOwnerへの接続参照。新しい在庫CounterやFIFOは作らない。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

OPERATIONS_PER_FRAME = 4
BASELINE_OPERATION, PLACEMENT_OPERATION, NEXT_OPERATION = 0, 1, 2
PAIR_SIZE, FIRST_GENERATION = 2, 1


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def clock(p: Any, frame: int, time_sec: float, operation: int = 0) -> Any:
    require(type(frame) is int and frame >= 0 and type(operation) is int
            and 0 <= operation < OPERATIONS_PER_FRAME, 'history_clock')
    return p.S.Clock(frame, time_sec, frame * OPERATIONS_PER_FRAME + operation)


@dataclass
class Binding:
    """履歴格子と実入力の参照だけを持つ。会計はowner.stateが唯一の正本。"""
    owner: Any
    policy: Any
    grid: Any
    current: Any
    scope: tuple[Any, ...]
    next_token: str | None = None
    next_started: tuple[int, float] | None = None
    candidate: Any = None
    clear_grid: Any = None
    clear_first: tuple[int, float] | None = None
    clear_last: tuple[int, float] | None = None
    clear_count: int = 0
    consumed_tokens: set[str] = field(default_factory=set)
    phase: str = 'historical_owned'


def establish(p: Any, view: Any, grid: Any, proof: dict[str, Any]) -> Binding:
    """呼出側が実STABLE/生raw/未登録originを照合した同callだけに使う。"""
    require(proof['kind'] == 'live_history_baseline' and proof['frame'] == view.frame
            and proof['time_sec'] == view.clock and proof['grid'] == grid, 'baseline_proof_binding')
    source, run, reset = view.scope[:3]
    scope = p.S.Scope(source.removeprefix('sha256:'), run, view.scope[-1],
        'live-private-NEXT-interval:' + p.digest([view.scope, view.frame]), reset)
    policy = p.BoundPolicy()
    owner = p.S.SplitOwner(scope, policy)
    now = clock(p, view.frame, view.clock, BASELINE_OPERATION)
    bound = dict(proof, scope=asdict(scope))
    evidence = p.S.BaselineEvidence(scope, now, p.S.color_counts(grid), p.digest(bound), now)
    policy.arm('baseline', evidence, owner.state, bound)
    owner.establish_baseline(evidence, now)
    return Binding(owner, policy, grid, grid, view.scope)


def start(p: Any, binding: Binding, token: str, frame: int, time_sec: float) -> None:
    require(binding.next_token is None and type(token) is str and bool(token)
            and token not in binding.consumed_tokens, 'hand_already_started_or_consumed')
    action = binding.owner.state.action + FIRST_GENERATION
    binding.owner.advance_action(action, clock(p, frame, time_sec, NEXT_OPERATION))
    binding.next_token, binding.next_started = token, (frame, time_sec)


def prepare(p: Any, binding: Binding, view: Any, grid: Any, proof: dict[str, Any]) -> Any:
    state = binding.owner.state
    require(binding.scope == view.scope and state.current is None, 'history_scope_or_current')
    require(binding.next_token == proof['token'] == view.tokens[0], 'history_token')
    require(proof['available_frame'] == view.frame and proof['available_time'] == view.clock,
            'history_available')
    require(binding.clear_first is not None and proof['occurred'] == binding.clear_first,
            'history_observed')
    added = tuple(a - b for a, b in zip(p.S.color_counts(grid), state.counter))
    require(all(value >= 0 for value in added) and sum(added) == PAIR_SIZE, 'history_delta')
    bound = dict(proof, scope=asdict(state.scope), before_grid=binding.grid, grid=grid)
    occurred = clock(p, *binding.clear_first)
    now = clock(p, view.frame, view.clock, PLACEMENT_OPERATION)
    evidence = p.S.PlacementEvidence(state.scope, p.digest(bound), state.action, now, added, occurred)
    require(not state.debts and not state.origins, 'unregistered_or_unsettled_origin')
    require(not any(entry.action == state.action or entry.event_id == evidence.event_id
                    for entry in state.history), 'history_duplicate')
    p.S.available(state.action_since, occurred)
    p.S.available(occurred, now)
    return {'evidence': evidence, 'proof': bound, 'old_state': state, 'grid': grid}


def commit(p: Any, binding: Binding, prepared: Any, new_token: str, view: Any) -> None:
    """native pop確認後に実行。途中例外は呼出側がfail-stopし巻き戻さない。"""
    require(binding.owner.state is prepared['old_state'], 'history_state_changed_after_prepare')
    evidence, proof = prepared['evidence'], prepared['proof']
    binding.policy.arm('placement', evidence, binding.owner.state, proof)
    binding.owner.add_placement(evidence, evidence.available_at)
    binding.consumed_tokens.add(binding.next_token)
    binding.grid = prepared['grid']
    binding.next_token, binding.next_started, binding.candidate = None, None, None
    binding.clear_grid = binding.clear_first = binding.clear_last = None
    binding.clear_count = 0
    start(p, binding, new_token, view.frame, view.clock)
