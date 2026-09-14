"""式より前の原raw/headを発火手の予測資格に束縛する。確定公開は行わない。"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import sys
from types import FunctionType, MethodType
from typing import Any
import front_probe as F

SIDE, FPS, PAIR_SIZE, PLACEMENT_OPERATION = '1P', 60, 2, 1


@dataclass
class Ticket:
    binding: Any
    state: Any
    scope: Any
    queue: Any
    head: Any
    token: str
    source: dict[str, Any]
    grid: Any
    before: Any
    prediction: Any
    environment: Any
    event: Any = None
    origin: Any = None
    consumed: bool = False


def qualified(frame: Any, factory: Any, environment: Any) -> Ticket:
    pipe, provider = frame.f_locals['self'], factory.provider
    binding = factory.controller.history[SIDE]
    def inferred(values: Any, current: Any, unused: Any, *args: Any) -> Any:
        return F.infer(values, current, environment, *args)
    capture = FunctionType(F.capture.__code__, dict(vars(F), infer=inferred))
    source = capture(frame, factory)
    assert source['score_value'] is None and source['formula_valid'] is True, 'firing_missing_formula_required'
    assert source['infer']['inferred'] == source['raw'] and source['infer']['deferred'] == 0, 'firing_infer_ambiguous'
    assert getattr(pipe, '_active_chain_1p') is None and not binding.owner.state.debts, 'firing_existing_origin'
    queue = pipe._pending_tsumo_1p
    owner = provider.journal.fifo.entries[(id(pipe), SIDE)]
    assert binding.scope[2] == owner['epoch'] and source['latest_O_frame'] < source['frame'], 'firing_scope_clock'
    grid = tuple(tuple(r) for r in source['infer']['inferred'])
    before = type(frame.f_locals['prev_confirmed']).from_dict({'grid': [list(r) for r in grid]})
    prediction = pipe._simulate_before_board(before)
    assert prediction is not None and prediction.chain_count >= 1, 'firing_no_prediction'
    assert F.key(prediction.final_board) != grid, 'firing_no_erasure'
    assert not any(c == 10 for row in grid for c in row), 'firing_unknown_grid'
    return Ticket(binding, binding.owner.state, binding.scope, queue, queue[0], owner['tokens'][0],
                  source, grid, before, prediction, environment)


def install(stack: Any, factory: Any, pipe: Any, patch: Any, rows: list[Any]) -> None:
    original = pipe._apply_chain_formula_early_fire
    def formula(self: Any, side: str, time_sec: float, prev_confirmed: Any) -> None:
        if side != SIDE or side not in factory.controller.history or self._active_chain_1p is not None:
            return original(side=side, time_sec=time_sec, prev_confirmed=prev_confirmed)
        binding = factory.controller.history[side]
        assert getattr(binding, 'firing_ticket', None) is None, 'firing_pending_ticket'
        ticket = qualified(sys._getframe(), factory, original.__func__.__globals__)
        binding.firing_ticket = ticket
        rows.append(dict(stage='formula_deferred', frame=ticket.source['frame'], source=ticket.source))
        # 元式イベントはまだ作らず、同update後段の原J消費へ資格を渡す。
    patch(stack, pipe, '_apply_chain_formula_early_fire', MethodType(formula, pipe))


def prepare(control: Any, binding: Any, item: Any, sm: Any, raw: Any, signals: Any, view: Any) -> Any:
    ticket = binding.firing_ticket
    assert not ticket.consumed and ticket.binding is binding and binding.owner.state is ticket.state
    assert binding.scope == ticket.scope == view.scope and view.frame == ticket.source['frame']
    assert view.clock == view.frame/FPS and not view.added and len(view.refs) == 1
    assert view.queue is ticket.queue and view.refs[0] is ticket.head and view.tokens == (ticket.token,)
    assert item.pair is ticket.head and binding.next_token == ticket.token and raw == ticket.grid
    assert signals.is_match_active and signals.chain_event is None and signals.effect_gate_window_active is False
    exact = control.observe_clear(binding, item, sm, raw, signals, view)
    assert exact and binding.clear_grid == raw and binding.clear_count >= PAIR_SIZE, 'firing_clear_support'
    p, state = control.inventory, binding.owner.state
    added = tuple(a-b for a,b in zip(p.S.color_counts(raw), state.counter))
    assert sum(added) == PAIR_SIZE and all(c >= 0 for c in added)
    clock = control._parts.C.H.clock
    now, occurred = clock(p, view.frame, view.clock, PLACEMENT_OPERATION), clock(p, *binding.clear_first)
    proof = dict(kind='live_firing_placement', token=ticket.token, source=ticket.source,
                 scope=asdict(state.scope), live_scope=ticket.scope, grid=raw,
                 action=state.action, available_frame=view.frame)
    evidence = p.S.PlacementEvidence(state.scope, p.digest(proof), state.action, now, added, occurred)
    return dict(kind='firing', evidence=evidence, proof=proof, old_state=state, grid=raw, ticket=ticket)
