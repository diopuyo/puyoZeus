"""原SMが選んだSTABLEだけを可視照合し、整数推定値を注入しない。"""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
import inspect
import sys
from types import SimpleNamespace
from typing import Any, Iterator
import path_support as P
import prefix_witness as W
import prefix_provenance as V


def eligible(control: Any, binding: Any, signals: Any, call: Any) -> Any:
    if not getattr(binding,'hidden_tail_consumed',False) or getattr(binding,'hidden_current',None) is not None:
        return None
    view,state = call['view'],binding.owner.state
    if (call['prepared'] is not None or call['consumed'] or binding.next_token is not None
        or view.refs or view.tokens or signals.is_match_active is not True
        or signals.chain_event is not None or signals.effect_gate_window_active is not False): return None
    if view.quiet is not True or not control._parts.T.valid_pair(view.next_pair): return None
    P.require(not state.origins and not state.debts and binding.scope==view.scope,'exit_scope')
    P.require(state.counter==control.inventory.S.color_counts(binding.grid),'exit_inventory')
    V.validate(control,binding,binding.hidden_prefix_votes)
    captured = W.observed(control,binding,signals,view)
    if captured is None or not P.compatible(captured[0],binding.grid): return None
    tail = binding.hidden_tail_votes
    P.require(tail.final==binding.grid and view.frame>tail.last[0],'exit_after_tail')
    return captured


def transition(control: Any, sm: Any, signals: Any, binding: Any, call: Any,
               target: Any, original: Any, rows: list[Any]) -> bool:
    captured = eligible(control,binding,signals,call)
    if captured is None: return False
    raw,proof = captured
    shadow,observed = deepcopy(sm),deepcopy(signals)
    original(shadow,target,observed)
    expected = control._parts.T.board_key(shadow.context.confirmed_board)
    matched = shadow.context.state.value=='stable' and expected[1:]==raw[1:]==binding.grid[1:]
    row = dict(kind='natural_hidden_exit_preview',frame=call['view'].frame,before=sm.context.state.value,
        preview=expected,raw=raw,visible_matches=matched,original_choice=True,
        history_votes=len(sm.context.non_stable_cnn_history),physical_certified=False)
    rows.append(row)
    if not matched: return False
    before = control._parts.T.board_key(sm.context.confirmed_board)
    original(sm,target,signals)
    actual = control._parts.T.board_key(sm.context.confirmed_board)
    P.require(actual==expected and sm.context.state.value=='stable','exit_actual_preview')
    binding.current = actual
    call['hidden_transition'] = dict(frame=call['view'].frame,raw_capture=proof,
        previous_current=before,actual_current=actual,natural_choice=True,raw_unchanged=True)
    P.require(control._parts.T.board_key(signals.cnn_board)==raw,'exit_raw_mutated')
    return True


def wrappers(control: Any, sm: Any, signals: Any, binding: Any, module: Any,
             call: Any, apply: Any, within: Any, rows: list[Any]) -> tuple[Any,Any]:
    env = sys.modules[type(sm).__module__]
    def check(current: Any, observed: Any, caller: Any) -> None:
        P.require(current is sm and observed is signals,'exit_objects')
        P.require(module.T.original_code(caller.f_code,'update') if caller.f_code.co_name=='update'
            else caller.f_code is apply.__code__,'exit_original_caller')
    def applied(current: Any, target: Any, observed: Any) -> None:
        check(current,observed,sys._getframe(1))
        if (target.value=='stable' and current.context.state.value in module.T.ACTIVE_STATES
            and transition(control,current,observed,binding,call,target,apply,rows)):
            pass
        elif target.value=='stable' and module.should_fall(current,observed,binding):
            apply(current,env.BoardState.TSUMO_FALL,observed)
        elif target.value=='stable' and current.context.state.value in module.T.ACTIVE_STATES:
            within(current,observed)
        else:
            apply(current,target,observed)
        P.require(module.T.board_key(current.context.confirmed_board)==binding.current,'exit_current')
    def updated(current: Any, observed: Any) -> None:
        check(current,observed,sys._getframe(1))
        if module.should_fall(current,observed,binding): apply(current,env.BoardState.TSUMO_FALL,observed)
        else: within(current,observed)
    return applied,updated


@contextmanager
def installed(control: Any, sm: Any, signals: Any, binding: Any,
              module: Any, rows: list[Any]) -> Iterator[None]:
    calls = [v for v in control.calls.values() if v['binding'] is binding
        and v['frame'].f_locals.get('signals') is signals]
    P.require(len(calls)==1,'exit_owned_call')
    cls,call = type(sm),calls[0]
    apply,within = cls._apply_transition,cls._update_within_current_state
    env = sys.modules[cls.__module__]
    P.require(module.T.original_code(apply.__code__,'_apply_transition')
        and module.original_within(within.__code__),'exit_original_SM')
    P.require(apply.__globals__ is within.__globals__ is vars(env),'exit_SM_globals')
    cls._apply_transition,cls._update_within_current_state = wrappers(
        control,sm,signals,binding,module,call,apply,within,rows)
    try:
        yield
    finally:
        cls._apply_transition,cls._update_within_current_state = apply,within


def install(stack: Any, control: Any, patch: Any, rows: list[Any]) -> None:
    cls,old = type(control),type(control).hold_transition
    module = SimpleNamespace(**inspect.unwrap(old).__globals__)
    P.require(hasattr(module,'T') and hasattr(module,'original_within'),'exit_module_binding')
    def hold(self: Any, sm: Any, signals: Any, binding: Any) -> Any:
        if not getattr(binding,'hidden_tail_consumed',False): return old(self,sm,signals,binding)
        return installed(self,sm,signals,binding,module,rows)
    patch(stack,cls,'hold_transition',hold)
