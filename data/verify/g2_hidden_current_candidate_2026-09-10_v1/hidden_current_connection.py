"""原J返却の正常閉鎖後にだけ別型currentを確定し、旧整数slotを保持する。専用名。"""
from __future__ import annotations
from dataclasses import asdict
import sys
from typing import Any
import conditional_current as C
import natural_exit as H


def next_fall(original: Any, sm: Any, signals: Any, binding: Any) -> bool:
    if original(sm,signals,binding): return True
    return (signals.is_match_active is True and signals.chain_event is None
        and sm.context.state.value=='stable' and binding.next_token is not None
        and C.compatible(binding.owner.state,binding))


def fall_connection(stack: Any, control: Any, patch: Any) -> None:
    values = type(control).prepared.__globals__
    module = sys.modules[values['__name__']]
    C.P.require(vars(module) is values,'conditional_v3_module')
    old = module.should_fall
    def selected(sm: Any, signals: Any, binding: Any) -> bool:
        return next_fall(old,sm,signals,binding)
    patch(stack,module,'should_fall',selected)


def capture(control: Any, journal: Any, item: Any, result: Any, provisional: Any) -> Any:
    frame = item['frame']
    call = control.calls.get(id(frame))
    if call is None: return None,None
    C.P.require(call['frame'] is frame and journal.active is None
        and frame.f_code in journal.codes,'conditional_actual_J_call')
    C.P.require(item['scope']['frame_idx']==call['view'].frame
        and item['scope']['time_sec']==call['view'].clock,'conditional_J_scope')
    return call,C.make(control,call,result,provisional)


def publish(control: Any, journal: Any, call: Any, made: Any,
            rows: list[Any], outputs: list[Any]) -> None:
    """正常cleanup後の再検査も失敗状態へ固定し、候補を発行しない。"""
    value,side = made
    binding = call['binding']
    try:
        C.P.require(not journal.errors and not control.sticky_error,'conditional_close_failed')
        C.P.require(C.anchor(binding.owner.state)==value.integer_anchor,'conditional_anchor_changed')
    except BaseException as error:
        control.sticky_error = repr(error)
        raise
    binding.hidden_current = value
    outputs.append((value,side))
    rows.append(dict(kind='conditional_current_after_original_J',frame=value.frame,
        current=asdict(value),original_result_unchanged=True,integer_current_recovered=False))


def install(stack: Any, factory: Any, patch: Any, provisional: Any,
            rows: list[Any], outputs: list[Any]) -> None:
    control = factory.controller
    fall_connection(stack,control,patch)
    H.install(stack,control,patch,rows)
    cls = type(control)
    v1 = cls.prepared.__globals__['V1']
    lifecycle = v1.L
    old_compatible = lifecycle.compatible_current
    def compatible(state: Any, binding: Any) -> bool:
        return old_compatible(state,binding) or C.compatible(state,binding)
    patch(stack,lifecycle,'compatible_current',compatible)
    journal,old_complete = factory.provider.journal,factory.provider.journal.complete_step
    def completed(item: Any, result: Any, error: Any, profile: Any) -> Any:
        call,made,failure = None,None,None
        try:
            if error is None: call,made = capture(control,journal,item,result,provisional)
        except BaseException as caught:
            failure = caught
        # 捕捉失敗でも原J/Controllerのcleanupを一回通してからfail-stopする。
        original = old_complete(item,result,error,profile)
        if error is not None: return original
        if failure is not None:
            control.sticky_error = repr(failure)
            raise failure
        if made is not None:
            publish(control,journal,call,made,rows,outputs)
        return original
    patch(stack,journal,'complete_step',completed)
