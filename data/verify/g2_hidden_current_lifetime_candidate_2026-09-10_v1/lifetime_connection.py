"""原J正常閉鎖後の発行・失効を結合。旧整数対応を候補寿命から分離する。"""
from __future__ import annotations
from typing import Any
import lifetime as L
import hidden_current_connection as H


def install(stack: Any, factory: Any, patch: Any, rows: list[Any]) -> None:
    control,journal = factory.controller,factory.provider.journal
    patch(stack,H.C,'compatible',L.compatible)
    old_publish = H.publish
    def publish(control: Any, journal: Any, call: Any, made: Any,
                events: list[Any], outputs: list[Any]) -> None:
        binding = call['binding']
        try:
            correspondence = L.issue(binding.owner.state,binding,made[0])
            old_publish(control,journal,call,made,events,outputs)
        except BaseException as error:
            control.sticky_error = repr(error)
            raise
        binding.hidden_anchor = correspondence
    patch(stack,H,'publish',publish)
    old_complete = journal.complete_step
    def complete(item: Any, result: Any, error: Any, profile: Any) -> Any:
        call = control.calls.get(id(item['frame']))
        context = None if call is None else (call['view'],call['binding'],
            call['frame'].f_locals['signals'].is_match_active is True)
        try:
            returned = old_complete(item,result,error,profile)
        except BaseException:
            if call is not None: call['binding'].hidden_current = None
            raise
        if context is not None:
            view,binding,active = context
            retired = L.retire(binding,view.frame,view.clock,
                stable=error is None and result is not None and result.state.value=='stable',
                active=active,
                pending=binding.next_token is not None or bool(view.queue))
            if retired:
                rows.append(dict(kind='candidate_retired',frame=view.frame,
                    anchor_retained=L.compatible(binding.owner.state,binding),
                    pending=binding.next_token is not None,action=binding.owner.state.action))
        return returned
    patch(stack,journal,'complete_step',complete)
