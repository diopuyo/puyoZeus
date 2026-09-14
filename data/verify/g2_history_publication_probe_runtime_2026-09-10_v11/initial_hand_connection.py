"""初回の既知未確定手だけを保守的NON-STABLEへ移す。票や盤面は書かない。"""
from __future__ import annotations
from typing import Any


def initial_hand(binding: Any) -> bool:
    state, item, token = binding.owner.state, binding.candidate, binding.next_token
    if (state.current is not None or state.history or binding.phase != 'historical_owned'
        or item is None or token is None or binding.next_started is None):
        return False
    return bool(item.token == token and item.scope == binding.scope
        and item.baseline == binding.grid == binding.current
        and item.started == binding.next_started[0] and token not in binding.consumed_tokens
        and item.queue and item.queue[0] is item.pair)


def should_fall(original: Any, sm: Any, signals: Any, binding: Any) -> bool:
    if original(sm, signals, binding):
        return True
    return (signals.is_match_active is True and signals.chain_event is None
        and sm.context.state.value == 'stable' and initial_hand(binding))


def configure(stack: Any, core: Any, module: Any) -> None:
    original = module.should_fall
    core.G.require(original.__globals__ is vars(module), 'initial_hand_original_globals')
    def selected(sm: Any, signals: Any, binding: Any) -> bool:
        return should_fall(original, sm, signals, binding)
    core.G.patch(stack, module, 'should_fall', selected)
