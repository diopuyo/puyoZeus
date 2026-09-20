"""精算後の最初のNEXT一枠は観測のみ行い、二枠handoffへ渡さない。"""
from __future__ import annotations
from types import SimpleNamespace
from typing import Any
from continuation_v1 import transforms as T, next_hand as N


def configure(stack: Any, base: Any, control: Any) -> Any:
    configured = T.configure(stack, base, control)
    original = configured.prepare
    def prepare(lifecycle: Any, owner: Any, binding: Any, item: Any, sm: Any,
                raw: Any, signals: Any, view: Any) -> Any:
        ticket = getattr(binding, 'firing_ticket', None)
        if ticket is not None and getattr(ticket, 'next_started', False) and len(view.refs) == 1:
            assert T.closed_origins(binding.owner.state, binding), 'single_head_unsettled'
            N.current(owner, view)
            assert binding.next_token == item.token == view.tokens[0], 'single_head_token'
            assert item.queue is view.queue and item.pair is view.refs[0], 'single_head_reference'
            owner.observe_clear(binding, item, sm, raw, signals, view)
            return None
        return original(lifecycle, owner, binding, item, sm, raw, signals, view)
    return SimpleNamespace(**(vars(configured) | {'prepare': prepare}))
