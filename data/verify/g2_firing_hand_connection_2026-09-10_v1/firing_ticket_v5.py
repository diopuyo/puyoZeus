"""固定変換の実updateと原Jが所有するNEXTインスタンスを結ぶ。"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any, Iterator
import firing_ticket_v4 as V4


@contextmanager
def installed(pipe: Any, evidence: dict[str, Any]) -> Iterator[None]:
    actual = V4.bound_update(pipe)
    controller = actual.__globals__['__next_live']
    v2, old = V4.V3.V2, V4.V3.V2.OLD.qualified
    def qualified(frame: Any, factory: Any, environment: Any) -> Any:
        assert sys._getframe(1) is frame, 'firing_front_frame_identity'
        caller = frame.f_back
        assert actual.__globals__.get('__next_live') is controller, 'firing_next_binding_changed'
        assert factory.provider.journal.controller is controller, 'firing_next_instance_mismatch'
        assert caller.f_code is actual.__code__ and caller.f_globals is actual.__globals__, 'firing_front_code'
        assert frame.f_locals['self'] is caller.f_locals['self'] is pipe, 'firing_front_pipe'
        assert Path(environment['__file__']).resolve() == v2.SOURCE, 'firing_front_module'
        value = v2.ORIGINAL(frame, factory, environment)
        value.source.update(caller_code_verified=True, caller_globals_verified=True,
            immediate_frame_verified=True, next_instance_verified=True)
        return value
    v2.OLD.qualified = qualified
    evidence.update(bound_original_update=True, fixed_NEXT_transform_verified=True,
                    bound_NEXT_instance_guard=True)
    try:
        yield
    finally:
        v2.OLD.qualified = old
        evidence['qualified_restored'] = v2.OLD.qualified is old
