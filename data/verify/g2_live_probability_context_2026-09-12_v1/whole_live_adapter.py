"""原constructor_guardの後へwhole producer/Session候補を設置。実動画GOはまだ禁止。"""
from __future__ import annotations
import hashlib
from pathlib import Path
from types import FunctionType
from typing import Any
import live_adapter as A
import loader as L
import whole_collector_capture as W
import whole_session_driver as DRIVER
import whole_dependencies as D

EVALUATION_FRAMES = (35370, 35410)  # 同video38の既存選択点。開始資格を代替しない。
FILES = ('whole_live_adapter.py', 'whole_collector_capture.py', 'whole_session_driver.py',
         'whole_dependencies.py')


def protect(stack: Any, owner: Any, pins: dict[str, str]) -> None:
    previous = owner.guards
    def guarded() -> dict:
        W.require(all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in pins.items()),
                  'whole_source_changed')
        return previous() | pins
    def restore(kind: Any, body: Any, trace: Any) -> bool:
        if owner.guards is guarded:
            owner.guards = previous
        elif body is None:
            raise ValueError('whole_guard_restore')
        return False
    owner.guards = guarded
    stack.push(restore)


def configured(stack: Any) -> Any:
    main = A.configured(stack)
    namespace = main.__globals__
    previous, common = namespace['collect'], namespace['K']
    session_common = namespace['S'].K
    original_guard = previous.__globals__['constructor_guard']
    def guard(inner: Any, collector: Any, state: Any, base: Any, env: Any = None) -> None:
        original_guard(inner, collector, state, base, env)
        load = L.bootstrap().load
        scope = D.Scope(inner)
        state['whole_dependency_scope'] = scope
        recorder, anchor, create = scope.prepare(load, EVALUATION_FRAMES)
        driver = DRIVER.Driver(state, create, EVALUATION_FRAMES)
        W.Bridge(collector, state, inner, recorder, anchor, tuple(common.FRAMES), driver)
    selected = FunctionType(previous.__code__, dict(previous.__globals__, constructor_guard=guard),
                            previous.__name__, previous.__defaults__, previous.__closure__)
    selected.__kwdefaults__ = previous.__kwdefaults__
    sources = tuple(A.ROOT / name for name in FILES) + D.source_paths()
    pins = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    protect(stack, common, pins)
    if session_common is not common:
        protect(stack, session_common, pins)
    def restore(kind: Any, body: Any, trace: Any) -> bool:
        if namespace['collect'] is selected:
            namespace['collect'] = previous
        elif body is None:
            raise ValueError('whole_live_alias_restore')
        return False
    stack.push(restore)
    namespace['collect'] = selected
    return main
