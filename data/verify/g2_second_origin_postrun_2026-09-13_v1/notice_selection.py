"""元の未登録対応Sessionを保持し、2P physical型だけ私有通知分類版へ選択する。"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import settled_notice as NOTICE
import notice_saved as SAVED

TARGET = Path(__file__).resolve().parent.parent / 'g2_m1_completion_candidate_2026-09-12_v1/second_mode_binding.py'


def session_class(original_factory: Any, original: Any, arrival: Any) -> type:
    base = original_factory(original, arrival)
    class Session(base):
        def __init__(self, stack: Any, context: Any, policy: Any, physical: Any,
                     contract: Any, members: Any, frames: tuple[int, ...]) -> None:
            super().__init__(stack, context, policy, physical, contract, members, frames)
            arrival.B.require(self.physical.Mode is arrival.BASE.Mode, 'notice_original_second_type')
            output = self.state['output'] / SAVED.FILENAME
            stream = stack.enter_context(output.open('x', encoding='utf-8'))
            def write(packet: dict[str, Any]) -> None:
                stream.write(json.dumps(packet, allow_nan=False) + '\n')
                stream.flush()
            self.physical = N(Mode=NOTICE.mode_class(arrival.BASE.Mode, arrival.B, write))
            self.second_settled_notice_stream = stream
    return Session


def install_load(stack: Any, value: Any, replace: Any) -> None:
    original_load = value.load
    factories: dict[Any, tuple[Any, Any]] = {}
    def load(alias: str, path: Any, injection: Any = None) -> Any:
        module = original_load(alias, path, injection)
        if Path(path).resolve() == TARGET:
            current = module.session_class
            if module not in factories:
                def factory(original: Any, arrival: Any) -> type:
                    return session_class(current, original, arrival)
                factories[module] = current, factory
            base, factory = factories[module]
            if current is not base and current is not factory:
                raise ValueError('notice_second_factory_changed')
            if current is not factory:
                replace(stack, module, 'session_class', factory)
        return module
    replace(stack, value, 'load', load)
