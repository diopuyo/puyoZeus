"""原handoff検証後、生成済み専用型だけへ欠測判定を接続する。"""
from __future__ import annotations
from typing import Any
import common as K


def configure(stack: Any, core: Any, current: Any) -> None:
    missing = K.load('_probe_missing_observation', K.MISSING/'connection.py', stack)
    original = current.factory_type
    def factory(assembly: Any, loaded: Any) -> type:
        base = original(assembly, loaded)
        class Factory(base):
            def make_controller(self, provider: Any, legal: Any, *, enabled: bool = False) -> Any:
                control = super().make_controller(provider, legal, enabled=enabled)
                stack.enter_context(missing.installed(type(control)))
                return control
        return Factory
    core.G.patch(stack, current, 'factory_type', factory)
