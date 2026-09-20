"""factory属性が生きているclose直前に、初期Registryと確率Bindingの同一性を保存する。"""
from __future__ import annotations
import json
from typing import Any

KEY = '_g2_probabilistic_scope_registry'


def install(context: Any) -> None:
    factory, state = context.factory, context.state
    connection = state['probabilistic_basis_connection']
    registry = connection.registry
    assert getattr(factory, KEY) is registry and state[KEY] is registry
    captured = dict(factory_id=id(factory), registry_id=id(registry))
    def closing(error_type: Any, original_error: Any, traceback: Any) -> bool:
        report = dict(captured, ready=False, error=None, body_error=None if original_error is None else repr(original_error))
        try:
            assert getattr(factory, KEY) is registry and state[KEY] is registry
            assert connection.registry is registry and registry.factory is factory
            binding = connection.binding
            if binding is not None:
                current = registry.current(binding)
                report.update(ready=True, binding_id=id(binding), source_call_token=binding.initial_call_token,
                              frame=current.frame, scope=list(current.scope))
        except BaseException as error:
            report['error'] = repr(error)
        try:
            with (state['output'] / 'PROBABILISTIC_FACTORY_ANCHOR.json').open('x', encoding='utf-8') as stream:
                json.dump(report, stream, indent=2)
        except BaseException:
            if original_error is None:
                raise
        return False  # 元のbody例外を消さず、原restoreへ処理を返す。
    context.stack.push(closing)  # 最後に登録し、元binding closeがfactory属性を消す前に走らせる。


def derived(parent: Any) -> Any:
    class Context(parent):
        def perform(self, advisory: Any, frame: int, clock: float) -> None:
            super().perform(advisory, frame, clock)
            install(self)
    return Context
