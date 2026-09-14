"""登録解除前に元Archive検査と型定義保持を行う。bodyの元例外は保持。"""
from __future__ import annotations
import json
from typing import Any
import archive_lifetime as L

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
                archive = state['repeat_scope_guard'].reset_lease.archive
                assert L.KEY not in state
                state[L.KEY] = L.Verifier(archive)
                report.update(ready=True, binding_id=id(binding), source_call_token=binding.initial_call_token,
                    frame=current.frame, scope=list(current.scope), archive_id=id(archive),
                    old_archive_verified=True, type_modules_retained_until_finish=True)
        except BaseException as error:
            report['error'] = repr(error)
        try:
            with (state['output'] / 'PROBABILISTIC_FACTORY_ANCHOR.json').open('x', encoding='utf-8') as stream:
                json.dump(report, stream, indent=2)
        except BaseException:
            if original_error is None:
                raise
        return False
    context.stack.push(closing)


def derived(parent: Any) -> Any:
    class Context(parent):
        def perform(self, advisory: Any, frame: int, clock: float) -> None:
            super().perform(advisory, frame, clock)
            install(self)
    return Context
