"""factory内の確率scope所有権を一意化する。実J資格の取得はadapterの責務。"""
from __future__ import annotations
from dataclasses import dataclass
from threading import Lock
from typing import Any,Callable
import belief as B


@dataclass(frozen=True)
class Binding:
    scope: tuple[Any,...]
    initial_call_token: str


class Registry:
    def __init__(self,factory: Any) -> None:
        self.factory=factory
        self._lock=Lock()
        self._bindings: dict[str,Binding]={}
        self._states: dict[tuple[Any,...],B.Belief]={}
        self._used: set[tuple[tuple[Any,...],str]]=set()

    def bind(self,factory: Any,value: B.Belief,call_token: str) -> Binding:
        B.require(factory is self.factory,'registry_factory')
        B.validate(value)
        B.require(type(call_token) is str and bool(call_token),'registry_call_token')
        with self._lock:
            B.require(value.scope[-1] not in self._bindings and value.scope not in self._states,'registry_duplicate_scope')
            binding=Binding(value.scope,call_token)
            self._bindings[value.scope[-1]]=binding
            self._states[value.scope]=value
            self._used.add((value.scope,call_token))
            return binding

    def _state(self,binding: Binding) -> B.Belief:
        B.require(type(binding) is Binding and self._bindings.get(binding.scope[-1]) is binding,'registry_stale_binding')
        return self._states[binding.scope]

    def current(self,binding: Binding) -> B.Belief:
        with self._lock: return self._state(binding)

    def transition(self,factory: Any,binding: Binding,expected: B.Belief,call_token: str,
                   operation: Callable[[B.Belief],tuple[B.Belief,Any]]) -> tuple[B.Belief,Any]:
        B.require(factory is self.factory,'registry_factory')
        B.require(type(call_token) is str and bool(call_token),'registry_call_token')
        with self._lock:
            value=self._state(binding)
            B.require(value is expected,'registry_stale_state')
            B.require((binding.scope,call_token) not in self._used,'registry_reused_call')
            result,detail=operation(value)
            B.validate(result)
            B.require(result.scope==value.scope and result.frame>value.frame,'registry_transition_scope_or_clock')
            B.require(result.tokens[:len(value.tokens)]==value.tokens,'registry_lost_tokens')
            self._states[binding.scope]=result
            self._used.add((binding.scope,call_token))
            return result,detail

    def retire(self,factory: Any,binding: Binding) -> B.Belief:
        B.require(factory is self.factory,'registry_factory')
        with self._lock:
            value=self._state(binding)
            del self._bindings[binding.scope[-1]]
            return value
