"""履歴だけに方向付きNEXTを導入。既存会計/公開gateは継承する。"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
import fixed as F
from evidence import Link


def provider_init(self: Any, journal: Any, occurrence_adapter: Any) -> None:
    self._parts.V.Provider.__init__(self, journal)
    self.link = Link(self._parts, journal, occurrence_adapter)
    self.handoff_proofs: dict[str, Any] = {}


def provider_attach(self: Any, stack: Any, controller: Any) -> None:
    self._parts.V.Provider.attach(self, stack, controller)
    self.link.attach(stack)


def advance_evidence(self: Any, item: Any, view: Any) -> Any:
    proof = self.link.proof(item, view, self.enqueues.get(view.scope[-1]))
    if proof is not None:
        self.handoff_proofs[view.scope[-1]] = proof
    return proof


def after_history_consume(self: Any, call: Any, caller: Any) -> None:
    self._parts.V.Provider.after_history_consume(self, call, caller)
    proof = self.handoff_proofs[call['view'].scope[-1]]
    self.link.consume(proof, call['view'])


def hand(self: Any, binding: Any, view: Any) -> Any:
    if binding.candidate is None and view.refs:
        accepted, dnext = self.provider.link.basis(view)
        view = replace(view, next_pair=accepted, dnext_pair=dnext)
    return self._parts.C.Controller.hand(self, binding, view)


def advance(self: Any, item: Any, view: Any) -> bool:
    return self.provider.advance_evidence(item, view) is not None


def prepared(self: Any, binding: Any, item: Any, sm: Any, raw: Any,
             signals: Any, view: Any) -> Any:
    exact = self.observe_clear(binding, item, sm, raw, signals, view)
    if not (exact and binding.clear_count >= self._parts.C.CLEAR_OBSERVATIONS
            and binding.clear_grid == raw):
        return None
    directional = self.provider.advance_evidence(item, view)
    if directional is None:
        return None
    proof = {'kind': 'live_historical_placement', 'token': item.token, 'pair': item.pair,
        'occurred': binding.clear_first, 'available_frame': view.frame, 'available_time': view.clock,
        'clear_last': binding.clear_last, 'clear_observations': binding.clear_count,
        'available_window': signals.effect_gate_window_active, 'new_token': view.added[0],
        'directional_commit': directional}
    return self._parts.H.prepare(self.inventory, binding, view, raw, proof)


def make_types(c: Any, v: Any, h: Any, t: Any, o: Any) -> Any:
    parts = F.parts(c, v, h, t, o)
    provider = type('Provider', (v.Provider,), {'_parts': parts, '__init__': provider_init,
        'attach': provider_attach, 'advance_evidence': advance_evidence,
        'after_history_consume': after_history_consume})
    controller = type('Controller', (c.Controller,), {'_parts': parts, 'hand': hand,
        'advance': advance, 'prepared': prepared})
    return SimpleNamespace(Controller=controller, Provider=provider, Link=Link, parts=parts)
