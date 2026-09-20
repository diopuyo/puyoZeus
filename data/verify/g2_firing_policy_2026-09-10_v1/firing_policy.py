"""既存BoundPolicyのログと参照を維持する同action発火用派生。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any
import firing_evidence as E
import firing_fixed as F

EXTRA_PURPOSES = frozenset(('origin', 'settlement'))


class FiringMethods:
    def __init__(self) -> None:
        type(self)._base_policy.__init__(self)
        self._sealed: Any = None
        self._firing_used: set[tuple[str, str]] = set()

    def arm(self, purpose: str, evidence: Any, state: Any, proof: dict[str, Any]) -> None:
        p, base = type(self)._inventory, type(self)._base_policy
        F.require(self._permit is None and self._sealed is None, 'policy_busy')
        key, saved = None, deepcopy(proof)
        sealed_evidence, sealed_state = E.frozen(evidence), E.frozen(state)
        logged = dict(purpose=purpose, proof=saved, proof_sha=p.digest(saved))
        sealed_log = E.frozen(logged)
        if purpose in EXTRA_PURPOSES:
            key = (purpose, getattr(E, purpose)(p, self.proofs, evidence, state, saved))
            F.require(key not in self._firing_used, 'permission_replay')
            self._permit = (purpose, p.encoded(asdict(evidence)), p.encoded(asdict(state)))
            self.proofs.append(logged)
        else:
            base.arm(self, purpose, evidence, state, saved)
        self._sealed = (purpose, sealed_evidence, sealed_state, len(self.proofs)-1, sealed_log, key)

    def authorize(self, purpose: str, evidence: Any, state: Any) -> None:
        F.require(self._sealed is not None, 'unarmed')
        expected, sealed_evidence, sealed_state, index, logged, key = self._sealed
        F.require(purpose == expected and E.frozen(evidence) == sealed_evidence
            and E.frozen(state) == sealed_state, 'armed_content_changed')
        F.require(index < len(self.proofs) and E.frozen(self.proofs[index]) == logged, 'armed_proof_changed')
        type(self)._base_policy.authorize(self, purpose, evidence, state)
        if key is not None: self._firing_used.add(key)
        self._sealed = None


def policy_type(p: Any) -> type:
    """Binding生成前に使用。既存owner/policyの参照差替えは行わない。"""
    F.verify(p)
    return type('FiringBoundPolicy', (FiringMethods, p.BoundPolicy),
        {'_inventory': p, '_base_policy': p.BoundPolicy, '__module__': __name__})
