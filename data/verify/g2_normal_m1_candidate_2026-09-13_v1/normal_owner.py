"""resetを偽装せず、正常窓の同factory確率Registryを排他的に所有する。"""
from __future__ import annotations
import json
from typing import Any
import belief as B
import conditioning as C
import serialization as S
import registry as R
import journal_context as J

KEY = 'normal_probability_owner'
REGISTRY_KEY = '_g2_probabilistic_scope_registry'


class Owner:
    def __init__(self, factory: Any, pipe: Any, state: dict, start: int, deadline: int) -> None:
        J.require(type(start) is int and type(deadline) is int and 0 <= start < deadline, 'normal_window')
        J.require(state['private_suffix_factory'] is factory and factory.provider.journal.pipe is pipe,
                  'normal_factory_journal_owner')
        J.require(KEY not in state and REGISTRY_KEY not in state
                  and not hasattr(factory, REGISTRY_KEY), 'normal_duplicate_owner')
        self.factory, self.pipe, self.state = factory, pipe, state
        self.start, self.deadline = start, deadline
        self.registry = R.Registry(factory)
        self.closed, self.error = False, None

    def require_live(self) -> None:
        J.require(not self.closed and self.error is None and self.state.get(KEY) is self
                  and self.state.get(REGISTRY_KEY) is self.registry
                  and getattr(self.factory, REGISTRY_KEY, None) is self.registry, 'normal_owner_lifetime')

    def eligible_frame(self, frame: int) -> bool:
        self.require_live()
        J.require(type(frame) is int and 0 <= frame <= self.deadline, 'normal_frame_bound')
        return frame >= self.start

    def create_mode(self, physical: Any, evidence: Any, witness: Any, policy: Any,
                    inner: Any, side: str) -> Any:
        self.require_live()
        frame = evidence.latest['frame']
        J.require(self.eligible_frame(frame), 'normal_basis_before_start')
        return physical.Mode(evidence, witness, self.pipe, self.registry, self.factory,
                             self.deadline, policy, inner, self.factory.provider, side=side)

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        if self.closed:
            return False
        try:
            self.require_live()
        except BaseException as error:
            self.error = error
        finally:
            # 他所有者を消さない。所有している部分だけ解放し、違反は終了票へ残す。
            if getattr(self.factory, REGISTRY_KEY, None) is self.registry:
                delattr(self.factory, REGISTRY_KEY)
            for key, value in ((REGISTRY_KEY, self.registry), (KEY, self)):
                if self.state.get(key) is value:
                    del self.state[key]
            self.closed = True
        packet = dict(closed=True, error=None if self.error is None else repr(self.error),
                      original_body=None if body is None else repr(body), start=self.start,
                      deadline=self.deadline, reset_claimed=False, quality_gate_clear=False)
        try:
            with (self.state['output'] / 'NORMAL_OWNER_STATUS.json').open('x', encoding='utf-8') as stream:
                json.dump(packet, stream, ensure_ascii=False, allow_nan=False)
        except BaseException as error:
            if self.error is None:
                self.error = error
        if body is None and self.error is not None:
            raise self.error
        return False


def install(stack: Any, factory: Any, pipe: Any, state: dict, start: int, deadline: int) -> Owner:
    owner = Owner(factory, pipe, state, start, deadline)
    stack.push(owner.close)
    setattr(factory, REGISTRY_KEY, owner.registry)
    state[REGISTRY_KEY], state[KEY] = owner.registry, owner
    return owner
