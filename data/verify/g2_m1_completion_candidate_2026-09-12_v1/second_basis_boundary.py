"""原資格検査を全て保持し、actionのみの境界は基準未発行のHOLDとする。"""
from __future__ import annotations

from types import FunctionType, SimpleNamespace
from typing import Any


def wrapped(original: Any, basis: Any) -> Any:
    def qualify(evidence: Any, witness: Any, pipe: Any) -> Any:
        held = False

        def require(ok: bool, reason: str) -> None:
            nonlocal held
            if not ok and reason == 'second_basis_generation':
                step = witness.pair(evidence.latest['frame'])[1]
                before, after = step['generation'], step['generation_after']
                left = {k: v for k, v in before.items() if k != 'action_revision'}
                right = {k: v for k, v in after.items() if k != 'action_revision'}
                action_before, action_after = before.get('action_revision'), after.get('action_revision')
                if (left == right and set(before) == set(after) and step['pipe_object_id'] == id(pipe)
                    and type(action_before) is int and type(action_after) is int
                    and action_after > action_before >= 0):
                    held = True
                    return  # 後続のcurrent/grid検査も原関数で実行し、最後にのみHOLD。
            basis.C.require(ok, reason)

        proxy = SimpleNamespace(**vars(basis.C))
        proxy.require = require
        selected = FunctionType(original.__code__, dict(original.__globals__, C=proxy),
                                original.__name__, original.__defaults__, original.__closure__)
        result = selected(evidence, witness, pipe)
        if held:
            raise basis.BasisHold('second_action_boundary_wait')
        return result

    return qualify
