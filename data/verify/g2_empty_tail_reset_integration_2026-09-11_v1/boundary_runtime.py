"""検収済みの公開時刻結合を固定版で読み、実factory側の全post状態も確認する。"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from typing import Any
import empty_reset as E
import run_qualification as Q
import retired_completion as R

PUBLICATION = Q.ROOT/'publication_boundary/verify.py'
PUBLICATION_SHA = '3bf639fb95c5d48e21ab551d957575e964e2adadaaa6371d32966e2da0d361c7'


def frame_state(factory: Any, lease: Any, frame: int) -> Any:
    binding = factory.controller.history.get(E.SIDE)
    if binding is None:
        assert lease.waiting and lease.recovery.pending is not None
        return dict(frame=frame, current=False, owner=False, old_evidence_leaked=False)
    state = binding.owner.state
    assert binding is not lease.archive.binding and binding.owner is not lease.archive.binding.owner
    assert state.scope.reset_epoch == lease.archive.binding.owner.state.scope.reset_epoch+1
    assert not state.origins and not state.debts and not state.consumed_ids
    factory.controller.inventory.S.validate_accounting(state)
    if state.current is not None:
        assert state.current.observed_at.frame > lease.empty_evidence.frame
        assert state.current.action <= state.action
    return dict(frame=frame, current=state.current is not None, owner=True,
        scope=list(binding.scope), old_evidence_leaked=False)


def publication(factory: Any, state: Any, recovery: Any) -> Any:
    assert hashlib.sha256(PUBLICATION.read_bytes()).hexdigest() == PUBLICATION_SHA
    module = Q.module('_empty_reset_publication_boundary', PUBLICATION)
    read = lambda name: [json.loads(line) for line in (state['output']/name).read_text().splitlines()]
    value = module.check(state['postcommit_publication_consumer'].rows,
        read('directional_history.jsonl'), read('atomic_journal.jsonl'), recovery.rows)
    assert recovery.control is factory.controller and recovery.journal is factory.provider.journal
    return value | dict(actual_factory=True)


def retire(lease: Any, factory: Any) -> None:
    # Archiveの型出所検査は、元モジュールが実在するinstaller範囲内で行う。
    result = R.verify(lease, factory)
    Q.KEPT['retired_samecall'] = deepcopy(result)
    Q.KEPT['retired_samecall_sha'] = E.digest(result)
    import runtime_world as W
    world = W.verify(factory,lease.recovery.state,lease)
    Q.KEPT['reset_world'] = deepcopy(world)
    Q.KEPT['reset_world_sha'] = E.digest(world)
