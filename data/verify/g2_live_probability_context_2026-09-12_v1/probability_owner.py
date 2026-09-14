"""確率移行後の旧ownerを、元samecall検査へ接続する限定終了部品。"""
from __future__ import annotations
from copy import copy
from dataclasses import asdict
from types import FunctionType
from typing import Any

REGISTRY_KEY = '_g2_probabilistic_scope_registry'
ARCHIVE_KEY = '_g2_archive_lifetime'
SIDE = '1P'
STRIDE, FPS = 2, 60


def tracking_selected(state: dict[str, Any]) -> bool:
    """設置だけの未発動は旧経路へ。不整合・失敗・部分発動は新認証で拒否する。"""
    if 'probabilistic_tracking_mode' not in state:
        return False
    mode = state['probabilistic_tracking_mode']
    return not (mode.native is None and mode.activation is None
                and mode.connection.binding is None and mode.error is None)


def retained_receipt(verifier: Any, archive: Any) -> Any:
    """元receipt本文を保持型で実行し、解除済み登録表へ再突入しない。"""
    verifier.verify(archive)
    result = verifier.original.receipt()
    if verifier.private is None:
        return result
    module = verifier.modules[type(archive).__module__]
    private = copy(verifier.private)
    assert type(private) is module.Extra and verifier.check.__self__ is verifier.private
    before = dict(vars(verifier.private))
    private.verify = verifier.check
    extra = module.Extra.receipt(private)
    assert set(vars(private)) == set(before) | {'verify'}
    assert all(vars(private)[key] is value and vars(verifier.private)[key] is value
               for key, value in before.items()), 'retained_receipt_fields'
    assert private.verify is verifier.check and set(vars(verifier.private)) == set(before)
    return result | dict(empty_tail=extra)


def probability_authority(state: dict[str, Any], lease: Any, factory: Any) -> None:
    """整数ownerを作らず、実接続された確率所有権と退役を照合する。"""
    connection = state['probabilistic_basis_connection']
    mode = state['probabilistic_tracking_mode']
    recovery = lease.recovery
    assert connection.recovery is recovery and mode.connection is connection
    assert connection.registry is state[REGISTRY_KEY] and connection.registry.factory is factory
    current = connection.registry.current(connection.binding)
    assert current is not None, 'probability_current_missing'
    scope = recovery.evidence.scope(factory, recovery.pipe)
    old_scope = lease.archive.binding.scope
    assert current.scope == connection.binding.scope == scope
    assert scope[:5] == lease.new_scope[:5] and scope[-1] == lease.new_scope[-1] == SIDE
    assert scope[2] == old_scope[2] + 1 and scope[5] == old_scope[5] + 1
    assert lease.new_generation == scope[5] and lease.guard.binding is None
    assert lease.empty_new_binding is None and lease.empty_new_owner is None
    assert SIDE not in factory.controller.history
    assert mode.native is not None and mode.native.connection is connection
    assert mode.error is None and recovery.error is None and recovery.failure is None
    assert connection.observer.error is None and connection.observer.recovery is recovery
    assert mode.activation['source_call_token'] == connection.binding.initial_call_token
    assert mode.activation['prior_pending']['empty_retirement'] == lease.retirement
    assert mode.activation['prior_pending']['epoch'] == scope[2]
    assert mode.activation['prior_pending']['used'] is False
    assert mode.activation['frame'] <= current.frame <= mode.native.last_frame <= current.deadline
    assert mode.activation['tracking_deadline'] == current.deadline == connection.deadline
    candidate = connection.observer.gate.candidate
    assert candidate.scope == scope and candidate.frame == mode.activation['frame']
    assert candidate.source_call_token == connection.binding.initial_call_token
    qualified = [row for row in lease.events if row['kind'] == 'reset_qualified']
    assert len(qualified) == 1 and qualified[0]['frame'] == lease.empty_evidence.frame + STRIDE
    assert qualified[0]['clock'] == qualified[0]['frame'] / FPS


def side_reset_authority(E: Any, lease: Any) -> None:
    """全体reset回数を偽装せず、採用済み片側resetの一回資格へ束縛する。"""
    assert lease.outer_calls == lease.native_calls == lease.depth == 0, 'unexpected_whole_reset'
    qualified = [(index, row) for index, row in enumerate(lease.events) if row['kind'] == 'reset_qualified']
    waiting = [(index, row) for index, row in enumerate(lease.events) if row['kind'] == 'side_reset_waiting']
    assert len(qualified) == len(waiting) == 1, 'side_reset_event_count'
    assert qualified[0][0] < waiting[0][0], 'side_reset_event_order'
    assert not any(row['kind'] == 'reset_waiting' for row in lease.events), 'mixed_whole_reset_event'
    event = waiting[0][1]
    assert event['outer_calls'] == event['native_calls'] == 0, 'side_reset_event_calls'
    assert tuple(event['scope']) == tuple(lease.new_scope), 'side_reset_event_scope'
    assert E.digest(qualified[0][1]['old']) == lease.empty_evidence.receipt_sha, 'side_reset_qualified_archive'


def retired_owner(original: Any, runtime: dict[str, Any], lease: Any, factory: Any,
                  binding: Any, current: Any, state: Any, key: Any) -> bool:
    """原旧ownerの全値検査を維持。終了後の整数observe_waitは再実行しない。"""
    E = original.E
    assert type(lease) is E.Lease and type(lease.empty_evidence) is E.PrefixEvidence
    proof, recovery = lease.empty_evidence, lease.recovery
    assert lease.used and not lease.active and not lease.waiting and proof.used
    assert recovery.reset_count == 1 and recovery.baseline_count == 0 and recovery.pending is None
    side_reset_authority(E, lease)
    assert recovery.factory is proof.factory is factory and recovery.control is factory.controller
    assert recovery.journal is factory.provider.journal
    assert lease.archive is proof.archive and lease.archive.binding is binding is proof.binding
    assert E.digest(retained_receipt(runtime[ARCHIVE_KEY], lease.archive)) == proof.receipt_sha
    assert E.digest(proof.result) == proof.result_sha
    assert len(recovery.archive) == 1
    old, saved, record = recovery.archive[0]
    assert old is binding and binding.owner.state is saved is current
    assert asdict(saved) == record['owner'] and current.scope == state.scope
    assert key[-1] == binding.scope[-1] == SIDE and key[0] <= proof.frame
    assert lease.retirement['old_scope'] == list(binding.scope)
    assert lease.retirement['archive_sha256'] == proof.receipt_sha
    assert lease.retirement['samecall'] == proof.result
    assert lease.retirement['missing_count'] == 'UNCERTIFIED'
    probability_authority(runtime, lease, factory)
    return True


def verify(original: Any, runtime: dict[str, Any], lease: Any, factory: Any) -> Any:
    """原verify自身のsource/code guardと同call再照合を、そのまま実行する。"""
    def owner(*args: Any) -> bool:
        return retired_owner(original, runtime, *args)
    function = original.verify
    copied = FunctionType(function.__code__, dict(function.__globals__, retired_owner=owner),
                          function.__name__, function.__defaults__, function.__closure__)
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied(lease, factory)
