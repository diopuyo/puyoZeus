"""原同call NEXT基点を保持し、生色先行時は次action開始を保留する。"""
from __future__ import annotations
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent/'g2_firing_hand_connection_2026-09-10_v1'
POLICY = ROOT.parent/'g2_firing_policy_2026-09-10_v1'
sys.path[:0] = [str(PARENT), str(POLICY)]
from continuation_v1 import next_hand as OLD
from firing_fixed import code_value, declared

SOURCE = PARENT/'continuation_v1/next_hand.py'
SOURCE_SHA = 'd5c1b90d498d9fc738eba672facfdcb35d9076af965db2bf560aeba2c2666c2d'
PAIR_SIZE, COLORS = 2, frozenset((1, 2, 3, 4, 5))


def verify() -> None:
    assert Path(OLD.__file__).resolve() == SOURCE and hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    for name in ('current', 'retain', 'install'):
        function = getattr(OLD, name)
        assert function.__globals__ is vars(OLD)
        assert code_value(function.__code__) == code_value(declared(SOURCE, (name,)))


def basis(row: Any) -> dict[str, Any]:
    for pair in (row['accepted'], row['dnext']):
        assert type(pair) is tuple and len(pair) == PAIR_SIZE
        assert all(type(color) is int and color in COLORS for color in pair)
    assert type(row['segment']) is str and row['segment']
    return dict(accepted=row['accepted'], dnext=row['dnext'], segment=row['segment'])


def retain(control: Any, ticket: Any, view: Any) -> Any:
    row, enqueue = OLD.current(control, view)
    assert len(view.refs) <= 1 and tuple(view.queue) == view.refs, 'firing_pending_multiple_hands'
    saved = getattr(ticket, 'next_witness', None)
    if view.added:
        assert saved is None and len(view.refs) == 1 and view.added == view.tokens
        assert view.tokens[0] != ticket.token and row['committed'] is True
        proof = dict(enqueue=deepcopy(enqueue), frame=view.frame, scope=view.scope,
            token=view.tokens[0], source_basis=basis(row))
        ticket.next_witness = dict(proof=proof, digest=control.inventory.digest(proof),
            queue=view.queue, head=view.refs[0], token=view.tokens[0])
    elif saved is not None:
        assert control.inventory.digest(saved['proof']) == saved['digest'], 'firing_next_proof_changed'
        assert saved['proof']['scope'] == view.scope, 'firing_next_source_scope'
        assert view.queue is saved['queue'] and view.tokens == (saved['token'],)
        assert len(view.refs) == 1 and view.refs[0] is saved['head']
        assert basis(row) == saved['proof']['source_basis'], 'firing_next_source_basis_changed'
    return row


def ready(ticket: Any, view: Any) -> bool:
    saved = ticket.next_witness
    source = saved['proof']['source_basis']
    return (view.quiet is True and view.next_pair == source['accepted']
        and view.dnext_pair == source['dnext'])


def install(stack: Any, control: Any, patch: Any, rows: list[Any]) -> None:
    verify()
    cls, original = type(control), control.hand
    # 原no_origin wrapperの設置/復元を再用し、手の開始条件だけを追加する。
    OLD.install(stack, control, patch, rows)
    def hand(self: Any, binding: Any, view: Any) -> Any:
        ticket = getattr(binding, 'firing_ticket', None)
        if ticket is None or not ticket.consumed or getattr(ticket, 'next_started', False):
            return original(binding, view)
        assert ticket.scope == binding.scope == view.scope
        retain(self, ticket, view)
        if not getattr(ticket, 'settled', False) or not view.refs or not ready(ticket, view):
            return None
        saved = ticket.next_witness
        assert binding.next_token is None and view.tokens == (saved['token'],)
        assert view.refs[0] is saved['head'] and saved['proof']['frame'] <= view.frame
        self._parts.C.H.start(self.inventory, binding, saved['token'], view.frame, view.clock)
        ticket.next_started = True
        rows.append(dict(stage='next_action_started', frame=view.frame, source_frame=saved['proof']['frame'],
            token=saved['token'], action=binding.owner.state.action, backdated=False,
            source_basis=deepcopy(saved['proof']['source_basis']), source_basis_verified=True))
        return original(binding, view)
    patch(stack, cls, 'hand', hand)
