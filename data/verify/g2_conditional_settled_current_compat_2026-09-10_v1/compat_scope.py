"""条件付き・隠し履歴を通常整数current互換の対象から明示除外する。"""
from __future__ import annotations
from typing import Any
import compat_fixed as K

FORBIDDEN_ATTRIBUTES=('conditional_firing_registered','conditional_firing_committed','conditional_firing_origin',
    'hidden_anchor','hidden_current','hidden_last_placement','hidden_prefix_votes','hidden_continuation_votes')
PREFIXES=('conditional_','hidden_')
ORIGIN_KIND='firing_origin/v1'
SETTLEMENT_KIND='firing_settlement/v1'


def evidence(value: Any) -> None:
    if type(value) is dict:
        kind=value.get('kind')
        K.require(not (type(kind) is str and kind.startswith(PREFIXES)),'conditional_history_kind')
        K.require(value.get('conditional_world') is not True,'conditional_world')
        for nested in value.values(): evidence(nested)
    elif type(value) in (tuple,list):
        for nested in value: evidence(nested)


def ordinary(binding: Any) -> None:
    K.require(all(getattr(binding,name,None) is None for name in FORBIDDEN_ATTRIBUTES),'conditional_binding')
    state=binding.owner.state
    K.require(bool(state.origins) and all(type(origin.event_identity) is str
        and not origin.event_identity.startswith('conditional_world:') for origin in state.origins),
        'conditional_origin_identity')
    for row in binding.policy.accounting.proofs:
        proof=row['proof']; evidence(proof)
        if row['purpose']=='origin': K.require(proof.get('kind')==ORIGIN_KIND,'origin_kind')
        if row['purpose']=='settlement': K.require(proof.get('kind')==SETTLEMENT_KIND,'settlement_kind')

