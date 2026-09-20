"""旧整数対応は保持し、同frame暫定候補だけを失効させる。評価/会計権は発行しない。"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any
import conditional_current as C


def scope_matches(state: Any, value: Any) -> bool:
    scope = state.scope
    return (scope.source_sha256==value.scope[0].removeprefix('sha256:')
        and scope.run_id==value.scope[1] and scope.reset_epoch==value.scope[2]
        and scope.side==value.scope[-1])


@dataclass(frozen=True)
class AnchorCorrespondence:
    certificate: C.ConditionalCurrent
    owner_scope: str
    current_permission: bool = field(default=False,init=False)
    accounting_permission: bool = field(default=False,init=False)
    production_permission: bool = field(default=False,init=False)


def issue(state: Any, binding: Any, value: Any) -> AnchorCorrespondence:
    C.P.require(type(value) is C.ConditionalCurrent and C.intact(value),'lifetime_certificate')
    C.P.require(scope_matches(state,value) and value.scope==binding.scope,'lifetime_scope')
    C.P.require(state.action==value.action and C.anchor(state)==value.integer_anchor
        and binding.current==value.sm_grid,'lifetime_issue_correspondence')
    return AnchorCorrespondence(value,C.encoded(asdict(state.scope)))


def compatible(state: Any, binding: Any) -> bool:
    mapping = getattr(binding,'hidden_anchor',None)
    if type(mapping) is not AnchorCorrespondence: return False
    value = mapping.certificate
    return (type(value) is C.ConditionalCurrent and C.intact(value)
        and mapping.owner_scope==C.encoded(asdict(state.scope)) and scope_matches(state,value)
        and value.scope==binding.scope and state.action>=value.action
        and C.anchor(state)==value.integer_anchor and binding.current==value.sm_grid)


def ready(state: Any, binding: Any, frame: int, clock: float, *, stable: bool,
          active: bool, pending: bool) -> bool:
    """呼出側が原J/原updateから得た現在時刻と状態だけを照合する。"""
    value = getattr(binding,'hidden_current',None)
    if type(value) is not C.ConditionalCurrent or not compatible(state,binding): return False
    return (stable is True and active is True and pending is False
        and value is binding.hidden_anchor.certificate and frame==value.frame and clock==value.clock
        and state.action==value.action and C.intact(value))


def retire(binding: Any, frame: int, clock: float, *, stable: bool,
           active: bool, pending: bool) -> bool:
    """候補だけを落とす。原整数・履歴・最後の対応anchorを破棄しない。"""
    if getattr(binding,'hidden_current',None) is None: return False
    if ready(binding.owner.state,binding,frame,clock,stable=stable,active=active,pending=pending):
        return False
    binding.hidden_current = None
    return True
