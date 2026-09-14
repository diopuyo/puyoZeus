"""原Jの復帰待機中に限り、未認証NEXTを設置推論へ渡さない私有hook。"""
from __future__ import annotations
import sys
from typing import Any,Callable


def require(ok: bool,reason: str) -> None:
    if not ok: raise RuntimeError('reset_inflight_quarantine:'+reason)


class Guard:
    def __init__(self,recovery: Any,original: Callable[...,Any]) -> None:
        self.recovery,self.original=recovery,original
        self.rows: list[dict[str,Any]]=[]

    def waiting(self,caller: Any) -> bool:
        recovery=self.recovery
        if recovery.pending is None: return False
        local=caller.f_locals
        if local.get('side')!='1P': return False
        require(local.get('self') is recovery.pipe,'pipe_identity')
        item=recovery.journal.active
        require(caller.f_code in recovery.journal.codes and item is not None
            and item['frame'] is caller and item['pipe'] is recovery.pipe,'actual_J_caller')
        require(recovery.error is None and not recovery.journal.errors,'prior_error')
        scope=recovery.evidence.scope(recovery.factory,recovery.pipe)
        require(scope[2]==recovery.pending['epoch'] and not recovery.pending['used'],'pending_epoch')
        require(scope[:2]==recovery.pending['old_scope'][:2]
            and scope[3:5]==recovery.pending['old_scope'][3:5]
            and scope[5]>recovery.pending['old_scope'][5] and scope[6]=='1P','scope_identity')
        require(not recovery.pipe._pending_tsumo_1p,'new_FIFO_before_baseline')
        self.rows.append(dict(frame=local['frame_idx'],journal_token=item['token'],scope=scope,
            original_pair=local.get('falling_pair'),passed_pair=None,
            reason='reset_crossing_unowned_pair',physical_certified=False,accounting_permission=False))
        return True

    def infer(self,previous: Any,current: Any,pair: Any,*args: Any,**kwargs: Any) -> Any:
        caller=sys._getframe(1)
        selected=None if self.waiting(caller) else pair
        return self.original(previous,current,selected,*args,**kwargs)


def install(stack: Any,recovery: Any,namespace: dict[str,Any]) -> Guard:
    original=namespace['infer_placement']
    value=Guard(recovery,original)
    wrapper=value.infer
    namespace['infer_placement']=wrapper
    def restore() -> None:
        require(namespace['infer_placement'] is wrapper,'restore_binding_changed')
        namespace['infer_placement']=original
    stack.callback(restore)
    return value
