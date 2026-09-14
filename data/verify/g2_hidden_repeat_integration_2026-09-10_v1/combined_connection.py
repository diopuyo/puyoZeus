"""原反復candidateを一回設置し、同じFactoryへ隠し非発火を追加する。"""
from __future__ import annotations
from contextlib import ExitStack
import sys
from typing import Any
import hidden_bundle as B
import routing as D
import lifecycle_binding as L


def install(stack: Any, factory: Any, pipe: Any, state: Any, rows: list[Any]) -> None:
    previous = list(sys.path)
    stack.callback(lambda:sys.path.__setitem__(slice(None),previous))
    sys.path.insert(0,str(B.SCOPE))
    candidate = B.K.load('_combined_repeat_candidate',B.REPEAT/'connection.py',stack)
    before = candidate.references(factory,pipe,state)
    captured,configure = [],candidate.P.Q.T.configure
    def record(*args: Any) -> Any:
        value = configure(*args)
        captured.append(value)
        return value
    with ExitStack() as capture:
        B.TAIL.patch(capture,candidate.P.Q.T,'configure',record)
        candidate.install(stack,factory,pipe,state,rows)
    assert len(captured)==1 and candidate.P.Q.T.configure is configure,'combined_install_once'
    configured = captured[0]
    control = factory.controller
    original = type(control).prepared.__globals__['V1'].prepared
    with ExitStack() as temporary:
        configured.patch(temporary,D.N,'install',D.natural_install)
        B.extensions(stack,factory,configured)
    L.install(stack,control,configured.patch)
    D.dispatch(stack,control,configured.patch,original)
    guard = B.K.load('_combined_scope_stop',B.SCOPE/'scope_stop.py',stack)
    state['repeat_scope_guard'] = guard.install(stack,factory,pipe,configured.patch,rows)
    state['combined_install'] = dict(configure_calls=len(captured),reset_recovery=False)
    def restored() -> None:
        assert before==candidate.references(factory,pipe,state),'combined_reference_restore'
    # 既存callbackの後に別途比較するため、実行元へ検査関数を返す。
    state['combined_restored'] = restored


def saved(state: Any, factory: Any) -> None:
    control,check = factory.controller,state.get('repeat_scope_guard')
    status = dict(installed=check is not None,
        error=None if check is None or check.error is None else repr(check.error),
        record=None if check is None else check.record,last_frame=None if check is None else check.frame,
        same_binding=check is not None and check.binding is control.history.get('1P'))
    B.K.write(state['output']/'SCOPE_STOP_STATUS.json',status)
    B.K.write(state['output']/'COMBINED_HIDDEN.json',dict(
        history=getattr(control,'hidden_history_rows',[]),events=getattr(control,'hidden_current_events',[]),
        lifetimes=getattr(control,'hidden_lifetime_rows',[]),installation=state.get('combined_install')))


def accepted(state: Any, factory: Any, final_frame: int) -> None:
    state['combined_restored']()
    check = state['repeat_scope_guard']
    assert check.error is None and check.record is None and check.frame==final_frame
    assert check.binding is factory.controller.history['1P']


def guards() -> dict[str,str]:
    with ExitStack() as stack:
        candidate = B.K.load('_combined_repeat_guards',B.REPEAT/'connection.py',stack)
        paths = [B.SCOPE/name for name in ('scope_stop.py','scope_stop_fixed.py','scope_stop_evidence.py')]
        return B.guards()|candidate.guards()|{str(p):B.K.sha(p) for p in paths}
