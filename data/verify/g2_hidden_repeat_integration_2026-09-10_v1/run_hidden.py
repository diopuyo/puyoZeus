"""共通installerの原整数currentあり65更新検収。"""
from __future__ import annotations
from types import FunctionType,SimpleNamespace
from typing import Any
import hidden_bundle as B
import combined_connection as C

R,I = B.HR.R,B.HR.I


def drive(*args: Any) -> Any:
    state,pipe,factory = args[2],args[3][0],args[6]
    firing = []
    def install(stack: Any, current: Any, unused: Any) -> None:
        C.install(stack,current,pipe,state,firing)
    def verify(*values: Any) -> Any:
        s,f,p,unused,trace = values
        assert not firing,'combined_unexpected_firing'
        return B.HR.verify(s,f,p,f.controller.hidden_history_rows,trace)
    try:
        value = FunctionType(R.drive.__code__,dict(vars(R),I=I,install=install,verify=verify))(*args)
        C.accepted(state,factory,I.FRAMES[-1])
        return value|dict(combined_installer=True,same_scope_guard=True)
    finally:
        C.saved(state,factory)
        B.K.write(state['output']/'COMBINED_FIRING.json',firing)


def main() -> int:
    helper = SimpleNamespace(**(vars(B.K)|dict(guards=C.guards,
        load=B.CURRENT.palette_fixture.loader(B.K.load))))
    return FunctionType(R.main.__code__,dict(vars(R),ROOT=B.ROOT,K=helper,I=I,drive=drive))()


if __name__=='__main__':
    raise SystemExit(main())
