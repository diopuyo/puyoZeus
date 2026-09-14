"""同じ共通candidateを原110/86へ接続し、実constructor前の結合を検証する。"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
from types import FunctionType,SimpleNamespace as N
from typing import Any
import importlib.util
import sys
import candidate as C

ROOT=C.ROOT
COMPAT=ROOT.parent/'g2_conditional_settled_current_compat_2026-09-10_v1'


def load(stack: Any) -> Any:
    paths=list(sys.path); stack.callback(sys.path.__setitem__,slice(None),paths)
    sys.path.insert(0,str(COMPAT))
    module=C.R.H.K.load('_conditional_shared_compat',COMPAT/'connection.py',stack)
    return C.adapted(stack,module.derive)


def normal_drive(original: Any,facade: Any,*args: Any) -> Any:
    state,factory=args[3],args[7]
    try:
        result=FunctionType(original.drive.__code__,dict(vars(original),C=facade))(*args)
        C.R.COMMON.accepted(state,factory,original.I.FRAMES[-1])
        control=factory.controller
        assert not control.hidden_history_rows and not control.hidden_current_outputs
        assert not control.conditional_next_hands and not state['conditional_full_rows']
        assert all(getattr(b,'conditional_firing_registered',None) is None for b in control.history.values())
        return result
    finally:
        C.R.COMMON.saved(state,factory)
        C.R.H.K.write(state['output']/'CONDITIONAL_FULL.json',dict(
            installs=state.get('conditional_full_installs',0),rows=state.get('conditional_full_rows',[]),
            conditional_checkpoint_count=len(getattr(factory.controller,'conditional_next_hands',{})),
            revision_connection=state.get('conditional_revision_connection'),
            physical_certified=False,quality_gate_clear=False))


def normal() -> int:
    with ExitStack() as stack:
        facade=load(stack)
        old=C.R.H.K.load('_shared_original_normal110',C.R.H.REPEAT/'run_cpu.py',stack)
        def drive(*args: Any) -> Any: return normal_drive(old,facade,*args)
        def accepted(output: Path) -> Any: return C.R.accepted(old,output)
        source=old.P.Q.R
        execute=FunctionType(source.execute.__code__,dict(vars(source),I=old.I,
            MAX_UPDATES=old.MAX_UPDATES,drive=drive))
        common=N(**(vars(old.K)|dict(guards=facade.guards)))
        return FunctionType(old.C.G.R.main.__code__,dict(vars(old.C.G.R),K=common,
            ROOT=ROOT,execute=execute,accepted=accepted))()


def conditional() -> int:
    with ExitStack() as stack:
        facade=load(stack)
        old=C.R.H.K.load('_shared_original_conditional86',C.R.NEXT/'run_next.py',stack)
        def loop(args: Any,inner: Any,data: Any) -> None:
            def install(scope: Any,factory: Any,pipe: Any,state: Any,rows: Any) -> None:
                facade.install(scope,factory,pipe,state,rows)
                data['admission']=state['conditional_full_rows']
            def already_installed(*values: Any) -> None: pass
            shared=N(**(vars(old.E.H.C)|dict(install=install)))
            history=N(**(vars(old.E.H)|dict(C=shared)))
            execution=N(**(vars(old.E)|dict(H=history)))
            FunctionType(old.loop.__code__,dict(vars(old),E=execution,
                C=N(install=already_installed)))(args,inner,data)
        def drive(*args: Any) -> Any:
            state,pipe,factory=args[2],args[3][0],args[6]
            before=facade.references(factory,pipe,state)
            try: return FunctionType(old.drive.__code__,dict(vars(old),loop=loop))(*args)
            finally: assert facade.references(factory,pipe,state)==before,'shared_conditional_restore'
        def guards() -> Any: return old.guards()|facade.guards()
        return FunctionType(old.main.__code__,dict(vars(old),ROOT=ROOT,drive=drive,guards=guards))()


if __name__=='__main__':
    mode=sys.argv.pop(1)
    assert mode in ('normal','conditional')
    raise SystemExit(normal() if mode=='normal' else conditional())
