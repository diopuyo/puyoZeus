"""既存共通設置/原constructorを再用し、新接続の復元対象を明示する。"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import inspect
import sys

ROOT=Path(__file__).resolve().parent
NORMAL=ROOT.parent/'g2_conditional_full_noninterference_2026-09-10_v1'
sys.path.insert(0,str(NORMAL))
import run_repeated as R


def references(base: Any,factory: Any,pipe: Any,state: Any) -> tuple[Any,...]:
    cls=type(factory.controller); v1=cls.prepared.__globals__['V1']
    lifecycle=R.B.C.L
    accounting=inspect.getclosurevars(factory.controller.inventory.BoundPolicy.__init__).nonlocals['p']
    return base.references(factory,pipe,state)+(v1.prepared,cls.prepared,cls.call,cls.consumed_history,
        lifecycle.registered,lifecycle.certificate,accounting.BoundPolicy,
        pipe._apply_chain_formula_early_fire)


def policy_module(base: Any) -> Any:
    """実v2が呼ぶ元Cを解決し、別moduleを捕捉しない。"""
    directory=ROOT.parent/'g2_firing_hand_connection_2026-09-10_v1/settled_current_v1'
    actual=base.P.C.install
    assert Path(actual.__globals__['__file__']).resolve()==directory/'settled_connection_v2.py'
    assert R.DISPATCH.F.code_value(actual.__code__)==R.DISPATCH.F.code_value(
        R.DISPATCH.F.declared(directory/'settled_connection_v2.py',('install',)))
    module=actual.__globals__['C']
    assert Path(module.__file__).resolve()==directory/'settled_connection.py'
    assert module.install_policy.__globals__ is vars(module)
    assert R.DISPATCH.F.code_value(module.install_policy.__code__)==R.DISPATCH.F.code_value(
        R.DISPATCH.F.declared(directory/'settled_connection.py',('install_policy',))),'runtime_install_policy_code'
    return module


def install(base: Any,derive: Any,stack: Any,factory: Any,pipe: Any,state: Any,rows: Any) -> None:
    """原revision引数だけを限定交換し、scope/commonを二重設置しない。"""
    assert callable(derive),'conditional_runtime_verified_revision_required'
    assert 'conditional_full_installs' not in state,'conditional_runtime_duplicate_install'
    state['conditional_full_rows']=[]; state['conditional_full_installs']=0
    module=policy_module(base); original=module.install_policy; calls=[]
    def connect(inner: Any,control: Any,patch: Any,old_revision: Any,output: Any) -> None:
        assert not calls and control is factory.controller,'conditional_runtime_revision_scope'
        path=ROOT.parent/'g2_settled_current_revision_2026-09-10_v1/recover.py'
        assert R.DISPATCH.F.code_value(old_revision.__code__)==R.DISPATCH.F.code_value(
            R.DISPATCH.F.declared(path,('recover_settled_current',)))
        revision=derive(old_revision)
        assert revision.__code__ is old_revision.__code__ and revision.__closure__ is old_revision.__closure__
        calls.append(old_revision)
        original(inner,control,patch,revision,output)
    with ExitStack() as temporary:
        R.H.TAIL.patch(temporary,module,'install_policy',connect)
        R.installed(stack,factory,pipe,state,rows)
    assert len(calls)==1 and module.install_policy is original,'conditional_runtime_revision_restored'
    state['conditional_revision_connection']=dict(installs=len(calls),original_install_policy_restored=True)


def adapted(stack: Any,derive: Any) -> Any:
    base=R.H.K.load('_conditional_constructor_original_candidate',R.H.REPEAT/'connection.py',stack)
    def owned(inner: Any,factory: Any,pipe: Any,state: Any,rows: Any) -> None:
        install(base,derive,inner,factory,pipe,state,rows)
    def capture(factory: Any,pipe: Any,state: Any) -> Any:
        return references(base,factory,pipe,state)
    def guards() -> dict[str,str]:
        assert callable(derive),'conditional_runtime_verified_revision_required'
        fixed=derive.__globals__['K']
        paths=[*ROOT.glob('*.py'),ROOT/'PLAN.md',*fixed.ROOT.glob('*.py')]
        return R.guards()|fixed.guards()|{str(p):R.H.K.sha(p) for p in paths}
    return N(**(vars(base)|dict(install=owned,references=capture,guards=guards)))
