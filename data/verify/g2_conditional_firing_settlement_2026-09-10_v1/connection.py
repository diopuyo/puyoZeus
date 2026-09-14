"""同原Jの消去票→精算→原自然STABLEを接続する有限条件付きレーン。"""
from __future__ import annotations
from pathlib import Path
from types import FunctionType,MethodType,SimpleNamespace
from typing import Any
import importlib.util
import sys
import conditional_settlement as S
import origin_connection as O
import clear_observation as C
import observed_connection as OBS
import formula_connection as FC

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_conditional_firing_lifecycle',ROOT/'lifecycle.py')
L=importlib.util.module_from_spec(spec); sys.modules[spec.name]=L
spec.loader.exec_module(L)


def delayed_formula(stack: Any, factory: Any, pipe: Any, patch: Any, rows: list[Any]) -> None:
    import firing_handoff_fixed as F
    def formula(self: Any, side: str, time_sec: float, prev_confirmed: Any, binding: Any, frame: Any) -> Any:
        assert sys._getframe(1) is frame
        actual=F.V5.V4.bound_update(self); caller=frame.f_back
        assert caller.f_code is actual.__code__ and caller.f_globals is actual.__globals__
        assert caller.f_locals['self'] is self is pipe and caller.f_locals['time_sec']==time_sec
        assert actual.__globals__['__next_live'] is factory.provider.journal.controller
        saved=L.registered(factory.controller,binding)
        assert side=='1P' and not self._pending_tsumo_1p and self._active_chain_1p is None
        rows.append(dict(stage='conditional_formula_continuation_deferred',frame=caller.f_locals['frame_idx'],
            origin_id=saved['origin'].origin_id,original_event=False,physical_certified=False))
    FC.install(stack,pipe,patch,formula)


def install(stack: Any, factory: Any, pipe: Any, patch: Any, rows: list[Any]) -> None:
    def policy_type(accounting: Any) -> type:
        return S.policy_type(O.P.policy_type(accounting))
    policy=SimpleNamespace(**(vars(O.P)|dict(policy_type=policy_type)))
    FunctionType(O.install.__code__,dict(vars(O),P=policy))(stack,factory,pipe,patch,rows)
    def observe(control: Any, binding: Any, view: Any, signals: Any, sm: Any, pipe: Any, rows: Any) -> Any:
        saved=L.registered(control,binding)
        if saved.get('settled',False): return None
        observed=C.observe(control,binding,view,signals,sm,pipe,rows)
        if observed is not None:
            result=S.complete(control,binding,observed)
            binding.conditional_firing_origin['settled']=True
            rows.append(result)
        return observed
    OBS.install(stack,factory,patch,observe,rows)
    L.install(stack,factory,patch,rows)
    delayed_formula(stack,factory,pipe,patch,rows)
    cls=type(factory.controller); old_hand=cls.hand
    def hand(self: Any, binding: Any, view: Any) -> Any:
        if getattr(binding,'conditional_firing_registered',None) is not None:
            L.registered(self,binding)
            assert not view.refs and not view.tokens and not view.queue,'conditional_new_NEXT_not_connected'
        return old_hand(self,binding,view)
    patch(stack,cls,'hand',hand)
