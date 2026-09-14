"""継続2手所有時は旧tailへfallbackせず、原消費後にのみ来歴を昇格。"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import rolling_prepare as R
import rolling_commit as C


def base_from_call(function: Any, history: Any) -> Any:
    """元prefix callの実closureから既設置Held moduleを取得する。"""
    seen = set()
    while function not in seen:
        seen.add(function)
        values = dict(zip(function.__code__.co_freevars, (cell.cell_contents for cell in function.__closure__ or ())))
        if (Path(function.__code__.co_filename).name == 'prefix_connection.py'
            and function.__globals__.get('H') is history):
            return values['base']
        assert 'old_call' in values, 'rolling_original_prefix_call_missing'
        function = values['old_call']
    raise AssertionError('rolling_original_call_cycle')


def guard_calls(stack: Any, control: Any, patch: Any, provenance: Any) -> None:
    """元callの返却後、native popに渡す前の実callで全保持票を照合する。"""
    cls, original = type(control), type(control).call
    control.hidden_rolling_prepop_checks = []
    def call(self: Any, caller: Any, binding: Any, view: Any, pipe: Any, side: str, proposal: Any) -> Any:
        result = original(self, caller, binding, view, pipe, side, proposal)
        if proposal is not None and 'rolling_previous_votes' in proposal:
            assert len(view.queue) == provenance.PAIR_SIZE, 'rolling_pop_already_happened'
            state = binding.owner.state
            C.before(provenance, self, result)
            assert binding.owner.state is state and len(view.queue) == provenance.PAIR_SIZE
            self.hidden_rolling_prepop_checks.append(dict(frame=view.frame, side=side,
                token=proposal['proof']['token'], evidence_id=proposal['evidence'].event_id,
                queue_slots=len(view.queue), state_action=state.action, current_permission=False))
        return result
    patch(stack, cls, 'call', call)


def install(stack: Any, factory: Any, patch: Any, history: Any, provenance: Any, owner: Any) -> None:
    from src import puyo_core_bridge as core
    # 汎用origin名を既存loaderへ混入させない。
    import importlib.util
    spec = importlib.util.spec_from_file_location('_rolling_origin', Path(__file__).with_name('origin.py'))
    origin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(origin)
    origin.verify(history)
    control, cls = factory.controller, type(factory.controller)
    v1 = cls.prepared.__globals__['V1']
    original, old_prepare, old_consume = R.derive(history), v1.prepared, cls.consumed_history
    base = base_from_call(cls.call, history)
    def prepared(self: Any, binding: Any, item: Any, sm: Any, raw: Any, signals: Any, view: Any) -> Any:
        owned = (getattr(binding, 'hidden_prefix_consumed', False)
            and not getattr(binding, 'hidden_tail_consumed', False) and len(view.refs) == 2)
        if owned:
            return R.prepare(original, provenance, owner, v1.L, base.D, core,
                self, binding, item, sm, signals, view)
        return old_prepare(self, binding, item, sm, raw, signals, view)
    def consumed(self: Any, call: Any, caller: Any) -> None:
        if 'rolling_previous_votes' in call['prepared']:
            return C.consumed(old_consume, provenance, self, call, caller)
        return old_consume(self, call, caller)
    patch(stack, v1, 'prepared', prepared)
    patch(stack, cls, 'consumed_history', consumed)
    guard_calls(stack, control, patch, provenance)
