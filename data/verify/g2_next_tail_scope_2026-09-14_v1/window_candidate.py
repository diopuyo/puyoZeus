"""元観測窓の束縛時に、同じ実NEXT moduleの尾部だけを延長する。"""
import importlib.util
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'g2_history_prediction_diagnosis_2026-09-13_v1/observer_window_candidate.py'
NATIVE = ROOT.parents[2]/'scripts/next_enqueue_live_shadow_v1.py'
NATIVE_NAME = 'scripts.next_enqueue_live_shadow_v1'
PROVIDER = ROOT.parent/'g2_directional_next_provider_2026-09-09_v2/provider.py'
spec = importlib.util.spec_from_file_location('_a34_original_observer_window',SOURCE)
ORIGINAL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ORIGINAL)
PATHS = ORIGINAL.PATHS
LAST, OLD_LAST = ORIGINAL.LAST, ORIGINAL.OLD_LAST
STRIDE = 2
FIRST, FRAMES = ORIGINAL.FIRST, ORIGINAL.FRAMES
NEXT_FIRST = 32494  # 原NEXT修復の下限は保持し、今回拡張しない。


def native() -> Any:
    value = sys.modules.get(NATIVE_NAME)
    if value is None or Path(value.__file__).resolve()!=NATIVE:
        raise ValueError('next_tail_native_owner')
    function = value.NextEnqueueController.enqueue
    if Path(function.__code__.co_filename).resolve()!=NATIVE or function.__globals__ is not vars(value):
        raise ValueError('next_tail_native_function_owner')
    return value


def bind_native(stack: Any, replace: Any) -> Any:
    value = native()
    if (value.FIRST_FRAME,value.LAST_FRAME,value.STRIDE,value.BASE.RESET_FRAME)!=(NEXT_FIRST,OLD_LAST,STRIDE,NEXT_FIRST):
        raise ValueError('next_tail_original_bounds')
    replace(stack,value,'LAST_FRAME',LAST)
    return value


def install(stack: Any, main: Any, replace: Any) -> None:
    original_bind = ORIGINAL.bind
    def bind(owner_stack: Any, latest: Any, owner_replace: Any) -> dict:
        values = original_bind(owner_stack,latest,owner_replace)
        # 元設定ではここでまだ未import。後続の原installと同じ正規moduleを先に読む。
        from scripts import next_enqueue_live_shadow_v1
        bind_native(owner_stack,owner_replace)
        return values
    replace(stack,ORIGINAL,'bind',bind)
    ORIGINAL.install(stack,main,replace)


def verify_state(state: dict) -> dict:
    result = ORIGINAL.verify_state(state)
    value = native()
    binding = state['directional_next_runtime']
    controller,adapter,provider = (binding[k] for k in ('controller','adapter','provider'))
    if (type(controller) is not value.NextEnqueueController or adapter.native is not value
        or adapter.controller is not controller or provider.controller is not controller
        or not adapter.enabled or value.LAST_FRAME!=LAST):
        raise ValueError('next_tail_live_connection')
    if controller.instances or controller.active is not None:
        raise ValueError('next_tail_started_before_verification')
    provider_module = sys.modules.get(type(provider).__module__)
    if provider_module is None or Path(provider_module.__file__).resolve()!=PROVIDER:
        raise ValueError('next_tail_provider_owner')
    return result|dict(next_last=LAST,next_original_last=OLD_LAST,
        next_first=value.FIRST_FRAME,next_verified_before_first_invocation=True)
