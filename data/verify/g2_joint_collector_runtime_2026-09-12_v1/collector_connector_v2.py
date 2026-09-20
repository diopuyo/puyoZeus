"""原connectの前後に設定とproducer捕捉を接続し、原raw/tail検査を保持する。"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace as N
from typing import Any
import raw_upgrade as RAW

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_observed_start_anchor_2026-09-12_v1'))
_spec = importlib.util.spec_from_file_location('_g2_joint_collector_start_capture_v2', ROOT / 'anchor_v2.py')
ANCHOR = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ANCHOR
_spec.loader.exec_module(ANCHOR)
import integrated_connection as INTEGRATED
import tail_auth as M

FIRST, LAST = 34772, 35410
BASE_GLOBALS = dict(vars(INTEGRATED.OLD))
BASE_CONNECT = INTEGRATED.OLD.connect
C = RAW.B


def connect(stack: Any, state: dict) -> Any:
    proxy = N(**(vars(C) | dict(build=RAW.build)))
    original = FunctionType(BASE_CONNECT.__code__, dict(BASE_GLOBALS, C=proxy, M=M, FIRST=FIRST, LAST=LAST))
    loop = original(stack, state)
    rec = state['provisional_context_observer']
    capture = ANCHOR.install(stack, loop, state['reset_metadata_tail'],
                             dict(source_id=rec.source_id, run_id=rec.run_id))
    state['joint_producer_capture'], state['joint_capture_stack'] = capture, stack
    stack.push(final_capture(state, capture))
    return loop


def final_capture(state: dict, capture: Any) -> Any:
    def close(kind: Any, body: Any, trace: Any) -> bool:
        try:
            if capture.last is None:
                if body is None: raise ValueError('joint_no_capture')
                return False
            value = capture.snapshot()
            path = state['output'] / 'JOINT_PRODUCER_CAPTURE.json'
            with path.open('x', encoding='utf-8') as stream:
                json.dump(value, stream, ensure_ascii=False)
        except BaseException as error:
            state['joint_capture_save_error'] = repr(error)
            if body is None: raise
        return False
    return close


def install(stack: Any) -> None:
    original = INTEGRATED.OLD
    facade = N(**(vars(original) | dict(vars(sys.modules[__name__]))))
    facade.restore = original.restore
    def restore() -> None:
        if INTEGRATED.OLD is not facade:
            raise ValueError('joint_connector_replaced')
        INTEGRATED.OLD = original
    stack.callback(restore)
    INTEGRATED.OLD = facade


