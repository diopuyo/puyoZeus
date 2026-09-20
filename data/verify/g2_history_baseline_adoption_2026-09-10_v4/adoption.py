"""既存factoryが作ったprovider/controllerへ初期化前の観測だけを追加する。"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
ALIAS = '_history_baseline_existing_slot_witness'


def load() -> Any:
    if ALIAS in sys.modules:
        raise RuntimeError('adoption_module_collision')
    spec = importlib.util.spec_from_file_location(ALIAS,ROOT/'witness.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[ALIAS] = module
    spec.loader.exec_module(module)
    return module


def bind(module: Any, provider: Any, controller: Any) -> Any:
    module.require('baseline' not in vars(controller), 'adoption_baseline_already_wrapped')
    expected = provider._parts.C.Controller.baseline
    module.require(controller.baseline.__func__ is expected, 'adoption_original_baseline_required')
    witness = module.Witness(provider,controller)
    previous_attach = provider.attach
    def attached(stack: Any, current: Any) -> None:
        module.require(current is controller and not witness.closed, 'adoption_controller_binding')
        previous_attach(stack,current)
        stack.callback(setattr,provider,'attach',previous_attach)
        journal,original = provider.journal,controller.baseline
        emit = journal.emit
        def captured(row: Any) -> None:
            emit(row)
            try:
                witness.capture(row,sys._getframe(1))
            except BaseException as error:
                witness.error = witness.error or repr(error)
                raise
        def baseline(sm: Any, view: Any, raw: Any, side: str) -> Any:
            try:
                return witness.baseline(original,sm,view,raw,side)
            except BaseException as error:
                witness.error = witness.error or repr(error)
                raise
        stack.callback(witness.close)
        stack.callback(setattr,journal,'emit',emit)
        stack.callback(vars(controller).pop,'baseline',None)
        journal.emit,controller.baseline = captured,baseline
    provider.attach = attached
    return witness
