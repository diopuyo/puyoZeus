"""実v4の原Observer生成直後へ非変更診断を付ける。原型と期限を維持する。"""
from __future__ import annotations
from dataclasses import replace
import importlib.util
import inspect
from pathlib import Path
import sys
from types import CellType, FunctionType
from typing import Any

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'g2_session_contract_runtime_2026-09-12_v4'
DIAG = ROOT.parent / 'g2_settled_basis_actual_mismatch_2026-09-12_v1'
sys.path.insert(0, str(DIAG))
import basis_observation_connection as C

SPEC = importlib.util.spec_from_file_location('_g2_basis_observation_base_adapter', BASE / 'owned_adapter.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
PRIOR, START, SIDE = A.PRIOR, A.START, A.SIDE
protect = A.protect


def sources() -> tuple[Path, ...]:
    extra = (ROOT / 'owned_adapter.py', BASE / 'owned_adapter.py',
             DIAG / 'basis_observation_connection.py', DIAG / 'basis_local_observation.py')
    return tuple(sorted(set(A.sources()) | set(extra)))


def wrap_modules(original: Any, stack: Any, owned: dict) -> Any:
    def modules() -> Any:
        parts = original()
        actual = parts.actual
        if actual in owned:
            C.require(actual.install is owned[actual], 'actual_install_changed')
            return parts
        previous = actual.install
        C.require(isinstance(previous, FunctionType) and previous.__globals__ is vars(actual) and
                  Path(previous.__code__.co_filename).resolve() == Path(actual.__file__).resolve(), 'actual_install_origin')
        def install(stack: Any, recovery: Any, state: dict, reset_frame: int, deadline: int) -> Any:
            C.require(actual.install is install, 'actual_install_owner')
            observer = previous(stack, recovery, state, reset_frame, deadline)
            C.require(type(observer) is actual.Observer and state['settled_basis_observer'] is observer,
                      'actual_observer_selection')
            C.require('basis_local_observation' not in state, 'duplicate_observation')
            state['basis_local_observation'] = C.install(stack, observer, state['output'])
            return observer
        A.replace_owned(stack, actual, 'install', install)
        owned[actual] = install
        return parts  # 原module/容器/型を置換せず、所有installだけを一度装着。
    return modules


def configured(stack: Any) -> Any:
    selected = A.configured(stack)
    owner = sys.modules[A.OWNED_ALIAS]
    original = owner.dependencies
    owned: dict[Any, Any] = {}
    def dependencies() -> Any:
        A.require_owned(owner)
        parts = original()
        return replace(parts, modules=wrap_modules(parts.modules, stack, owned))
    # v4のbuild code/型/guardをそのまま実行し、選択済み依存closureだけを更新する。
    previous_build = owner.build
    bindings = inspect.getclosurevars(previous_build).nonlocals
    C.require(Path(previous_build.__code__.co_filename).resolve() == BASE / 'owned_adapter.py'
              and previous_build.__globals__ is vars(A) and bindings['owner'] is owner
              and bindings['selected_dependencies'] is original
              and bindings['original_build'].__globals__ is vars(owner), 'original_build_owner')
    A.replace_owned(stack, owner, 'dependencies', dependencies)
    C.require(previous_build.__closure__ is not None and
              previous_build.__code__.co_freevars.count('selected_dependencies') == 1, 'dependency_cell')
    closure = tuple(CellType(dependencies) if name == 'selected_dependencies' else cell
                    for name, cell in zip(previous_build.__code__.co_freevars, previous_build.__closure__))
    build = FunctionType(previous_build.__code__, previous_build.__globals__, previous_build.__name__,
                         previous_build.__defaults__, closure)
    build.__kwdefaults__ = previous_build.__kwdefaults__
    build.__qualname__, build.__module__ = previous_build.__qualname__, previous_build.__module__
    build.__dict__.update(previous_build.__dict__)
    A.replace_owned(stack, owner, 'build', build)
    return selected
