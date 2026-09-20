"""原lease資格は保持し、内部固定操作とperform時時計だけを渡す未接続候補。"""
from __future__ import annotations
import builtins
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace as N
from typing import Any
import side_lease as S

ROOT = Path(__file__).resolve().parent.parent
QUALIFICATION_LABEL = 'original-empty-lease'
PINS = {
    'original': ('g2_historical_chain_next_reset_2026-09-12_v1/side_reset_op.py',
                 '49a8f8d3048620e193639b5e43f7c1cc1ccf4088d1a49ced2d5bda520a921be4'),
    'helper': ('g2_side_reset_exception_v2_2026-09-11_v1/helper_v2.py',
               '6d5d863352f3733ef59dc3cd9eee0e54a33c39e78ee24449260061f236eaca1d'),
    'join': ('g2_fixed_side_join_2026-09-11_v1/fixed_join.py',
             '571c6b4566454f765f38e745b3c7803d10e153ffb10cc802f54792c0cacc932f'),
}


def load(key: str, injections: dict[str, Any]) -> Any:
    relative, sha = PINS[key]
    path, alias = ROOT / relative, '_g2_selected_side_' + key
    S.require(hashlib.sha256(path.read_bytes()).hexdigest() == sha, 'selected_source_' + key)
    if alias in sys.modules:
        module = sys.modules[alias]
        S.require(Path(module.__file__).resolve() == path.resolve(), 'selected_alias')
        return module
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    original = builtins.__import__
    def importing(name: str, globals: Any = None, locals: Any = None,
                  fromlist: Any = (), level: int = 0) -> Any:
        return injections[name] if level == 0 and name in injections else original(name, globals, locals, fromlist, level)
    module.__dict__['__builtins__'] = dict(vars(builtins), __import__=importing)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    if key == 'join':
        peers = [m for m in tuple(sys.modules.values()) if m is not None and m is not module
                 and getattr(m, '__file__', None) and Path(m.__file__).resolve() == path.resolve()]
        sets = {id(m._IN_PROGRESS): m._IN_PROGRESS for m in peers}
        S.require(len(sets) <= 1, 'split_join_reentrancy')
        if sets:
            module._IN_PROGRESS = next(iter(sets.values()))
    return module


def parts() -> Any:
    original = load('original', {})
    helper = load('helper', {'side_reset_op': original})
    return N(helper=helper, join=load('join', {'side_reset_op': helper}))


def epoch(lease: Any, runtime: Any) -> tuple[Any, ...]:
    """scope[2]は対象NEXT世代、scope[5]は対象SMのreset世代。"""
    guard, journal = lease.guard, lease.guard.factory.provider.journal
    scope = lease.evidence.scope(guard.factory, guard.pipe)
    S.require(scope[2] == journal.epoch(guard.pipe, '1P') == runtime.histories['1P'].epoch,
              'selected_epoch_sources')
    return scope


def preflight(lease: Any, modules: Any, clock: float) -> tuple[Any, ...]:
    guard, join = lease.guard, modules.join
    journal = guard.factory.provider.journal
    controller, pipe, tracker = journal.controller, guard.pipe, journal.tracker
    S.require(guard.controller is controller, 'selected_controller')
    runtime = controller.instances.get(id(pipe))
    S.require(runtime is not None, 'selected_native_runtime_missing')
    try:
        S.require(controller._runtime(pipe) is runtime, 'selected_runtime_identity')
    except (ValueError, RuntimeError) as error:
        raise join.SideJoinRejected('native_runtime_rejected', {'error': repr(error)}) from error
    path = Path(join.__file__).resolve()
    peers = [m for m in tuple(sys.modules.values()) if m is not None and getattr(m, '__file__', None)
             and Path(m.__file__).resolve() == path]
    if any(m._IN_PROGRESS is not join._IN_PROGRESS for m in peers):
        raise join.SideJoinRejected('split_join_reentrancy', {'side': '1P'})
    native, error = join.native_history_type(controller)
    reason = error or join._reject_preconditions(controller, pipe, runtime, tracker, '1P', native)
    if reason is not None or id(runtime) in join._IN_PROGRESS:
        raise join.SideJoinRejected(reason or 'reentrant_join', {'side': '1P'})
    modules.helper.reset_recognition_side(pipe, '1P', now_sec=clock,
                                         qualification=QUALIFICATION_LABEL, dry_run=True)
    managed = frozenset(name for name, _ in modules.helper._names('1p'))
    before, error = join._safe_snapshot(pipe, tracker, runtime, '1P', managed)
    if before is None:
        raise join.SideJoinRejected('pre_snapshot_failed', {'snapshot_error': error})
    return controller, pipe, tracker, runtime, epoch(lease, runtime)


def install(stack: Any, lease: Any) -> None:
    """外部operation/side/時計を受け取らず、原perform引数から時計を一回だけ運ぶ。"""
    S.require('perform' not in vars(lease), 'foreign_perform')
    modules, active = parts(), {}
    def operation() -> Any:
        S.require(bool(active), 'operation_outside_perform')
        controller, pipe, tracker, runtime, before = active['inputs']
        result = modules.join.join_fixed_side_reset(controller, pipe, tracker, '1P',
            now_sec=active['clock'], qualification=QUALIFICATION_LABEL)
        after = epoch(lease, runtime)
        S.require(after[2] == before[2] + 1 and after[5] == before[5] + 1, 'selected_generation')
        return result
    fixed = S.build(lease, operation)
    def perform(self: Any, recovery: Any, frame: int, clock: float) -> Any:
        S.require(self is lease and not active, 'selected_reentrant_perform')
        inputs = preflight(lease, modules, clock)  # 原Recoveryへ入る前。無変更拒否を偽成功にしない。
        active.update(inputs=inputs, clock=clock)
        try:
            return fixed(self, recovery, frame, clock)  # 原qualify/archive/失敗処理は元のまま。
        finally:
            active.clear()
    wrapper = MethodType(perform, lease)
    lease.perform = wrapper
    def restore() -> None:
        S.require(vars(lease).get('perform') is wrapper, 'foreign_restore')
        del lease.perform
    stack.callback(restore)
