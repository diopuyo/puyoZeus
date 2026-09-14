"""初回Noneの原J世代検査だけを元conditional Stage1へ接続する。権限は追加しない。"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import FunctionType, ModuleType, SimpleNamespace as N
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT.parent/'g2_private_suffix_finalizer_stage1_2026-09-10_v1'
PINS = {
    'private_fixed.py': '042ceba922b8076a9adb09a30843e8852fcb53b4dedbf8310baa96b46e3e4a73',
    'private_generation.py': 'eb5dfc5875bc94c9c6102079419ac1cae14897c77e1239c5deaab4148c826f9f'}


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError('rolling_stage1_bridge:'+reason)


def libraries() -> Any:
    previous = sys.modules.get('private_fixed')
    modules = {}
    try:
        for name, digest in PINS.items():
            path = PRIVATE/name
            require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'generation_source')
            spec = importlib.util.spec_from_file_location('_rolling_bridge_'+name[:-3], path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules[name] = module
            if name == 'private_fixed.py':
                sys.modules['private_fixed'] = module
        return N(F=modules['private_fixed.py'], G=modules['private_generation.py'])
    finally:
        if previous is None:
            sys.modules.pop('private_fixed', None)
        else:
            sys.modules['private_fixed'] = previous


def verify(stage: Any, lib: Any) -> None:
    fixed = lib.F
    require(type(stage) is ModuleType and Path(stage.__file__).resolve() == fixed.OLD/'compat.py', 'stage_module')
    require(all(fixed.sha(fixed.OLD/name) == value for name, value in fixed.PINS.items()), 'stage_source')
    for name in fixed.PINS:
        module = stage if name == 'compat.py' else sys.modules.get(name[:-3])
        require(type(module) is ModuleType, 'stage_dependency:'+name)
        fixed.module_check(module, fixed.OLD/name)
    require(stage.J is sys.modules['conditional_finalizer_journal']
            and stage.S is sys.modules['conditional_finalizer_state']
            and stage.F is sys.modules['conditional_finalizer_fixed'], 'stage_import_identity')
    require(stage.S.J is stage.J and stage.S.F is stage.J.F is stage.F, 'state_import_identity')


def clone(original: Any, **changed: Any) -> Any:
    value = FunctionType(original.__code__, dict(original.__globals__, **changed),
                         original.__name__, original.__defaults__, original.__closure__)
    value.__kwdefaults__ = original.__kwdefaults__
    return value


def derive(original: Any, *, events: list[Any] | None = None) -> Any:
    """原audit同bodyに専用S facadeを渡す。原module/旧globalsは無変更。"""
    stage = sys.modules.get(original.__globals__.get('__name__'))
    lib = libraries()
    verify(stage, lib)
    require(original is stage.audit_stage1 and original.__globals__ is vars(stage), 'original_audit_identity')
    recorded = [] if events is None else events
    def created(audit: Any, token: str, item: Any) -> Any:
        result = lib.G.created(stage.J, audit, token, item)
        if result['generation']['action_revision'] is None:
            recorded.append(dict(token=token, created_frame=result['frame_idx'], consumed_frame=item['frame_idx'],
                initial_revision=None, coerced=False, original_J_chain_verified=True,
                physical_identity_certified=False))
        return result
    journal = N(**vars(stage.J))
    journal.created = created
    journal.pop = clone(stage.J.pop, created=created)
    state = N(**vars(stage.S))
    state.advance = clone(stage.S.advance, J=journal)
    state.verify = clone(stage.S.verify, J=journal, advance=state.advance)
    return clone(original, S=state)


@contextmanager
def installed(stage1: Any) -> Iterator[Any]:
    """FUSION.modules返却への一時facade。旧selected/通常分岐/参照は一切patchしない。"""
    lib, events = libraries(), []
    verify(stage1, lib)
    modules = [stage1, stage1.S, stage1.J]
    originals = [(module, dict(vars(module))) for module in modules]
    facade = N(**vars(stage1))
    facade.audit_stage1 = derive(stage1.audit_stage1, events=events)
    facade.nullable_generation_checks = events
    try:
        yield facade
    finally:
        verify(stage1, lib)
        require(all(set(vars(module)) == set(before)
                    and all(vars(module)[key] is value for key, value in before.items())
                    for module, before in originals), 'original_globals_changed')
