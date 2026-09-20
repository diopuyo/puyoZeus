"""元session_creatorの親保存経路を保ち、run_v6生成点のみ正常側へ置換する。"""
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
ORIGINAL_CREATE = ROOT.parent / 'g2_belief_publication_runtime_2026-09-11_v1/run_v6.py'


def owned_load(load: Any, owned: dict, alias: str, path: Any, injection: Any) -> Any:
    before = sys.modules.get(alias)
    if alias in owned and before is not owned[alias]:
        raise ValueError('normal_selected_alias_changed:' + alias)
    try:
        return load(alias, path, injection)
    finally:
        value = sys.modules.get(alias)
        if before is None and value is not None and Path(value.__file__).resolve() == Path(path).resolve():
            owned[alias] = value


def creator(original: Any, owner: Any, stack: Any) -> Any:
    owned: dict = {}
    def close(kind: Any, body: Any, trace: Any) -> bool:
        foreign = []
        for alias, value in owned.items():
            if sys.modules.get(alias) is value:
                sys.modules.pop(alias)
            elif alias in sys.modules:
                foreign.append(alias)
        if foreign and body is None:
            raise ValueError('normal_selected_restore:' + ','.join(foreign))
        return False
    stack.push(close)
    def create(load: Any, frames: tuple[int, ...]) -> Any:
        bootstrap = owner.bootstrap()
        if bootstrap.load is not load:
            raise ValueError('normal_selected_initial_owner')
        def selected(alias: str, path: Any, injection: Any = None) -> Any:
            if Path(path).resolve() != ORIGINAL_CREATE:
                return owned_load(load, owned, alias, path, injection)
            prior = owned_load(load, owned, alias + '_normal_original', path, injection)
            dependencies = owner.dependencies().modules()
            settings = owned_load(load, owned, '_normal_selected_settings', ROOT / 'normal_settings.py', None)
            dependencies.binding.B.require(frames == settings.EARLIEST, 'normal_creator_frames')
            return owned_load(load, owned, alias, ROOT / 'normal_create.py',
                        dict(original_run_v6=prior, normal_dependencies=dependencies, normal_settings=settings))
        def restore(kind: Any, body: Any, trace: Any) -> bool:
            if bootstrap.load is selected:
                bootstrap.load = load
            elif body is None:
                raise ValueError('normal_selected_loader_changed')
            return False
        stack.push(restore)
        bootstrap.load = selected
        return original(selected, frames)
    return create
