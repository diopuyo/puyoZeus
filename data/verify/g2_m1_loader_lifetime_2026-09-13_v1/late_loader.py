"""原createへ渡すローダーを、同じ所有bootstrapの現行参照へ遅延委譲する。"""
from __future__ import annotations
from pathlib import Path
import inspect
import sys
from typing import Any, Callable

PATCH = Path(__file__).resolve().parent.parent / 'g2_m1_completion_candidate_2026-09-12_v1/runtime_patch.py'


def install(stack: Any, dependencies: Any, owner: Any, replace: Callable[..., Any]) -> None:
    """生成時刻は変えず、後発の検収済みフックを保存時にも通す。"""
    original = dependencies.session_creator
    active = [True]

    def creator(load: Any, frames: tuple[int, ...]) -> Any:
        bootstrap = owner.bootstrap()
        if bootstrap.load is not load:
            raise ValueError('late_loader_initial_owner')
        accepted: list[Any] = []

        def current(alias: str, path: Any, injection: Any = None) -> Any:
            if not active[0] or owner.bootstrap() is not bootstrap:
                raise ValueError('late_loader_closed_or_foreign')
            if sys.modules.get(bootstrap.__name__) is not bootstrap:
                raise ValueError('late_loader_module_replaced')
            selected = bootstrap.load
            if accepted and selected is not accepted[0]:
                raise ValueError('late_loader_hook_replaced_or_released')
            if selected is not load:
                closure = inspect.getclosurevars(selected).nonlocals
                if Path(selected.__code__.co_filename).resolve() != PATCH or closure.get('original') is not load:
                    raise ValueError('late_loader_unowned_hook')
                if not accepted:
                    accepted.append(selected)
            return selected(alias, path, injection)

        return original(current, frames)

    stack.callback(lambda: active.__setitem__(0, False))
    replace(stack, dependencies, 'session_creator', creator)
