"""v20を保持したまま2P終了通知の非消費分類だけを追加する。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'g2_m1_prefix_runtime_2026-09-13_v20'
NOTICE = ROOT.parent / 'g2_second_origin_postrun_2026-09-13_v1'
SPEC = importlib.util.spec_from_file_location('_g2_v21_previous', PREVIOUS / 'owned_adapter.py')
PREVIOUS_ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREVIOUS_ADAPTER)
A = PREVIOUS_ADAPTER.A
PRIOR, START, SIDE, protect = A.PRIOR, A.START, A.SIDE, A.protect
CANDIDATE, SAFETY, BASE = PREVIOUS_ADAPTER.CANDIDATE, PREVIOUS_ADAPTER.SAFETY, PREVIOUS_ADAPTER.BASE
UNBOUND_MODULE = PREVIOUS_ADAPTER.UNBOUND_MODULE


def sources() -> tuple[Path, ...]:
    return tuple(sorted(set(PREVIOUS_ADAPTER.sources()) | {ROOT / 'owned_adapter.py'} |
        {NOTICE / name for name in ('settled_notice.py', 'notice_saved.py', 'notice_selection.py')}))


def install_private(stack: Any, bootstrap: Any) -> None:
    load = bootstrap.load
    aliases = ('_g2_second_notice_v21', '_g2_second_notice_saved_v21', '_g2_second_notice_select_v21')
    if any(alias in sys.modules for alias in aliases):
        raise ValueError('notice_private_foreign_alias')
    owned: dict[str, Any] = {}
    def release(kind: Any, body: Any, trace: Any) -> bool:
        changed = []
        for alias, value in reversed(tuple(owned.items())):
            if sys.modules.get(alias) is value:
                sys.modules.pop(alias)
            elif alias in sys.modules:
                changed.append(alias)
        if changed and body is None:
            raise ValueError('notice_private_alias_changed')
        return False
    stack.push(release)
    def take(alias: str, path: Path, injection: Any = None) -> Any:
        try:
            return load(alias, path, injection)
        finally:
            value = sys.modules.get(alias)
            if value is not None and Path(value.__file__).resolve() == path:
                owned[alias] = value
    notice = take(aliases[0], NOTICE / 'settled_notice.py')
    saved = take(aliases[1], NOTICE / 'notice_saved.py')
    selection = take(aliases[2], NOTICE / 'notice_selection.py',
                     dict(settled_notice=notice, notice_saved=saved))
    selection.install_load(stack, bootstrap, A.A.A.V4.replace_owned)


def configured(stack: Any) -> Any:
    selected = PREVIOUS_ADAPTER.configured(stack)
    owner = sys.modules[A.A.A.V4.OWNED_ALIAS]
    original = owner.bootstrap
    installed: list[Any] = []
    def bootstrap() -> Any:
        value = original()
        if not installed:
            install_private(stack, value)
            installed.append(value)
        elif installed[0] is not value:
            raise ValueError('notice_bootstrap_changed')
        return value
    A.A.A.V4.replace_owned(stack, owner, 'bootstrap', bootstrap)
    return selected
