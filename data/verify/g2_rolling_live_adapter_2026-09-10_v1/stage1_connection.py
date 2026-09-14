"""原二段検査の最初のJ世代だけを既存厳格連鎖へ接続する。"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Iterator
import sys


def install(stack: Any, previous: Any, bridge: Any) -> None:
    fusion = sys.modules['_conditional_live_finalizer'].F
    original = fusion.modules
    @contextmanager
    def modules() -> Iterator[Any]:
        with original() as (stage1, world), bridge.installed(stage1) as adapted:
            yield adapted, world
    previous.patch(stack, fusion, 'modules', modules)
