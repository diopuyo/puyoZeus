"""旧CPU入力は保持し、実測MROの中間3層だけ新Coreへ置換する。"""
from __future__ import annotations
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
END_FRAME = 36298


def owned(alias: str, name: str) -> Any:
    path = ROOT / name
    if alias in sys.modules:
        value = sys.modules[alias]
        assert Path(value.__file__).resolve() == path, 'core_fixture_alias'
        return value
    spec = importlib.util.spec_from_file_location(alias, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[alias] = value
    spec.loader.exec_module(value)
    return value


class Deferred:
    """perform開始までcold依存を解決しない。実liveと同じローダーを使う。"""

    def __init__(self) -> None:
        self.parts: Any = None

    def resolve(self) -> Any:
        if self.parts is None:
            self.parts = owned('_g2_core_fixture_loader', 'loader.py').dependencies()
        return self.parts

    def modules(self) -> Any:
        return self.resolve().modules()

    @property
    def quarantine(self) -> Any:
        return self.resolve().quarantine

    def votes(self) -> Any:
        return self.resolve().votes()


def replace(result: Any, refs: dict) -> Any:
    hidden = result.OLD.BASE.H
    prior = hidden.Context
    expected = json.loads((ROOT / 'CPU_COMPOSITION_v1.json').read_bytes())['context_mro_before_main_anchor']
    actual = [dict(name=c.__qualname__, module=c.__module__,
                   perform_source=None if 'perform' not in vars(c) else inspect.getsourcefile(c.perform),
                   perform_line=None if 'perform' not in vars(c) else c.perform.__code__.co_firstlineno)
              for c in prior.__mro__]
    assert actual == expected and len(actual) == 9, 'original_fixture_mro_changed'
    classes = prior.__mro__
    snapshot, baseline, occurrence, side = (sys.modules[c.__module__] for c in classes[:4])
    deferred = Deferred()
    core = owned('_g2_core_fixture_context', 'context.py').core(classes[-2], deferred, END_FRAME)
    current = snapshot.derived(baseline.derived(occurrence.derived(side.derived(core))), side)
    # 原mainが外側へarchive anchorを一度付ける。ここでは追加しない。
    hidden.Context = current
    refs.update(core=core, configured=current, original=prior, deferred=deferred, hidden=hidden)
    return result
