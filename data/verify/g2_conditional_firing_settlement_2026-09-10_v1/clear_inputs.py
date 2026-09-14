"""発火後の人工消去rawだけを追加し、元70入力の発火以前は保持する。"""
from __future__ import annotations
from types import FunctionType,SimpleNamespace
from typing import Any
import run_entry as E

FIRST_CLEAR,LAST=34912,34924
FRAMES=tuple(range(E.I.FRAMES[0],LAST+E.I.STRIDE,E.I.STRIDE))


def saved() -> Any:
    base,raw=E.I.saved()
    final=E.I.generated()['final']
    observed=tuple(tuple(E.I.P.UNKNOWN if r==0 and c==0 else v for c,v in enumerate(row))
        for r,row in enumerate(final))
    raw.update({frame:observed for frame in FRAMES if frame>=FIRST_CLEAR})
    return base,raw


def install(*args: Any) -> Any:
    return FunctionType(E.I.install.__code__,dict(vars(E.I),saved=saved))(*args)


INPUT=SimpleNamespace(**(vars(E.I)|dict(FRAMES=FRAMES,saved=saved,install=install)))
