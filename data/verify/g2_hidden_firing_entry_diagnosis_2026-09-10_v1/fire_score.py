"""既存人工OCR/式画像の時計だけを延長fixtureへ移す。元認識flagは書き換えない。"""
from __future__ import annotations
from types import FunctionType
from typing import Any
import sys
import fire_inputs as I

sys.path.insert(0,str(I.VERIFY/'g2_chain_firing_continuous_cpu_2026-09-10_v1'))
import artificial_inputs as O
import constructor_input as C


class Clock:
    """既存OCRのFIREを基準に入力時計を平行移動する。"""
    def __init__(self, clock: Any) -> None:
        self.clock=clock

    def __getitem__(self, key: str) -> int:
        assert key=='frame'
        return self.clock[key]-I.FIRE+O.FIRE


def install(stack: Any, pipe: Any, clock: Any) -> Any:
    return O.score_input(stack,pipe,Clock(clock))


def paint(cap: Any, frame: int) -> None:
    O.paint_score(cap,frame-I.FIRE+O.FIRE)


def configured(fixture: Any) -> Any:
    """元palette注入器の外へ保存PLAN由来3flagを追加する。"""
    from types import SimpleNamespace
    plan=I.H.B.K.VERIFY/'video38_history_publication_probe_live_2026-09-10_v6/PLAN.json'
    source=I.H.B.K.read(plan)['actual_collector_kwargs']
    kwargs={name:source[name] for name in C.NAMES}
    assert all(type(v) is bool and v for v in kwargs.values())
    palette=I.H.B.CURRENT.palette_fixture.configured(fixture)
    def original_real(original: Any, evidence: Any) -> Any:
        formula: dict[str,Any]={}
        evidence['formula_constructor_fixture']=formula
        return palette.install.__globals__['original_real'](C.transport(original,kwargs,formula),evidence)
    return SimpleNamespace(**(vars(palette)|dict(install=FunctionType(palette.install.__code__,
        dict(palette.install.__globals__,original_real=original_real)))))
