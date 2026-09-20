"""原CPU driveの省略loopだけを原採録まで連続するloopへ置換する。"""
from __future__ import annotations
from contextlib import ExitStack
import json
import sys
from types import SimpleNamespace as N
from typing import Any
import collector_continuous as C
import metadata_tail as M

FIRST, LAST = 34772, 34980
RESTORE: list[Any] = []


def connect(stack: Any, state: Any) -> Any:
    sink = state['collector_metadata_sink']
    bounded = sys.modules[type(sink).__module__]
    original = bounded.O
    assert sink.rows.count==0 and not sink.busy and not sink.errors
    before = original.FIRST,original.LAST
    RESTORE.append(lambda: (setattr(original,'FIRST',before[0]),setattr(original,'LAST',before[1])))
    original.FIRST,original.LAST = FIRST,LAST
    M.install(stack,state)
    loop = C.build(sink.collector,original.SOURCE)
    stack.callback(loop.generator.close)
    state['complete_cpu_collector'] = loop
    return loop


def wrap(original: Any) -> Any:
    def drive(q: Any, m: Any, state: Any, real: Any, fixture: Any, binding: Any, factory: Any) -> Any:
        with ExitStack() as stack:
            loop = connect(stack,state)
            facade = N(**(vars(m)|dict(loop=N(collector_loop=lambda: loop))))
            try:
                return original(q,facade,state,real,fixture,binding,factory)
            finally:
                stream = state['collector_metadata_sink'].rows
                value = dict(actual_original_downstream_loop=True, artificial_input_pipeline=False,
                    artificial_pixels_and_initial_history=True, metadata_rows=stream.count,
                    actual_append_count=stream.append_count, configurations=loop.configuration,
                    quality_gate_clear=False, actual_full_video=False)
                with (state['output']/'CPU_COLLECTOR_CONNECTION.json').open('x',encoding='utf-8') as handle:
                    json.dump(value,handle,ensure_ascii=False,indent=2,default=str)
    return drive


def restore() -> None:
    while RESTORE: RESTORE.pop()()
