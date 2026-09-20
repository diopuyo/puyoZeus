"""前段資格は固定原updateのcode/globalsと実呼出frameに限定する。"""
from __future__ import annotations
from functools import lru_cache
import hashlib
from pathlib import Path
import sys
from types import CodeType
from typing import Any
import firing_ticket as OLD
from firing_fixed import code_value

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/src/recognition_pipeline.py'
SOURCE_SHA = '6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02'
ORIGINAL = OLD.qualified


@lru_cache(maxsize=1)
def expected() -> CodeType:
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA, 'firing_source_changed'
    module = compile(SOURCE.read_bytes(), str(SOURCE), 'exec', dont_inherit=True)
    cls = next(c for c in module.co_consts if type(c) is CodeType and c.co_name == 'RecognitionPipeline')
    return next(c for c in cls.co_consts if type(c) is CodeType and c.co_name == 'update')


def qualified(frame: Any, factory: Any, environment: Any) -> Any:
    assert sys._getframe(1) is frame, 'firing_front_frame_identity'
    caller = frame.f_back
    assert Path(environment['__file__']).resolve() == SOURCE, 'firing_front_module'
    assert caller.f_globals is environment and code_value(caller.f_code) == code_value(expected()), 'firing_front_code'
    value = ORIGINAL(frame, factory, environment)
    value.source.update(caller_code_verified=True, caller_globals_verified=True, immediate_frame_verified=True)
    return value
