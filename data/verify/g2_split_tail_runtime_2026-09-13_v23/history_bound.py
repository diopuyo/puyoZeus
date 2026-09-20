"""実envの元Recorderだけへ排他上限を束縛し、時計検査とsuperを保持する。"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import FunctionType
from typing import Any, Iterator

FIRST, LAST, STRIDE, FPS = 29052, 36900, 2, 60
END_EXCLUSIVE = LAST + STRIDE
OLD_END = 36300
SOURCE = Path(__file__).resolve().parents[3] / 'scripts/diagnose_video38_accounting_history_v1.py'


def bind(stack: Any, history: Any, replace: Any) -> Any:
    original = history.HistoryRecorder.begin_frame
    assert Path(history.__file__).resolve() == SOURCE, 'history_bound_source'
    assert Path(original.__code__.co_filename).resolve() == SOURCE, 'history_bound_method'
    assert original.__globals__ is vars(history), 'history_bound_original_globals'
    assert tuple(original.__globals__[k] for k in ('FIRST_FRAME', 'STRIDE', 'FPS', 'END_FRAME')) == (
        FIRST, STRIDE, FPS, OLD_END), 'history_bound_original_constants'
    selected = FunctionType(original.__code__, dict(original.__globals__, END_FRAME=END_EXCLUSIVE),
                            original.__name__, original.__defaults__, original.__closure__)
    selected.__kwdefaults__ = original.__kwdefaults__
    selected.__annotations__ = dict(original.__annotations__)
    assert selected.__code__ is original.__code__ and selected.__closure__ is original.__closure__
    replace(stack, history.HistoryRecorder, 'begin_frame', selected)
    return original


def install(stack: Any, main: Any, replace: Any) -> None:
    session = main.__globals__['S']
    original = session.configured
    @contextmanager
    def configured() -> Iterator[Any]:
        with original() as env:
            history = env['runtime'].M.A.entry.previous.history
            with ExitStack() as inner:
                before = bind(inner, history, replace)
                assert history.HistoryRecorder.begin_frame.__globals__['END_FRAME'] == END_EXCLUSIVE
                yield env
            assert history.HistoryRecorder.begin_frame is before, 'history_bound_restore'
    replace(stack, session, 'configured', configured)
