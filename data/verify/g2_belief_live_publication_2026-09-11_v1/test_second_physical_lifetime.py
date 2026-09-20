"""物理更新直後の原writer失敗を握り潰さず、公開資格を停止して復元する。"""
from __future__ import annotations

from contextlib import ExitStack
from typing import Any
import pytest
from test_second_physical import saved,live,context,policy,observed,physical,setup,step,consumption
from test_second_tracking import advance
import second_tracking as T
import reflection as R


def test_original_writer_failure_blocks_publication(observed: Any,policy: Any,physical: Any) -> None:
    o = observed
    mode = setup(o,policy,physical)
    board = o.pipe._sm_2p.context.confirmed_board.copy()
    board.set(12,0,5)
    board.set(11,0,5)
    advance(o)
    original = o.witness.journal.complete_step
    error = OSError('original writer failed')
    class Writer:
        def write(self,text: str) -> None:
            raise error
    o.witness.journal.stream = Writer()
    with ExitStack() as stack:
        T.install(stack,mode)
        with pytest.raises(OSError) as caught:
            step(o,board,'step:601',consumption(mode))
        assert caught.value is error and mode.error is error
        assert len(mode.applied)==1  # 私有状態の更新は起きたが、公開は拒否する。
        value = o.registry.current(mode.connection.binding)
        with pytest.raises(ValueError,match='reflection_owner'):
            R.verify(mode,value,'step:601',value.frame)
    assert mode.closed and o.witness.journal.complete_step == original


def test_body_exception_and_hook_restoration(observed: Any,policy: Any,physical: Any) -> None:
    o = observed
    mode = setup(o,policy,physical)
    original = o.witness.journal.complete_step
    error = RuntimeError('outer body failed')
    with pytest.raises(RuntimeError) as caught:
        with ExitStack() as stack:
            T.install(stack,mode)
            raise error
    assert caught.value is error and mode.closed
    assert o.witness.journal.complete_step == original
