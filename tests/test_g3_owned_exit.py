"""二重ExitStackの元例外、二次失敗、正常対照と開始失敗を検査する。"""
from __future__ import annotations
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace as N
from typing import Any
import pytest
from scripts import g3_owned_exit as E


@pytest.mark.parametrize('mode', ['normal', 'primary', 'secondary', 'both', 'install'])
def test_exception_preservation(mode: str) -> None:
    """元releaseと同じbody分岐を通す。失敗を成功へ変換しない。"""
    seen = []
    primary = LookupError('original_failure')
    def install(stack: Any, *args: Any) -> None:
        def release(kind: Any, body: Any, trace: Any) -> bool:
            seen.append(body)
            if mode in ('secondary', 'both', 'install'):
                raise OSError('cleanup_failure')
            return False
        stack.push(release)
        if mode == 'install':
            raise primary
    candidate = N(references=lambda *a: (), install=install,
                  P=N(Q=N(R=N(Q=N(installed=lambda *a: nullcontext())))))
    evidence = dict(rows=[], qualification={})
    caught = None
    try:
        with ExitStack() as stack:
            E.attach(stack, candidate, None, object(), {}, evidence)
            if mode in ('primary', 'both'):
                raise primary
    except BaseException as error:
        caught = error
    if mode in ('primary', 'both', 'install'):
        assert caught is primary and seen == [primary]
    elif mode == 'secondary':
        assert isinstance(caught, OSError) and seen == [None]
    else:
        assert caught is None and seen == [None]
    assert evidence['closed'] and evidence['references_restored']
    assert bool(evidence['cleanup_errors']) == (mode in ('secondary', 'both', 'install'))
