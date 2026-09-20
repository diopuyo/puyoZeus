"""派生した実extendのfinallyを通す。上流setupはstub、元例外保持だけを検査。"""
from __future__ import annotations
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace as N
from typing import Any
import pytest

RESET_ROOT = Path(__file__).resolve().parents[1]/'g2_empty_tail_reset_integration_2026-09-11_v1'
sys.path.insert(0, str(RESET_ROOT))
import entry_continuation as OLD
import entry_continuation_v2 as NEW


@pytest.mark.parametrize('fixed', [False, True])
def test_original_install_exception_not_masked(tmp_path: Path, fixed: bool) -> None:
    holder = type('Holder', (), {'__module__': '_fake_recovery_setup'})()
    lease = N(evidence=object(), events=[])
    recovery = N(rows=[], pending=None, error=None)
    module = N(I=object(), install=lambda *args: recovery, recording=lambda *args: None)
    q = N(KEPT={'prefix_evidence': object()}, recovery=lambda *args: holder)
    state = dict(repeat_scope_guard=N(reset_lease=lease), output=tmp_path,
        postcommit_current_receiver=N(issued=0, released=0))
    def fail(*args: Any) -> Any: raise RuntimeError('primary-install-error')
    derived = NEW.derived() if fixed else OLD.derived()
    values = dict(derived.__globals__, Q=q, I=N(RESET=34932, shifted=lambda value: value),
        sys=N(modules={'_fake_recovery_setup': module}), ENTRY=N(install=fail))
    function = FunctionType(derived.__code__, values)
    with ExitStack() as stack:
        context = dict(state=state, factory=object(), pipe=object(), stack=stack,
            clock={}, cap=object(), sink=object())
        expected = RuntimeError if fixed else UnboundLocalError
        with pytest.raises(expected) as caught:
            function(context, {})
    if fixed:
        assert str(caught.value) == 'primary-install-error'
        saved = json.loads((tmp_path/'EMPTY_RESET_CONTINUATION.json').read_bytes())
        assert saved['entry_boundary'] is None
    else:
        assert str(caught.value.__context__) == 'primary-install-error'
