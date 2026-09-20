"""独立Opusの許容clock反例を、原enqueueと原更新成功完了で閉鎖する。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
from typing import Any
import pytest

path = Path(__file__).resolve().parent/'opus_observer_v1/test_opus_observer_qa.py'
spec = importlib.util.spec_from_file_location('_desync_original_clock_qa', path)
Q = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = Q
spec.loader.exec_module(Q)
modules, h = Q.modules, Q.h


@pytest.mark.parametrize('side', ['1P', '2P'])
@pytest.mark.parametrize('skew', [0.0, Q.SKEW, -Q.SKEW])
def test_actual_clock_preserved_and_signal_delivered(h: Any, side: str, skew: float) -> None:
    observer = Q.V.install(h.stack, h.adapter, side)
    Q.prefix(h, side, skew=skew)
    Q.drive_skewed(h, Q.F+8, Q.YP, Q.PP, quiet=True, side=side, skew=skew)
    signal = observer.take(h.pipe, Q.F+8)
    assert signal is not None and signal.invocation.time_sec == (Q.F+8)/60+skew
    assert signal.invocation.runtime.completed == Q.F+8
    assert [f.frame for f in signal.facts] == [Q.F+6, Q.F+8]
    assert not getattr(h.pipe, '_pending_tsumo_'+side.lower())
    assert not signal.reset_permission and not signal.current_permission
    assert observer.take(h.pipe, Q.F+8) is None
    Q.T.invariant(h, side)
