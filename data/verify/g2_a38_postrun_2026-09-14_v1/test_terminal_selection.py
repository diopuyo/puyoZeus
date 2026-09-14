"""実arrival_loader/汎用loaderへの型接続と原参照復元。実Session実走は別。"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace as N
import pytest
import terminal_selection as C
import test_first_terminal_candidate as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'g2_arrival_ack_candidate_2026-09-12_v1'))
import test_arrival_loader as A


def test_original_loader_types_and_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = N(__file__=str(ROOT.parent/'g2_prefix_lane_integration_2026-09-13_v1/prefix_final_saved.py'),
        verify=lambda *a: {'artificial_prefix':True})
    previous_prefix = prefix.verify
    monkeypatch.setitem(sys.modules, '_g2_prefix_final', prefix)
    parts = N(belief=F.F.base.B, mode=F.F.base)
    with ExitStack() as stack:
        load = C.wrap_load(A.LOADER.load, lambda: stack, F.F.A.CORE.patch)
        selected = A.V.build(parts, load)
        mode, source, saved = selected.mode, selected.mode.E, selected.arrival_saved
        assert mode.BASE is parts.mode
        assert mode.Mode.__module__ == 'first_terminal_candidate'
        assert source.Capture.__module__ == 'first_terminal_candidate'
        assert prefix.verify.__module__ == 'terminal_selection'
        assert saved.verify.__module__ == '_g2_arrival_final_saved'
        second = load('_post38_second_connection_test',
            ROOT.parent/'g2_second_terminal_arrival_2026-09-14_v1/session_connection.py')
        assert second.mode_class.__module__ == 'second_inactive_candidate'
        assert A.V.build(parts, load).mode is mode  # 同loaderの再取得は再deriveしない。
    assert mode.Mode.__module__ == '_g2_arrival_mode'
    assert source.Capture.__module__ == '_g2_arrival_source'
    assert saved.verify.__module__ == '_g2_arrival_final_saved'
    assert prefix.verify is previous_prefix
    assert second.mode_class.__module__ == '_post38_second_connection_test'


def test_distinct_module_lifetimes_and_closed_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = N(__file__=str(ROOT.parent/'g2_prefix_lane_integration_2026-09-13_v1/prefix_final_saved.py'),
        verify=lambda *a: {'artificial_prefix':True})
    monkeypatch.setitem(sys.modules, '_g2_prefix_final', prefix)
    second = ROOT.parent/'g2_second_terminal_arrival_2026-09-14_v1/session_connection.py'
    saved = ROOT.parent/'g2_arrival_ack_candidate_2026-09-12_v1/arrival_final_saved.py'
    values = {second:N(__file__=str(second), mode_class=lambda base, session:base),
        saved:N(__file__=str(saved), verify=lambda *a: {'unchanged':True})}
    original_second, original_saved = values[second].mode_class, values[saved].verify
    with ExitStack() as root, ExitStack() as nested:
        current = [root]
        load = C.wrap_load(lambda alias,path,injection=None:values[path], lambda:current[0], F.F.A.CORE.patch)
        load('second', second)
        current[0] = nested
        load('saved', saved)
        assert load('second', second) is values[second]  # root所有の型は内側で再利用可能。
        nested.close()
        assert values[saved].verify is original_saved
        with pytest.raises(ValueError, match='module_replaced_or_closed'):
            load('saved', saved)
    assert values[second].mode_class is original_second
