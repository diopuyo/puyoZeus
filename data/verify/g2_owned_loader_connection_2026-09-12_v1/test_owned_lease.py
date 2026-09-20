"""実owned生成から選ばれたsideを、既存の原Recovery/会計rigへ接続する。"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import sys
from typing import Any
import pytest
import owned_adapter as R
from probe_after import collector

# 原型は凍結collectorと同じsrcから読む。fixtureは人工盤面/lease資格のまま。


def old_rig(stack: Any) -> tuple:
    # 実configured後に検査専用の同名candidateを一時注入し、原aliasを戻す。
    with pytest.MonkeyPatch.context() as patch:
        patch.delitem(sys.modules, 'candidate', raising=False)
        patch.syspath_prepend(str(R.VERIFY / 'g2_side_lease_connection_2026-09-11_v1'))
        import test_selected_side_lease as base
    generator = base.rig.__wrapped__()
    value = next(generator)
    stack.callback(generator.close)
    return base, value


def test_owned_generation_to_original_recovery() -> None:
    assert '_g2_actual_side_lease_selected' not in sys.modules, 'lease_preprimed'
    generic = R.A.L.dependencies
    with ExitStack() as stack:
        R.configured(stack)
        collector()
        context = sys.modules[R.OWNED_ALIAS].build(object, end_frame=36298)
        BASE, rig = old_rig(stack)
        globals_used = [cls.__dict__['perform'].__globals__ for cls in context.__mro__ if 'perform' in cls.__dict__]
        selected = next(value for value in globals_used if value.get('__file__') == str(R.SIDE / 'side_actual_connection.py'))
        module = selected['selected']()
        assert Path(module.parts().helper.V1.__file__).resolve() == R.SIDE / 'side_reset_op.py'
        pipe = rig['pipe']
        pipe._chain_start_next_1p, pipe._chain_start_next_2p = (4, 3), (3, 4)
        lease, recovery, events = BASE.objects(rig)
        shared = module.parts().join._vrepr(pipe._ojama_fall_accounting_tracker)
        other = rig['runtime'].histories['2P']
        with ExitStack() as operation:
            module.install(operation, lease)
            lease.perform(recovery, 6120, 102.0)
            assert events == ['qualify', 'archive'] and recovery.reset_count == 1
            assert recovery.pending is not None and lease.outer_calls == lease.native_calls == 0
            assert pipe._chain_start_next_1p is None and pipe._chain_start_next_2p == (3, 4)
            assert rig['runtime'].histories['2P'] is other
            assert module.parts().join._vrepr(pipe._ojama_fall_accounting_tracker) == shared
        assert 'perform' not in vars(lease)
    assert R.A.L.dependencies is generic and R.SIDE_ALIAS not in sys.modules
