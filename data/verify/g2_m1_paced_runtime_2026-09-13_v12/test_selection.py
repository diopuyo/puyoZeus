"""実private生成で休止classが選択され、終了時に原classへ復元される。"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import owned_adapter as R


def test_actual_private_driver_selection() -> None:
    with ExitStack() as stack:
        R.configured(stack)
        module = R.A.A.A.V4.A.DRIVER
        selected = module.Driver
        original = selected.__mro__[2]
        assert Path(selected.__call__.__code__.co_filename).resolve() == R.PACING_ROOT / 'frame_pacing.py'
        assert Path(selected.__mro__[1].close.__code__.co_filename).resolve() == R.BASE / 'driver_status.py'
        assert Path(original.__call__.__code__.co_filename).name == 'whole_session_driver.py'
    assert module.Driver is original
