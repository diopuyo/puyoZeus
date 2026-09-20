"""cold生成と2P専用型の選択/復元。Session基部は人工で全constructorの代用ではない。"""
from contextlib import ExitStack
import io
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import owned_adapter as A


def test_cold_selected_second_and_cleanup(tmp_path: Path) -> None:
    with ExitStack() as stack:
        A.configured(stack)
        import probe_native_merge
        owner = sys.modules[A.A.A.A.V4.OWNED_ALIAS]
        parts = owner.dependencies().modules()
        first = parts.mode.Mode.__new__(parts.mode.Mode)
        first.connection, first.native, first.activation = N(binding=None), None, None
        first.arrival_ledger, first.error, first.rows, first.stream = None, None, 0, io.StringIO()
        class Session:
            def __init__(self, stack: Any, context: Any, policy: Any, physical: Any,
                         contract: Any, members: Any, frames: Any) -> None:
                self.physical, self.state = physical, context['state']
        binding = owner.bootstrap().load('_notice_test_binding', A.CANDIDATE / 'second_mode_binding.py')
        stack.callback(lambda: sys.modules.pop('_notice_test_binding', None))
        selected = binding.session_class(N(Session=Session), parts.mode)
        context = dict(state=dict(probabilistic_tracking_mode=first,
            probabilistic_basis_connection=first.connection, output=tmp_path))
        value = selected(stack, context, None, N(Mode=parts.mode.Mode), None, None, ())
        assert issubclass(value.physical.Mode, parts.mode.BASE.Mode)
        assert value.physical.Mode is not parts.mode.BASE.Mode
        assert Path(value.physical.Mode.capture_origin.__code__.co_filename).name == 'settled_notice.py'
        assert Path(parts.mode.Mode.capture_origin.__code__.co_filename).name == 'prefix_live_adapter.py'
        again = owner.dependencies().modules()
        assert again.mode is parts.mode
        stream = value.second_settled_notice_stream
    assert stream.closed
    assert not [k for k in sys.modules if k.startswith(('_g2_second_notice_', '_g2_prefix_'))]
