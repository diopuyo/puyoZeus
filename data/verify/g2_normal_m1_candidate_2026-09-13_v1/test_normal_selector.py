"""原session_creatorから正常createを実選択し、未起動の親保存経路/型/解放を確認。"""
from contextlib import ExitStack
from pathlib import Path
import sys
from typing import Any
import test_normal_basis as F
parts = F.parts
ROOT = Path(__file__).resolve().parent


def test_original_session_creator_selects_normal(parts: Any) -> None:
    whole = F.T.A.A.A.A.V4.A
    owner = sys.modules[F.T.A.A.A.A.V4.OWNED_ALIAS]
    selector = parts.loader('_normal_selector_test', ROOT / 'normal_selector.py')
    before = set(sys.modules)
    with ExitStack() as stack:
        selected = selector.creator(whole.D.session_creator, owner, stack)
        create = selected(parts.loader, (33726, 33766))
        assert callable(create)
        runtime = sys.modules['_whole_session_create_v6']
        assert Path(runtime.__file__).resolve() == ROOT / 'normal_create.py'
        assert runtime.inner_mode() is parts.original.mode.BASE.Mode
        assert runtime.inner_mode() is not parts.original.mode.Mode
        assert sys.modules['_whole_runtime_session'].V6 is runtime
        assert sys.modules['_whole_runtime_session'].PUBLICATION is sys.modules['_whole_parent_publication']
    assert '_whole_session_create_v6_normal_original' not in sys.modules
    assert '_whole_session_create_v6' not in sys.modules
    assert '_normal_selected_settings' not in sys.modules
