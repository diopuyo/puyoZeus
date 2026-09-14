"""既存runtime_patchの型選択factoryだけを私有loader内で置換する。"""
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'g2_m1_completion_candidate_2026-09-12_v1/second_mode_binding.py'
ALIAS = '_g2_unbound_second_binding'


def install_load(stack: Any, bootstrap: Any, replace: Any) -> None:
    """source path/所有moduleを確認して元loadの終了時復元を使う。"""
    original = bootstrap.load
    if ALIAS in sys.modules:
        raise ValueError('unbound_binding_foreign_alias')
    owned: list[Any] = []

    def release(kind: Any, body: Any, trace: Any) -> bool:
        if owned and sys.modules.get(ALIAS) is owned[0]:
            sys.modules.pop(ALIAS)
        elif owned and body is None:
            raise ValueError('unbound_binding_alias_changed')
        return False

    stack.push(release)

    def load(alias: str, path: Any, injection: Any = None) -> Any:
        value = original(alias, path, injection)
        if Path(path).resolve() == SOURCE:
            if owned and sys.modules.get(ALIAS) is not owned[0]:
                raise ValueError('unbound_binding_alias_changed')
            try:
                patched = original(ALIAS, ROOT / 'second_mode_binding.py')
            finally:
                partial = sys.modules.get(ALIAS)
                partial_path = getattr(partial, '__file__', None)
                if not owned and isinstance(partial_path, (str, Path)) and Path(partial_path).resolve() == ROOT / 'second_mode_binding.py':
                    owned.append(partial)
            if value.session_class is not patched.session_class:
                replace(stack, value, 'session_class', patched.session_class)
        return value

    replace(stack, bootstrap, 'load', load)


def install(stack: Any, owner: Any, replace: Any) -> None:
    """cold Board初期化後の最初のbootstrap取得でのみ選択を設置する。"""
    original = owner.bootstrap
    installed: list[Any] = []

    def bootstrap() -> Any:
        value = original()
        if not installed:
            install_load(stack, value, replace)
            installed.append(value)
        elif installed[0] is not value:
            raise ValueError('unbound_bootstrap_owner_changed')
        return value

    replace(stack, owner, 'bootstrap', bootstrap)
