"""G3予測入力窓のFIRST_FRAME差し替えを、人工のadapter/entryで検査する。

実走ランナー・実動画・実G2ファイルの読み込みは一切行わない。
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any

import pytest

from scripts import g3_native_scope as S


def replace(stack: Any, owner: Any, name: str, value: Any) -> None:
    """テスト専用の単純replace。原scriptの`adapter.A.A.A.V4.replace_owned`の代用。"""
    old = getattr(owner, name)
    stack.callback(setattr, owner, name, old)
    setattr(owner, name, value)


def make_native(tmp_path: Path, first: int = 32494, last: int = 36900) -> Any:
    """next_enqueue_live_shadow_v1.py相当の人工native module。"""
    path = tmp_path / 'fake_next_enqueue_live_shadow_v1.py'
    path.write_text('# fake native module for test\n', encoding='utf-8')
    return N(FIRST_FRAME=first, LAST_FRAME=last, __file__=str(path),
              __name__='scripts.next_enqueue_live_shadow_v1')


def make_adapter(native: Any, original_bind: Any = None) -> Any:
    """adapter.OC.ORIGINAL / adapter.OC.native() / adapter.A.A.A.V4.replace_owned だけを持つ人工adapter。"""
    window = N(bind=original_bind or (lambda owner_stack, latest, owner_replace: {'ok': True}))
    oc = N(ORIGINAL=window, native=lambda: native)
    v4 = N(replace_owned=replace)
    chain = N(A=N(A=N(V4=v4)))
    return N(OC=oc, A=chain)


def make_entry(first_frame: int = 0) -> Any:
    k = N(bounds=lambda: {'first_frame': first_frame})
    return N(__globals__={'K': k})


def test_install_returns_zero_replaced_before_any_bind(tmp_path: Path) -> None:
    """window.bindが一度も呼ばれない間は、票はreplaced==0のままである。"""
    native = make_native(tmp_path)
    adapter, entry = make_adapter(native), make_entry()
    with ExitStack() as stack:
        receipt = S.install(stack, adapter, entry)
        assert receipt == dict(before_first=None, before_last=None, after_first=None,
                                replaced=0, restored=None, native_module=None, native_source_sha=None)
        assert native.FIRST_FRAME == 32494


def test_bind_patches_first_only_and_restores_on_scope_exit(tmp_path: Path) -> None:
    """FIRST_FRAMEだけが実走開始frameへ変わり、LAST_FRAMEは触らず、scope終了で元へ戻る。"""
    native = make_native(tmp_path, first=32494, last=36900)
    adapter, entry = make_adapter(native), make_entry(first_frame=0)
    with ExitStack() as stack:
        receipt = S.install(stack, adapter, entry)
        window = adapter.OC.ORIGINAL
        with ExitStack() as inner:
            values = window.bind(inner, None, replace)
            assert values == {'ok': True}
            assert native.FIRST_FRAME == 0  # entry.K.bounds()['first_frame']
            assert native.LAST_FRAME == 36900  # 触っていない
            assert receipt['before_first'] == 32494
            assert receipt['before_last'] == 36900
            assert receipt['after_first'] == 0
            assert receipt['replaced'] == 1
            assert receipt['native_module'] == 'scripts.next_enqueue_live_shadow_v1'
            assert receipt['native_source_sha'] is not None
        assert native.FIRST_FRAME == 32494  # innerスコープ終了で復元
        assert receipt['restored'] is True


def test_unexpected_before_first_raises_named_error(tmp_path: Path) -> None:
    """firstが32494でなければ、実値付きで名前つきに落ちる。lastは問わない。"""
    native = make_native(tmp_path, first=999, last=36900)
    adapter, entry = make_adapter(native), make_entry()
    with ExitStack() as stack:
        S.install(stack, adapter, entry)
        window = adapter.OC.ORIGINAL
        with pytest.raises(ValueError, match='native_scope_unexpected:999'):
            window.bind(stack, None, replace)
    assert native.FIRST_FRAME == 999  # 差し替え前に落ちたので無変更


def test_duplicate_bind_rejected(tmp_path: Path) -> None:
    """window.bindが二重に呼ばれたら、二度目は名前つきで落ちる (二重束縛の防止)。"""
    native = make_native(tmp_path)
    adapter, entry = make_adapter(native), make_entry()
    with ExitStack() as stack:
        S.install(stack, adapter, entry)
        window = adapter.OC.ORIGINAL
        window.bind(stack, None, replace)
        with pytest.raises(ValueError, match='native_scope_duplicate_bind'):
            window.bind(stack, None, replace)


def test_last_frame_untouched_even_when_larger_than_expected(tmp_path: Path) -> None:
    """last値がどんな実測値でも(36900以外でも)、記録するだけで書き換えない。"""
    native = make_native(tmp_path, first=32494, last=40000)
    adapter, entry = make_adapter(native), make_entry(first_frame=5)
    with ExitStack() as stack:
        receipt = S.install(stack, adapter, entry)
        window = adapter.OC.ORIGINAL
        window.bind(stack, None, replace)
        assert native.LAST_FRAME == 40000
        assert receipt['before_last'] == 40000
